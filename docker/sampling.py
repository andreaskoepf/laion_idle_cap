import math
import time
from typing import Optional
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import clip
import os

print( os.getcwd())
from BLIP.models.blip import blip_decoder
from BLIP.models.blip_itm import blip_itm
# os.chdir("../")
import glob


def cos_sim(a, b, normalize=True):
    if normalize:
        a = a / torch.norm(a, dim=-1, keepdim=True)
        b = b / torch.norm(b, dim=-1, keepdim=True)
    return a @ b.T


@torch.no_grad()
def clip_rank(device, clip_model, preprocess, image_pil, text_list):

    similarities= []
    image = preprocess(image_pil).unsqueeze(0).to(device)

    image_features = clip_model.encode_image(image)

    for txt in text_list:
        text_tokens = clip.tokenize(txt, truncate=True).to(device)
        text_features = clip_model.encode_text(text_tokens)
        s = cos_sim(text_features, image_features).item()
        similarities.append(s)

    return similarities


@torch.no_grad()
def blip_rank(device, model_blip, image_pil, text_list, image_size=384, mode="itm"):
    similarities= []

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 
    image = transform(image_pil).unsqueeze(0).to(device)   

    for caption in text_list:
        if mode == 'itm':
            itm_output = model_blip(image, caption, match_head='itm')
            itm_score = F.softmax(itm_output, dim=1)[:,1]
            similarities.append(itm_score.item())
        elif mode == 'itc':
            itc_score = model_blip(image, caption, match_head='itc')
            similarities.append(itc_score.item())
        else:
            raise RuntimeError(f'blip ranking mode "{mode}" not supported')

    return similarities


def repetition_penalty_apply(logits, tokens, penalty):
    tok_logits = torch.gather(logits, -1, tokens)
    tok_logits = torch.where(tok_logits < 0, tok_logits * penalty, tok_logits / penalty)
    logits.scatter_(-1, tokens, tok_logits)
    return logits


def top_k_top_p_filtering_batch(logits, top_k=0, top_p=0.0, filter_value=float('-inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    batch_size = logits.size(0)
    num_logits = logits.size(-1)
    device = logits.device
    #print('top_k', type(top_k), top_k)
    if type(top_k) == float:
        if top_k > 0 and top_k < 1:
            top_k = max(1, int(top_k * num_logits))
        else:
            top_k = int(top_k)
    # Remove all tokens with a probability less than the last token of the top-k
    if type(top_k) == int:
        if top_k > 0:
            cutoff = torch.topk(logits, k=top_k, largest=True).values[:, -1:]
            indices_to_remove = logits < cutoff
            logits[indices_to_remove] = filter_value      
    elif torch.any(top_k > 0):
        assert top_k.size(0) == batch_size
        top_k = top_k.clamp_max(num_logits)
        for i in range(batch_size):
            k = top_k[i] 
            if k <= 0:
                continue
            if k < 1:
                k = max(1, int(k * num_logits))
            cutoff = torch.topk(logits[i], k=k, largest=True).values[-1]
            indices_to_remove = logits[i] < cutoff
            logits[i][indices_to_remove] = filter_value
    if type(top_p) == float and top_p > 0.0 or torch.any(top_p > 0):
        if type(top_p) == torch.Tensor and top_p.size(-1) != 1:
            top_p = top_p.unsqueeze(-1)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        # convert sorted indices into flat indices
        row_starts = torch.arange(sorted_indices.shape[0], device=device).unsqueeze(1) * sorted_indices.shape[1]
        sorted_indices_flat = sorted_indices + row_starts
        indices_to_remove = sorted_indices_flat[sorted_indices_to_remove]
        logits = logits.contiguous()
        logits.view(-1)[indices_to_remove] = filter_value
    return logits


@torch.no_grad()
def generate(
    model,
    inputs: Optional[torch.Tensor],
    encoder_hidden_states,
    encoder_attention_mask,
    eos_token_id,
    top_p,
    top_k,
    min_length,
    max_length,
    repetition_penalty: Optional[float] = None,
    min_alternate_prob=0,
    force_eos_log_prob=math.log(0.9)
):
    # run until max or no candidates remaining
    total_max_length = max_length.max()

    results = []

    eos_probs = torch.empty(inputs.size(0), 0, device=inputs.device)
    #rel_entropies = torch.zeros(inputs.size(0), 0, device=inputs.device)

    for i in range(total_max_length):
        if inputs.size(0) == 0:
            break

        outputs = model.forward(
            input_ids = inputs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
            output_attentions=model.config.output_attentions,
            output_hidden_states=model.config.output_hidden_states,
        )
        logits = outputs['logits']      # [1, 4, 30524]

        last_token_logits = logits[:,-1,:]
        raw_p = F.softmax(last_token_logits, dim=-1)
        min_indices = torch.nonzero(i < min_length).view(-1)
        last_token_logits[min_indices, eos_token_id] = float('-inf')

        if repetition_penalty is not None and repetition_penalty > 0:
            repetition_penalty_apply(last_token_logits, tokens=inputs, penalty=repetition_penalty)

        last_token_logits = top_k_top_p_filtering_batch(last_token_logits, top_p=top_p, top_k=top_k)
        p = F.softmax(last_token_logits, dim=-1)

        # 1. record probabilities of EOS token
        eos_prob = raw_p[:, eos_token_id].log()
        
        # 2. flatness of token probability distribution, e.g. D_KL between token probabilities and uniform distribution
        num_tokens = p.shape[-1]
        #d_kl = torch.sum(raw_p * torch.log(1e-10 + raw_p * num_tokens), dim=-1)
   
        next_token_samples = torch.multinomial(p, 2, replacement=False) # [40, 2]

        next_token = next_token_samples[:, :1]
        completed = torch.logical_or(next_token.squeeze(-1) == eos_token_id, max_length <= i)
        
        stop_at_high_eos = True
        if force_eos_log_prob < 0:
            if stop_at_high_eos:
                completed = torch.logical_or(completed, eos_prob > force_eos_log_prob)
            else:
                hi_eos = torch.logical_and(torch.logical_not(completed), eos_prob > force_eos_log_prob)
                if torch.any(hi_eos):
                    results.append([inputs[hi_eos], min_length[hi_eos], max_length[hi_eos], top_p[hi_eos], eos_probs[hi_eos]]) #, rel_entropies[hi_eos]])

        if torch.any(completed):
            results.append([inputs[completed], min_length[completed], max_length[completed], top_p[completed], eos_probs[completed]]) #, rel_entropies[completed]])

            # check possible replacements            
            if min_alternate_prob > 0:
                potential_continue = torch.logical_and(completed, max_length > i)
                if torch.any(potential_continue):
                    alternate_sample = next_token_samples[:, 1:]        
                    alternate_probs = torch.gather(p, -1, alternate_sample)     # get probability of alternate sample
                    potential_continue = torch.logical_and(potential_continue, alternate_sample.squeeze(-1) != eos_token_id)
                    potential_continue = torch.logical_and(potential_continue, alternate_probs.squeeze(-1) > min_alternate_prob)
                    if torch.any(potential_continue):
                        next_token[potential_continue] = alternate_sample[potential_continue]
                        completed = torch.logical_and(completed, torch.logical_not(potential_continue))

            not_completed = torch.logical_not(completed)
            inputs = inputs[not_completed]
            eos_prob = eos_prob[not_completed]
            #d_kl = d_kl[not_completed]
            eos_probs = eos_probs[not_completed]
            #rel_entropies = rel_entropies[not_completed]
            next_token = next_token[not_completed]
            if type(top_p) == torch.Tensor:
                top_p = top_p[not_completed]
            if type(top_k) == torch.Tensor: 
                top_k = top_k[not_completed]
            min_length = min_length[not_completed]
            max_length = max_length[not_completed]
            encoder_hidden_states = encoder_hidden_states[not_completed]
            encoder_attention_mask = encoder_attention_mask[not_completed]

        #print('eos_probs', eos_probs.shape, eos_prob.shape)
        #print('rel_entropies', rel_entropies.shape, d_kl.shape)
        inputs = torch.cat([inputs, next_token], dim=-1)
        eos_probs = torch.cat([eos_probs, eos_prob.unsqueeze(-1)], dim=-1)
        #rel_entropies = torch.cat([rel_entropies, d_kl.unsqueeze(-1)], dim=-1) 

    if inputs.size(0) > 0:
        results.append([inputs, min_length, max_length, top_p, eos_probs]) #, rel_entropies])

    return results

import random
@torch.no_grad()
def sample(image, blip_model, sample_count=3, top_p=0, top_k=0, min_len=0, max_len=0, repetition_penalty=1.3, force_eos_log_prob=math.log(0.9), prompt='a picture of ', unique=True, num_runs=1):
    batch_size = image.size(0)
    device = image.device
    
    torch.manual_seed(random.randint(0,100000))



    image_embeds = blip_model.visual_encoder(image)

    image_embeds = image_embeds.repeat_interleave(sample_count, dim=0)    
    image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)

    bos_token_id = blip_model.tokenizer.bos_token_id
    eos_token_id = blip_model.tokenizer.sep_token_id
    
    prompt_ = [prompt] * batch_size
    input_ids = blip_model.tokenizer(prompt_, return_tensors="pt").input_ids.to(device)
    input_ids[:,0] = bos_token_id   # replace begin token 
    input_ids = input_ids[:, :-1]   # remove end token
    input_ids = input_ids.repeat_interleave(sample_count, dim=0)
    num_prompt_tokens = input_ids.size(1)

    outputs = []
    for i in range(num_runs):
        outputs = outputs + generate(blip_model.text_decoder, input_ids, image_embeds, image_atts, 
            eos_token_id=eos_token_id,
            top_p=top_p,
            top_k=top_k,
            min_length=min_len,
            max_length=max_len,
            repetition_penalty=repetition_penalty,
            force_eos_log_prob=force_eos_log_prob)

    captions = []
    parameters = []
    stats = []
    for output in outputs:
        for i,o in enumerate(output[0]):
            caption = blip_model.tokenizer.decode(o, skip_special_tokens=True)
            tokens = blip_model.tokenizer.convert_ids_to_tokens(o)
            caption_without_prompt = caption[len(prompt):]
            if unique and caption_without_prompt not in captions:
                captions.append(caption[len(prompt):])  # remove prompt
                parameters.append([output[1][i].item(), output[2][i].item(), output[3][i].item()])
                #stats.append({ 'eos_prob': output[4][i], 'rel_entropy': output[5][i], 'tokens': tokens[num_prompt_tokens:]})
                stats.append({ 'eos_prob': output[4][i], 'tokens': tokens[num_prompt_tokens:]})
    return captions, parameters, stats



def load_blip_decoder(device, image_size=384):
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
    
    model = blip_decoder(pretrained=model_url, image_size=384, vit='large', med_config='/mnt/spirit/c_h/BLIP/configs/med_config.json')
    model.eval()
    model = model.to(device)
    return model, transform


def load_blip_ranking_model(device, image_size=384):
    blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth'
    blip_model = blip_itm(pretrained=blip_model_url, image_size=image_size, vit='large', med_config='/mnt/spirit/c_h/BLIP/configs/med_config.json')
    blip_model.eval()
    blip_model.to(device)
    return blip_model

