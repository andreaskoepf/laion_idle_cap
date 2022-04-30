import os
import io
import math
import random
import time
import multiprocessing as mp
import json
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


def c_h(n_gpu):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(n_gpu)

    '''
	os.system("git clone --branch caption-gen https://github.com/LAION-AI/crawlingathome/")
	os.system("pip install -r ./crawlingathome/requirements.txt")
	os.system("pip install webdataset")

	os.system("git clone https://github.com/LAION-AI/BLIP")
	os.system("pip install -r ./BLIP/requirements.txt")
	os.system("pip install --upgrade simplet5")
	os.system("pip install clip-anytorch")



	os.system("pip install --upgrade --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113")

	os.system("wget --no-check-certificate https://captions.christoph-schuhmann.de/spirit/simplet5-epoch-2-train-loss-0.2212-val-loss-0.2188.zip")
	os.system("unzip simplet5-epoch-2-train-loss-0.2212-val-loss-0.2188.zip")

	'''

    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    import webdataset as wds
    import clip
    from simplet5 import SimpleT5
    import crawlingathome as cah
    from sampling import clip_rank, load_blip_decoder, sample, blip_rank, load_blip_ranking_model

    client = cah.init(
        url="http://cah.io.community/",
        # cuda:0 at machine1 (I'd recommend zero-padding the gpu number if gpus > 9).
        device_id="machine1:0"
    )                          # You could also specify the worker number, i.e. `machine1:0.0` if you use 2 workers (to download in the background whilst processing another job)

    print('cuda available:', torch.cuda.is_available())

    force_eos_prob = 0.17937666742679637
    num_sampling_runs = 1
    mode = "ViT-L/14@336px"  # "CLIP-ViT-L+RN50x64"

    top_k = 9607
    set_min_len = 5
    image_size = 224
    set_max_len = None
    set_top_p = 0.2427084604681384
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])

    seed = 1257
    torch.manual_seed(seed)

    random.seed(seed)

    if torch.cuda.is_available():
        torch.device('cuda')
        device0 = torch.device('cuda')
        device1 = torch.device('cuda')
    else:
        torch.device('cpu')
        device0 = torch.device('cpu')
        device1 = torch.device('cpu')
    #print("device0" )
    print(device0)
    if mode == 'CLIP-ViT-L+RN50x64':
        clip_model_name1 = "ViT-L/14"
        clip_model_name2 = "RN50x64"
        print('loading CLIP: ', clip_model_name1)
        clip_model1, clip_preprocess1 = clip.load(
            clip_model_name1, device=device1)
        print('loading CLIP: ', clip_model_name2)
        clip_model2, clip_preprocess2 = clip.load(
            clip_model_name2, device=device0)
    elif mode == 'CLIP-ViT-L':
        clip_model_name1 = "ViT-L/14"
        print('loading CLIP: ', clip_model_name1)
        clip_model1, clip_preprocess1 = clip.load(
            clip_model_name1, device=device1)
    elif mode == 'ViT-L/14@336px':
        clip_model_name1 = "ViT-L/14@336px"
        print('loading CLIP: ', clip_model_name1)
        clip_model1, clip_preprocess1 = clip.load(
            clip_model_name1, device=device1)
    elif mode == 'CLIP-RN50x64':
        clip_model_name1 = "RN50x64"
        print('loading CLIP: ', clip_model_name1)
        clip_model1, clip_preprocess1 = clip.load(
            clip_model_name1, device=device1)
    elif mode == 'ITC' or mode == 'ITM':
        blip_ranking_model = load_blip_ranking_model(device0)
    else:
        raise RuntimeError(f'Unsupported mode "{mode}"')

    model_T5 = SimpleT5()
    if torch.cuda.is_available():
        # outtest/simplet5-epoch-2-train-loss-0.1273-val-loss-0.1379", use_gpu=False) #repair1/simplet5-epoch-1-train-loss-0.2125-val-loss-0.2048", use_gpu=True)
        model_T5.load_model(
            "t5", "./simplet5-epoch-2-train-loss-0.2212-val-loss-0.2188", use_gpu=True)
    else:
        # outtest/simplet5-epoch-2-train-loss-0.1273-val-loss-0.1379", use_gpu=False) #repair1/simplet5-epoch-1-train-loss-0.2125-val-loss-0.2048", use_gpu=True)
        model_T5.load_model(
            "t5", "./simplet5-epoch-2-train-loss-0.2212-val-loss-0.2188", use_gpu=False)

    model, transform = load_blip_decoder(device1)
    all_sims = []
    winner_sims = []

    def make_caption(raw_image):
        image = transform(raw_image).unsqueeze(0).to(device1)

        if set_top_p is not None:
            top_p = torch.tensor([set_top_p]*40, device=device1)
        else:
            top_p = torch.tensor(
                ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]*5), device=device1)

        if set_min_len is not None:
            min_len = torch.tensor(([set_min_len]*40), device=device1)
        else:
            min_len = torch.tensor(
                ([5]*8 + [10]*8 + [15]*8 + [20]*8 + [30]*8), device=device1)

        if set_max_len is not None:
            max_len = torch.tensor(([set_max_len]*40), device=device1)
        else:
            max_len = torch.tensor(
                ([20]*8 + [30]*8 + [30]*8 + [45]*8 + [45]*8), device=device1)

        captions, _, _ = sample(
            image,
            model,
            sample_count=min_len.size(0),
            top_p=top_p,
            top_k=top_k,
            min_len=min_len,
            max_len=max_len,
            force_eos_log_prob=math.log(force_eos_prob),
            prompt='a picture of ',
            num_runs=num_sampling_runs)

        if mode == 'CLIP-ViT-L+RN50x64':
            sims = clip_rank(device1, clip_model1,
                             clip_preprocess1, raw_image, captions)
            top_indices = np.argsort(np.asarray(sims))[-5:][::-1]
            best_captions = [captions[i] for i in top_indices]
            sims2 = clip_rank(device0, clip_model2,
                              clip_preprocess2, raw_image, best_captions)
            best_index = np.argmax(np.asarray(sims2))
            winner_sims.append(sims2[best_index])
            synth_caption = best_captions[best_index]

        elif mode == 'CLIP-ViT-L' or mode == 'CLIP-RN50x64' or mode == 'ViT-L/14@336px':
            sims = clip_rank(device1, clip_model1,
                             clip_preprocess1, raw_image, captions)
            best_index = np.argmax(np.asarray(sims))
            winner_sims.append(sims[best_index])
            synth_caption = captions[best_index]
        elif mode == 'ITC' or mode == 'ITM':
            sims = blip_rank(device0, blip_ranking_model,
                             raw_image, captions, mode=mode.lower())
            best_index = np.argmax(np.asarray(sims))
            winner_sims.append(sims[best_index])
            synth_caption = captions[best_index]

        #all_sims = all_sims + sims

        #print('synth: ', synth_caption)
        synth_caption = model_T5.predict(synth_caption)[0]  # synth_caption

        return synth_caption, captions, sims

    def upload(file):
        result = os.system(f"rsync -av {file} deploy.laion.ai::spirit ")
        return result

    while client.jobCount() > 0 and not client.shouldDie():
        client.newJob()
        print(str(client.tar_url))
        # str(client.tar_url).split("/")[-1] + " -"   #client.tar_url
        url = "pipe:aws s3 cp " + str(client.tar_url) + " -"
        print(url)
        upload_address = client.upload_address
        print(upload_address)
        client.log("Processing...")
        # work on the data

        try:
            os.mkdir("./c_h/")
        except:
            pass

        # dataset = wds.WebDataset("dataset.tar.gz")
        dataset = wds.WebDataset(url)
        captioning_results = {}
        for i, d in enumerate(dataset):
            # print(i)
            # print(d)
            start = time.time()
            try:
                raw_image = Image.open(io.BytesIO(d['jpg'])).convert('RGB')
                # raw_image.save("./c_h/Test"+str(i)+".jpg")
                winner_cap, all_captions, all_sims = make_caption(raw_image)
            except:
                continue

            captioning_result = {}
            captioning_result["winner_cap"] = winner_cap
            captioning_result["all_captions"] = all_captions
            captioning_result["all_similarties"] = all_sims

            captioning_results[str(i)] = captioning_result
            #print( captioning_results )
            print(winner_cap)

            print(time.time()-start)
            # break

        with open('./c_h/captioning_result_'+url.split("/")[-1].split(".")[0]+'.json', 'w') as fp:
            json.dump(captioning_result, fp)

        for abc in range(100):
            resp = upload('./c_h/captioning_result_' +
                          url.split("/")[-1].split(".")[0]+'.json')
            if resp == 5888:
                print('error while uploading')
            elif resp == 0:
                print('upload successful')
                break
            else:
                print('unknown upload error')
            time.sleep(5)

        client.completeJob()  # - server is live so avoid actually calling this ;)

    client.bye()  # disconnects the worker from the machine - either there are no more jobs or the client has been flagged to be killed by the server (`client.shouldDie()`)


def main():
    #mp.set_start_method('spawn')

    #n_gpus = 8
    visible_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    jobs = []
    j = 0
    while True:
        print("starting new batch of 8 tars")
        print('loop:', j)
        j += 1
        for i in visible_gpus:  # range(n_gpus):
            print('launching worker-process:', i)
            p = mp.Process(target=c_h, kwargs=dict(n_gpu=i))
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()


if __name__ == '__main__':
    main()
