import torch
import clip
from sampling import load_blip_decoder, load_blip_ranking_model


def main():
    device = torch.device('cpu')
    clip_model_name1 = "ViT-L/14@336px"
    
    print('loading CLIP: ', clip_model_name1)
    clip_model1, clip_preprocess1 = clip.load(clip_model_name1, device=device)
    print('ok')

    print('loading BLIP')
    model, transform = load_blip_decoder(device)
    print('ok')

    #print('loading BLIP ranking model')
    #blip_ranking_model = load_blip_ranking_model(device)
    #print('ok')


if __name__ == '__main__':
    main()
