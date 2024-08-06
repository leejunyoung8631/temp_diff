import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch




def save_data(data, outdir, prefix, tag, plot_verbose = False):
    save_filename = '{}.{}.npy'
    filename = os.path.join(outdir, save_filename.format(prefix, tag))
    np.save(filename, data)
    
    if plot_verbose:
        b, c, t = data.shape
    
        x = np.arange(1, t+1)
        for i in range(b):
            
            plt.figure()
            plt.ylim(-0.1, 1.1)
            for k in range(c):
                plt.plot(x, data[i,k,:])
            
            filename = os.path.join(outdir, save_filename.format(prefix, tag))
            save_figname = filename + "_{}.png"
            filename = save_figname.format(i+1)
            plt.savefig(filename)
            plt.close()



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_latest_powermodel_path(path):
    files = os.listdir(path)
    steps = [int(file.split('_')[-1]) for file in files]
    steps = np.array(steps)
    max_id = np.argmax(steps)
    name = files[max_id]
    print(f"get model weight from the latest. {os.path.join(path, name)}")
    return os.path.join(path, name)


def to_negone_to_one(data):
    return 2*data - 1

def to_zero_to_one(data):
    return (data + 1) / 2



def print_specifics(args):
    print("\n")
    print("device : ", args.device)
    print("iteration : ", args.iterations)
    print("batch_size : ", args.batch_size)
    print("init_buffer : ", args.init_buffer)
    print("replaybuffer_size : ", args.replaybuffer_size)
    print("n_action : ", args.n_action)
    print("\n")
    
    
    

from itertools import combinations_with_replacement
def distribute_apples(total_apples, num_people, min_apples_each):
    # Calculate remaining apples after each person gets the minimum number
    remaining_apples = total_apples - num_people * min_apples_each

    # Generate all combinations of distributing remaining_apples among num_people people
    combinations = combinations_with_replacement(range(remaining_apples + 1), num_people - 1)

    # Convert combinations to distributions
    distributions = []
    for combo in combinations:
        combo = (0,) + combo + (remaining_apples,)
        distribution = [combo[i+1] - combo[i] + min_apples_each for i in range(num_people)]
        if sum(distribution) == total_apples:
            distributions.append(distribution)

    return distributions








from utils.utils_model import load_clip
from model.autoencoder.autoencoder import AutoEncoderKL
from model.clip.clip import CLIP_XAV
from model.unet.unets import UNet1DConditionModel
from model.stable import StableDiff1d
from diffusers import DDPMScheduler

def get_surrogate(cfg, args):
    # autoencoder model
    autoencoder = AutoEncoderKL(in_channels=cfg.in_channels,
                                out_channels=cfg.out_channels,
                                down_block_types=cfg.autoencoder["encoder_block"],
                                up_block_types=cfg.autoencoder["decoder_block"],
                                block_out_channels=cfg.autoencoder["block_out_channels"],
                                layers_per_block=cfg.autoencoder["n_layers_per_block"],
                                latent_channels=cfg.latent_channels,
                                act_fn=cfg.autoencoder["act_fn"],
                                norm_num_groups=cfg.autoencoder["norm_num_groups"],
                                )
    autoencoder.to(args.device)
    model_state_dict = torch.load(args.kl_weight, map_location=args.device)
    autoencoder.load_state_dict(model_state_dict)
    autoencoder.eval()
    
    # clip model
    text_encoder = CLIP_XAV(
                  width=128, # dimension of latent vector
                  layers=4, # the number of ResidualAttension block 
                  heads=4, # the number of head of attension
                  length=cfg.length, # the range of timesteps
                  m_in=cfg.text_channels, # input channel size of metric
                  c_in=cfg.in_channels, # input channel size of counter
                  )
    text_encoder = load_clip(text_encoder, args.clip_weight)
    text_encoder.to(args.device)
    text_encoder.eval()
    
    # unet model
    # Unet
    unet = UNet1DConditionModel(
        in_channels=cfg.unet_channels, 
        out_channels=cfg.latent_channels, 
        cross_attention_dim=128 # cross_attention_dim have to be controlled
        ) 
    unet.cuda()
    unet.eval()

    #temp 
    saved_weight = torch.load(os.path.join(args.ldm_weight, "pytorch_model.bin"))
    unet.load_state_dict(saved_weight)
    
    # scheduler 
    noise_scheduler = DDPMScheduler(prediction_type="v_prediction") # for ddpm scheduler
    
    surrogate = StableDiff1d(autoencoder=autoencoder, textencoder=text_encoder, unet=unet, scheduler=noise_scheduler, is_inpaint=True, cfg=cfg, args=args)
    surrogate.cuda()
    surrogate.eval()
    
    
    return surrogate 