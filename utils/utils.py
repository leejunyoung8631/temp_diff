from typing import Any, Callable, Dict, List, Optional, Union
import os
import inspect
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from collections import deque

import torch

import argparse



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



class MovingAverage(object):
    """ Computes and stores the moving average """
    def __init__(self, window=1000):
        self.window = window
        self.reset()

    def reset(self):
        self.vals = deque(maxlen=self.window)

    def update(self, val):
        self.vals.append(val)

    def get(self):
        if len(self.vals) == 0:
            return 0
        return np.mean(self.vals)
    


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




# def check_inputs(self, image, height, width):
#         if (
#             not isinstance(image, torch.Tensor)
#             and not isinstance(image, PIL.Image.Image)
#             and not isinstance(image, list)
#         ):
#             raise ValueError(
#                 "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
#                 f" {type(image)}"
#             )

#         if height % 8 != 0 or width % 8 != 0:
#             raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        
        

def check_inputs(metrics, events, length, encode):
    return








def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


    
    
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
    from model.clip.temp_module import CLIP_MM
    # text_encoder = CLIP_XAV(
    #               width=128, # dimension of latent vector
    #               layers=4, # the number of ResidualAttension block 
    #               heads=4, # the number of head of attension
    #               length=cfg.length, # the range of timesteps
    #               m_in=cfg.text_channels, # input channel size of metric
    #               c_in=cfg.in_channels, # input channel size of counter
    #               )
    text_encoder = CLIP_MM(
                  width=128, # dimension of latent vector
                  layers=4, # the number of ResidualAttension block 
                  heads=4, # the number of head of attension
                  length=cfg.length, # the range of timesteps
                  m_in=cfg.text_channels, # input channel size of metric
                  c_in=cfg.in_channels, # input channel size of counter
                  eos=True
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
    
    surrogate = StableDiff1d(autoencoder=autoencoder, textencoder=text_encoder, unet=unet, scheduler=noise_scheduler, is_inpaint=False, cfg=cfg, args=args)
    surrogate.cuda()
    surrogate.eval()
    
    
    return surrogate 