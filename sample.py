import argparse
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model.autoencoder.autoencoder import AutoEncoderKL

from model.unet.unets import UNet1DConditionModel

from model.clip.clip import CLIP_XAV

from utils.utils_model import load_clip, get_latest_file, get_latest_file_clip, get_latest_file_ldm, load_config
from utils.utils import save_data, MovingAverage

from dataset import l2_data_gather, L2_topdown_dataset, get_data, get_data_test

from diffusers import PNDMScheduler, DDPMScheduler, LCMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr

from accelerate import Accelerator, DistributedType

from model.stable import StableDiff1d


def parse_args():
    parser = argparse.ArgumentParser()

    # other configuration
    parser.add_argument('--cfgpath', type=str, default="./cfg", help='json file for model specification')
    parser.add_argument('--outdir', type=str, default="./weight/ldm", help="outfile for model weight")
    parser.add_argument('--batchsize', type=int, default=16, help="batchsize")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    # savefile
    parser.add_argument('--savedir', type=str, default=None, help="outfile for model weight")
    
    # weight path
    parser.add_argument("--kl_weight", type=str, default=None, help="weight path for autoencoderkl.")
    parser.add_argument("--clip_weight", type=str, default=None, help="weight path for CLIPmodel.")
    parser.add_argument("--ldm_weight", type=str, default=None, help="weight path for Unet.")
    
    # for adam optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    
    # noise
    parser.add_argument("--input_perturbation", type=float, default=0.1, help="recommended 0.1")
    
    # ema
    parser.add_argument("--use_ema", default=False, help="Whether to use EMA model.")
    
    # lr scheduler
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'), )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    
    parser.add_argument("--checkpointing_steps", type=int, default=2000, help="save steps.")
    
    # whether to choose lcm, ldm and inpaint
    parser.add_argument("--is_inpaint", default=False, help="save steps.")
    parser.add_argument("--is_lcm", default=False, help="lcm or ldm")
    
    args = parser.parse_args()
    
    
    # for save file
    # if args.savedir is None:
    #     if args.is_inpaint:
    #         if args.is_lcm:
    #             args.savedir = "./sampleplot_lcm_inpaint"
    #         else:
    #             args.savedir = "./sampleplot_inpaint"
    #     else:
    #         if args.is_lcm:
    #             args.savedir = "./sampleplot_lcm"
    #         else:
    #             args.savedir = "./sampleplot"
        
    # # for weight file
    # if args.kl_weight == None:
    #     args.kl_weight = get_latest_file("./weight/autoencoder")
    # if args.clip_weight == None:
    #     args.clip_weight = get_latest_file_clip("./weight/clip")
    # if args.ldm_weight == None:
    #     if args.is_inpaint:
    #         if args.is_lcm:
    #             args.ldm_weight = get_latest_file_ldm("./weight/lcm_inpaint")
    #         else:
    #             args.ldm_weight = get_latest_file_ldm("./weight/ldm_inpaint")
    #     else:
    #         if args.is_lcm:
    #             args.ldm_weight = get_latest_file_ldm("./weight/lcm")
    #         else:
    #             args.ldm_weight = get_latest_file_ldm("./weight/ldm")
    args.savedir = "sampleplot_metricdoublemean"
    args.kl_weight = "/ssd/ssd3/ljy/zd2/weight/autoencoder/epoch_128"
    args.clip_weight = "/ssd/ssd3/ljy/zd2/weight/meanclip/checkpoint_127_120832.pt"
    args.ldm_weight = "/ssd/ssd3/ljy/zd2/weight/meanldm/checkpoint-120832"
    # args.ldm_weight = "/ssd/ssd3/ljy/zd2/weight/meanldm_inpaint/checkpoint-120832"
    
    os.makedirs(args.savedir, exist_ok=True)
        
    return args




def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def model_size_in_bytes(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    return param_size





if __name__ == "__main__":
    
    
    # set seed if needed
    from accelerate.utils import set_seed
    set_seed(50)
    
    
    
    args = parse_args()
    
    print("bring weight from : ", args.ldm_weight)
    
    if args.is_inpaint:
        cfg = load_config(os.path.join(args.cfgpath, "stable_inpaint.json"))
    else:
        cfg = load_config(os.path.join(args.cfgpath, "stable_normal.json"))
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    # traindata setting
    merged_metric, merged_event, l2_dataset = get_data(cfg.length, mean=False)
    print("total len : ", merged_metric.shape[0])
    train_dataloader = DataLoader(l2_dataset, batch_size=args.batchsize, shuffle=True)
    
    ''' later addd if needed'''    
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
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
    
    from model.clip.temp_module import CLIP_MM
    
    # clip, this is my implementation.
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
    
    # noise scheduler & set timestep (default = 1000)
    if args.is_lcm:
        noise_scheduler = LCMScheduler(prediction_type="v_prediction") # for LCM scheduler    
    else:
        noise_scheduler = DDPMScheduler(prediction_type="v_prediction") # for ddpm scheduler
    
    # Unet
    unet = UNet1DConditionModel(
        in_channels=cfg.unet_channels, 
        out_channels=cfg.latent_channels, 
        cross_attention_dim=128 # cross_attention_dim have to be controlled
        ) 
    unet.cuda()
    unet.eval()
    
    
    # # autoencoder
    # autopara = count_parameters(autoencoder)
    # autosize = model_size_in_bytes(autoencoder)
    
    # # clip
    # clippara = count_parameters(text_encoder)
    # clipsize = model_size_in_bytes(text_encoder)
    
    # # uset
    # unetpara = count_parameters(unet)
    # unetsize = model_size_in_bytes(unet)
    
    
    # print(f"autoencoder : N_para {autopara}, Size {autosize}")
    # print(f"CLIP : N_para {clippara}, Size {clipsize}")
    # print(f"Unet : N_para {unetpara}, Size {unetsize}")
    
    
    # exit()
    
    
    
    

    #temp 
    saved_weight = torch.load(os.path.join(args.ldm_weight, "pytorch_model.bin"))
    unet.load_state_dict(saved_weight)
    
    
    print("##########################################")
    print("load clip weight from : ", args.clip_weight)
    print("load AutoKL weight from : ", args.kl_weight)
    print("load ldm weight from : ", args.ldm_weight)
    print("##########################################")
    
    
    # some condition
    guidance_scale = 1
    guidance_rescale = 0
    clip_skip = 0
    cross_attention_kwargs = 0
    
    stablediffu = StableDiff1d(autoencoder=autoencoder, textencoder=text_encoder, unet=unet, scheduler=noise_scheduler, is_inpaint=args.is_inpaint, cfg=cfg, args=args)
    stablediffu.eval()
    
    metric_save = []
    for events, metrics, in train_dataloader:
        events = torch.Tensor(events).float().to(args.device)
        metrics = torch.Tensor(metrics).float().to(args.device) 
        b, c, w = events.shape
        
                
        
        # # ##### case1 : normal generation
        
        if (args.is_lcm == False) and (args.is_inpaint == False):
            # path = "/ssd/ssd3/ljy/zd/weight/ldm_real"
            # all_items = os.listdir(path)
            # dir_list = [item for item in all_items if os.path.isdir(os.path.join(path, item))]
            
            # for dirname in dir_list:
            #     w_path = os.path.join(path, dirname)
            #     saved_weight = torch.load(os.path.join(w_path, "pytorch_model.bin"))
            #     unet.load_state_dict(saved_weight)
                
            #     print(f"infer {w_path}")
                
            #     # test with no guidance
            #     guidance_scale = 1
            #     gen = stablediffu(metrics, guidance_scale=guidance_scale)[0] # as return is (output, )
            #     gen_event = gen.cpu().detach().numpy()
            #     gen_event = (gen_event + 1) / 2
            #     gen_event[gen_event > 1] = 1
            #     gen_event[gen_event < 0] = 0
                
            #     savedir = os.path.join(args.savedir, dirname)
            #     os.makedirs(savedir, exist_ok=True)
            #     save_data(gen_event, savedir, "sample_", f"cond{guidance_scale}_", True)
            
            print("run ldm normal generation")
            
            # original image
            original_event = events.cpu().detach().numpy()
            save_data(original_event, args.savedir, "sample_", "origin", True)
            
            # test with no guidance
            guidance_scale = 1
            gen = stablediffu(metrics, guidance_scale=guidance_scale)[0] # as return is (output, )
            gen_event = gen.cpu().detach().numpy()
            gen_event = (gen_event + 1) / 2
            gen_event[gen_event > 1] = 1
            gen_event[gen_event < 0] = 0
            save_data(gen_event, args.savedir, "sample_", f"cond{guidance_scale}_", True)
            
            
            metric_save.append(metrics.cpu().detach().numpy())
            metric_save = np.array(metric_save)
            
            freq_path = os.path.join(args.savedir, "metric.npy")
            np.save(freq_path, metric_save,)
        
        
        ##### lcm normal generation
        
        if (args.is_lcm == True) and (args.is_inpaint == False):
            
            print("run lcm normal generation")
            
            # original image
            original_event = events.cpu().detach().numpy()
            save_data(original_event, args.savedir, "sample_lcm_", "origin", True)
            
            # test with no guidance
            guidance_scale = 1
            gen = stablediffu(metrics, guidance_scale=guidance_scale, num_inference_steps=50)[0] 
            gen_event = gen.cpu().detach().numpy()
            gen_event = (gen_event + 1) / 2
            gen_event[gen_event > 1] = 1
            gen_event[gen_event < 0] = 0
            save_data(gen_event, args.savedir, "sample_lcm_", f"cond{guidance_scale}_", True)
            
            
            metric_save.append(metrics[:,-1,:].cpu().detach().numpy())
            metric_save = np.array(metric_save)
            
            savefile = os.path.join(args.savedir, "metric.npy")
            np.save(savefile, metric_save,)
        
        
        
        #### case2 : inpaint generation
        
        if (args.is_lcm == False) and (args.is_inpaint == True):
            
            print("run ldm inpaint generation")
            
            # original image
            original_event = events.cpu().detach().numpy()
            save_data(original_event, args.savedir, "sample_inpaint_", "origin", True)
            mask_condition = torch.zeros((b, 1, w)).to(args.device)
            mask_condition[:,:, cfg.length//2:] = 1
            mask_condition = torch.tensor(mask_condition).to(args.device)
            
            # test inpaint with no guidance
            guidance_scale = 1
            gen = stablediffu(metrics, events=events, mask_condition=mask_condition, guidance_scale=guidance_scale, num_inference_steps=50)[0] # as return is (output, )
            gen_event = gen.cpu().detach().numpy()
            gen_event = (gen_event + 1) / 2
            gen_event[gen_event > 1] = 1
            gen_event[gen_event < 0] = 0
            save_data(gen_event, args.savedir, "sample_inpaint_", f"cond{guidance_scale}_", True)

            
            metric_save.append(metrics.cpu().detach().numpy())
            metric_save = np.array(metric_save)
            
            freq_path = os.path.join(args.savedir, "metric.npy")
            np.save(freq_path, metric_save)
            
            
        
        if (args.is_lcm == True) and (args.is_inpaint == True):
            
            print("run lcm inpaint generation")
            
            # original image
            original_event = events.cpu().detach().numpy()
            save_data(original_event, args.savedir, "sample_inpaint_", "origin", True)
            mask_condition = torch.zeros((b, 1, w)).to(args.device)
            mask_condition[:,:, cfg.length//2:] = 1
            mask_condition = torch.tensor(mask_condition).to(args.device)
            
            # test inpaint with no guidancelssls
            guidance_scale = 1
            gen = stablediffu(metrics, events=events, mask_condition=mask_condition, guidance_scale=guidance_scale)[0] # as return is (output, )
            gen_event = gen.cpu().detach().numpy()
            gen_event = (gen_event + 1) / 2
            gen_event[gen_event > 1] = 1
            gen_event[gen_event < 0] = 0
            save_data(gen_event, args.savedir, "sample_inpaint_", f"cond{guidance_scale}_", True)

            
            metric_save.append(metrics[:,-1,:].cpu().detach().numpy())
            metric_save = np.array(metric_save)
            
            freq_path = os.path.join(args.savedir, "metric.npy")
            np.save(freq_path, metric_save)
        
        
        
        
        
        
        # ##### case3 : inpaint generation with only freq provided
        
        # args.savedir = "./sampleplot_inpaint_one"
        
        # # original image
        # original_event = events.cpu().detach().numpy()
        # save_data(original_event, args.savedir, "sample_inpaint_", "origin", True)
        # mask_condition = torch.zeros((b, 1, w)).to(args.device)
        # mask_condition[:,:, cfg.length//2:] = 1
        # mask_condition = torch.tensor(mask_condition).to(args.device)
        
        
        # # test inpaint with no guidance
        
        # metric_mask = torch.zeros_like(metrics)
        # # metric_mask[:, -1, :] = 1 # only freq is not masked
        # metric_mask[:, 4, :] = 1 # only frontendbound is not masked
        # metric_mask[:, 5, :] = 1 # only frontendbound is not masked
        # metrics = metrics * metric_mask
        
        # guidance_scale = 1
        # gen = stablediffu(metrics, events=events, mask_condition=mask_condition, guidance_scale=guidance_scale)[0] # as return is (output, )
        # gen_event = gen.cpu().detach().numpy()
        # gen_event = (gen_event + 1) / 2
        # gen_event[gen_event > 1] = 1
        # gen_event[gen_event < 0] = 0
        # save_data(gen_event, args.savedir, "sample_inpaint_", f"cond{guidance_scale}_", True)

        
        # metric_save.append(metrics[:,-1,:].cpu().detach().numpy())
        # metric_save = np.array(metric_save)
        
        # np.save("./metric.npy", metric_save,)
        
        break
    

    print("done data generation !!!!")
    print("done data generation !!!!")
    print("done data generation !!!!")
        
        
        
        

                    
            