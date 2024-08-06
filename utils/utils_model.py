import os
import json
from glob import glob
from types import SimpleNamespace
import numpy as np
import random

import torch






# load model from accelerate checkpoint
def load_model(path, model, optimizer=None, scheduler=None,):
    SCALER_NAME = "scaler.pt"
    MODEL_NAME = "pytorch_model"
    RNG_STATE_NAME = "random_states"
    OPTIMIZER_NAME = "optimizer"
    SCHEDULER_NAME = "scheduler"
    
    # load weight
    weights_name = f"{MODEL_NAME}.bin" 
    input_model_file = os.path.join(path, weights_name)
    model.load_state_dict(torch.load(input_model_file, )) 
    
    # optimizer_name = f"{OPTIMIZER_NAME}.bin"
    # input_optimizer_file = os.path.join(path, optimizer_name)
    # optimizer_state = torch.load(input_optimizer_file)
    # optimizer.load_state_dict(optimizer_state)
    
    # scheduler_name = f"{SCHEDULER_NAME}.bin"
    # input_scheduler_file = os.path.join(path, scheduler_name)
    # scheduler.load_state_dict(torch.load(input_scheduler_file))
    
    try:
        states = torch.load(os.path.join(path, f"{RNG_STATE_NAME}_{0}.pkl"))
        random.setstate(states["random_state"])
        np.random.set_state(states["numpy_random_seed"])
        torch.set_rng_state(states["torch_manual_seed"])
        torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
    except:
        print("there is no random state in checkpoint. skip seed setting")
    
    return model, optimizer, scheduler



def load_config(cfgpath):
    cfg = {}

    with open(cfgpath, 'r') as f:
        load_cfg = json.load(f)
    cfg.update(load_cfg)

    return SimpleNamespace(**cfg)


def get_latest_file(path, ):
    filenames = list(glob(os.path.join(path, "epoch_*")))
    steps = [int(x.replace('.pt', '').split('_')[-1]) for x in filenames]
    
    return filenames[np.argmax(steps)]


def get_latest_file_clip(path, ):
    filenames = list(glob(os.path.join(path, "checkpoint_*")))
    steps = [int(x.replace('.pt', '').split('_')[-1]) for x in filenames]
    
    return filenames[np.argmax(steps)]

def get_latest_file_ldm(path, ):
    filenames = list(glob(os.path.join(path, "checkpoint-*")))
    steps = [int(x.split('-')[-1]) for x in filenames]
    
    return filenames[np.argmax(steps)]



def load_clip(model, path, device=None):
    # Find the latest chip model
    filenames = list(glob(path, ))
    steps = [int(x.replace(".pt", "").split('_')[-1]) for x in filenames]
    clippath = filenames[np.argmax(steps)]
    
    # Load the weights
    checkpoint = torch.load(clippath, map_location=device)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    
    return model