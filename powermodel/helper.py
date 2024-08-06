import os
import numpy as np

import torch


def cycle(dl):
    while True:
        for data in dl:
            yield data   
            

def get_file_list(directory_path):
    try:
        # Get the list of files in the specified directory
        file_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        return file_list
    except OSError as e:
        # Handle any errors that might occur while accessing the directory
        return []



def load_model(model, path, specific_path=None):
    model = model
    
    if specific_path:
        cpt = torch.load(specific_path)
        model.load_state_dict(cpt['model'])
        print(f"get model weight from {specific_path}")
    else:
        files = os.listdir(path)
        steps = [int(file.split('_')[-1]) for file in files]
        steps = np.array(steps)
        max_id = np.argmax(steps)
        name = files[max_id]
        
        try:
            cpt = torch.load(os.path.join(path, name))
            model.load_state_dict(cpt['model'])
            print(f"get model weight from the latest. {os.path.join(path, name)}")
        except:
            print("no saved file. run model without saved weight")
    
    return model
    
    

# gather multiples of npz files
def gather_data(dirpath):
    files = get_file_list(dirpath)
    
    concated_data = None
    for i, file in enumerate(files):
        path = os.path.join(dirpath, file)
        data = np.load(path)[:,25:-25]
        if concated_data is None:
            concated_data = data
        else:
            concated_data = np.concatenate([concated_data, data], axis=1)
        
    return concated_data


# find strange value
def trim_data(data):
    print("before : ", data.shape)
    
    # find negative values and exclude them
    all_id = np.ones(data.shape[-1], dtype=bool)
    neg_id = np.where(data < 0)[1]
    
    if neg_id.shape != (0,):
        print("negative time step excluded. timesteps at : ", neg_id)
    
    all_id[neg_id] = False
    data = data[:,all_id]
    
    print("after : ", data.shape)
    
    return data


# get min and max of events
def get_min_max(data):
    v_min = np.min(data, axis=1)
    v_max = np.max(data, axis=1)
    e_len = len(v_min)
    for i in range(e_len):
        print(v_min[i], v_max[i])
    
    return