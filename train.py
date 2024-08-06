import argparse
import numpy as np
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model.autoencoder.autoencoder import AutoEncoderKL

from dataset import get_data, L2_EVENT_MIN_MAX

from utils.logger import init_sLog, sLog

from utils.utils import save_data, str2bool, get_surrogate
from utils.utils_model import load_config

from policynet import DQN_FC

from powermodel.model import PowerNN
from itertools import cycle





def get_latest_powermodel_path(path):
    files = os.listdir(path)
    steps = [int(file.split('_')[-1]) for file in files]
    steps = np.array(steps)
    max_id = np.argmax(steps)
    name = files[max_id]
    print(f"get model weight from the latest. {os.path.join(path, name)}")
    return os.path.join(path, name)



 
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfgpath', type=str, default="./cfg", help='json file for model specification')
    parser.add_argument('--outdir', type=str, default="./weight/autoencoder", help="outfile for model weight")
    parser.add_argument('--epoch', type=int, default=128, help="num of iterations")
    parser.add_argument('--batchsize', type=int, default=256, help="num of iterations")
    parser.add_argument('--lr', type=float, default=0.01, help="num of iterations")
    parser.add_argument('--save_epoch', type=int, default=16, help="save interval for model")
    parser.add_argument('--save_loss', type=int, default=100, help="save interval for loss")
    parser.add_argument('--loadweight', type=str, default=None, help="save interval for loss")
    
    parser.add_argument('--iteration', type=int, default=10000, help="total iteration")
    
    parser.add_argument('--length', type=int, default=128, help="length")
    parser.add_argument("--use_sw_cls", type=str2bool, default=False, )
    
    parser.add_argument("--train", type=str2bool, default=True, )
    parser.add_argument("--val", type=str2bool, default=False, )
    
    
    parser.add_argument("--test", type=str2bool, default=False, )
    
    args = parser.parse_args()
    
    # for log
    # args.logfile = "autoencoder.log"
    # args.outdir = "./weight/autoencoder"
    args.savedir = "./weight/policy"
    
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.savedir, exist_ok=True)
    
    args.kl_weight = "./weight/autoencoder/epoch_128"
    args.clip_weight = "./weight/meanclip/checkpoint_127_120832.pt"
    args.ldm_weight = "./weight/meanldm/checkpoint-120832"
    
    args.logfile = "loss2.log"
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"




class DIFFenv():
    def __init__(self, surrogate, power, policy, optim, dl, args):
        self.surrogate = surrogate
        self.powermodel = power
        self.policy = policy
        self.optim = optim
        self.device = args.device
        
        self.dl = dl
        self.dl = cycle(self.dl)
        
        
        self.v_min = np.array([L2_EVENT_MIN_MAX[key][0] for key in L2_EVENT_MIN_MAX.keys()])
        self.v_max = np.array([L2_EVENT_MIN_MAX[key][1] for key in L2_EVENT_MIN_MAX.keys()])
        
        self.g_step = 0
        
        
        self.metric, self.event = None, None
        
        
    def compute_action(self, state):
        with torch.no_grad():
            self.policy.eval()
            return self.policy(state).max(1)[1].reshape(-1)
    
    
    def action_to_freq(self, action):
        # return normed freq
        freq = (10 + action) / 10
        freq = (freq - 1.0) / (3.9 - 1.0) 
        freq = freq * 100
        
        return freq
    
    
    def set_metric(self, freq):
        for b in range(freq.shape[0]):
            self.metric[b,8].fill_(freq[b])
            self.metric[b,17].fill_(freq[b])
    
    
    def infer(self, ):
        self.metric = self.metric.float().to(self.device)
        guidance_scale = 1
        new_data = self.surrogate(self.metric, guidance_scale=guidance_scale, num_inference_steps=50)[0] 
        new_data = (new_data + 1) / 2
        new_data[new_data > 1] = 1
        new_data[new_data < 0] = 0
        
        return new_data
        
        
    def state_to_input(self, state):
        inp = state.mean(dim=-1)
        if isinstance(inp, torch.Tensor):
            inp = torch.tensor(inp, dtype=torch.float).to(self.device)
        return inp
        
        
    def calculate_power(self, state):
        power = self.powermodel(state).squeeze(-1)
        power = (80 - 23) * power + 23 
        power = power / 10
        return power
    
    
    def get_effcycle(self, state):
        cycle = (self.v_max[1] + self.v_min[1]) * state[:, 1] + self.v_min[1]
        stall =  (self.v_max[13] + self.v_min[13]) * state[:, 13] + self.v_min[13]
        effcycle = (cycle - stall) / cycle
        
        return effcycle
        
    
    def calculate_reward(self, state):
        power = self.calculate_power(state).squeeze(-1)
        effcycle = self.get_effcycle(state)
        reward = effcycle / power
        
        return reward
        
    
    def train(self, ):
        self.event, self.metric = next(self.dl)
        inp = self.state_to_input(self.event)
        actions = self.compute_action(inp)
        freq = self.action_to_freq(actions)
        self.set_metric(freq)
        new_event = self.infer()
        newdata_inp = self.state_to_input(new_event)
        
        prev_reward = self.calculate_reward(inp)
        next_reward = self.calculate_reward(newdata_inp)
        loss = next_reward.mean()
        
        # difference = next_reward - prev_reward
        # penalty = torch.relu(prev_reward - next_reward)
        # loss = -difference.mean() + penalty.mean()
        # loss = F.smooth_l1_loss(next_reward, prev_reward)
        if self.g_step % 10:
            print(loss.item())
        sLog().d("loss: {:.4f}".format(loss.item()))
        
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        
        self.g_step += 1
        
        
        
        
        


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="3"


if __name__ == "__main__":    
    N_ACTION = 30
    
    args = parse_args()
    cfg = load_config(os.path.join(args.cfgpath, "stable_normal.json"))
    
    init_sLog(os.path.join(args.cfgpath, args.logfile))
    
    # for experiment
    random_seed = 50
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(random_seed)

    # surrogate
    diffmodel = get_surrogate(cfg, args)
    diffmodel.eval()

    # for seed data
    merged_metric, merged_event, l2_dataset = get_data(cfg.length, mean=False)
    train_dataloader = DataLoader(l2_dataset, batch_size=args.batchsize, shuffle=True)
    
    dl = cycle(train_dataloader)
    
    # policy network 
    policynet = DQN_FC(in_dim=15, n_actions=N_ACTION).to(args.device)
    # policy_weight_path = "/ssd/ssd3/ljy/zd1000/weight/policy_twoepoch/po_model_state_942.pt"
    # policynet.load_state_dict(torch.load(policy_weight_path, map_location=torch.device(args.device)))
    policynet.train()
    
    # powermodel
    power_weight_path = get_latest_powermodel_path("./powermodel/data/modeldata")
    powermodel = PowerNN().to(args.device)
    powermodel.load_state_dict(torch.load(power_weight_path, map_location=torch.device(args.device))["model"])
    powermodel.eval()
    
    # optimizer
    import torch.optim as optim
    optimizer = optim.Adam(policynet.parameters(), lr=args.lr)
    
    # trainenv
    env = DIFFenv(surrogate=diffmodel,
                  power=powermodel,
                  policy=policynet,
                  optim=optimizer,
                  dl=train_dataloader,
                  args=args)
    
    
    for i in range(args.iteration):
        env.train()
        
        if i % 100:
            torch.save(policynet.state_dict(), os.path.join("weight/policy", f"po_model_state_{i}.pt"))
    
    
    