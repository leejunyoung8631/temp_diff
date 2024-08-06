import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


L2_EVENT_MIN_MAX = {"idq_uops_not_delivered.core":[1100000.0, 1600000000.0],
                    "cpu_clk_unhalted.thread_any":[200000000.0, 780000000.0],
                    "uops_retired.retire_slots":[20000000.0, 3000000000.0],
                    "uops_issued.any":[35000000.0, 2930000000.0],
                    "int_misc.recovery_cycles_any":[20000.0, 120000000.0],
                    "idq_uops_not_delivered.cycles_0_uops_deliv.core":[100000.0, 230000000.0],
                    "uops_retired.macro_fused":[20000.0, 550000000.0],
                    "inst_retired.any":[250000.0, 3000000000.0],
                    "br_misp_retired.all_branches":[250.0, 20000000.0],
                    "machine_clears.count":[150.0, 2400000.0],
                    "exe_activity.2_ports_util":[1000000.0, 400000000.0],
                    "cycle_activity.stalls_mem_any":[100000.0, 688000000.0],
                    "exe_activity.bound_on_stores":[1000.0, 780000000.0],
                    "cycle_activity.stalls_total":[450000.0, 760000000.0],
                    "exe_activity.1_ports_util":[300000.0, 400000000.0]
                    }





class L2_topdown_dataset(Dataset):
    def __init__(self, metric, event, length=128):
        self.metric = metric # timestep, dim
        self.event = event
        self.length = 0
        
        self.v_min = np.array([L2_EVENT_MIN_MAX[key][0] for key in L2_EVENT_MIN_MAX.keys()])[:, np.newaxis] 
        self.v_max = np.array([L2_EVENT_MIN_MAX[key][1] for key in L2_EVENT_MIN_MAX.keys()])[:, np.newaxis]
        
        self.norm()
        self.slcing(length=length)
        
        
    def norm(self ):
        self.metric = self.metric.transpose() # dim, timesteps
        self.event = self.event.transpose() #  dim, teimsteps
        
        # event
        self.event = (self.event - self.v_min) / (self.v_max - self.v_min)
        self.event[self.event < 0] = 0
        self.event[self.event > 1] = 1
        
        # metirc
        self.metric[-1,:] = (self.metric[-1,:] - 1.0) / (3.9 - 1.0) # freq
        self.metric[:-1,:] = self.metric[:-1,:] / 100 # topdowmetricn 
        
        
    def slcing(self, length):
        sliced_metric = []
        sliced_event = []
        for i in range(self.metric.shape[-1] - length + 1):
            sliced_metric.append(self.metric[:, i:i+length])
            sliced_event.append(self.event[:, i:i+length])
            
        
        sliced_metric = np.array(sliced_metric)
        sliced_event = np.array(sliced_event)    
        
        self.metric = sliced_metric
        self.event = sliced_event
        
        self.length = self.event.shape[0]
    
    
    def get_datashape(self, ):
        return self.metric.shape, self.event.shape
        
        
        
    def __len__(self, ):
        return self.length
    
    
    def __getitem__(self, idx):
        return self.event[idx], self.metric[idx]
    






class L2_topdown_dataset_mean(Dataset):
    def __init__(self, metric, event, length=128, second=False):
        self.metric = metric # timestep, dim
        self.event = event
        self.length = 0
        
        self.v_min = np.array([L2_EVENT_MIN_MAX[key][0] for key in L2_EVENT_MIN_MAX.keys()])[:, np.newaxis] 
        self.v_max = np.array([L2_EVENT_MIN_MAX[key][1] for key in L2_EVENT_MIN_MAX.keys()])[:, np.newaxis]
        
        
        self.norm()
        if second:
            self.slcing_second(length=length)
        else:
            self.slcing(length=length)
        
        # self.metric = self.metric / 100
        
        
    def norm(self ):
        self.metric = self.metric.transpose() # dim, timesteps
        self.event = self.event.transpose() #  dim, teimsteps
        
        # event
        self.event = (self.event - self.v_min) / (self.v_max - self.v_min)
        self.event[self.event < 0] = 0
        self.event[self.event > 1] = 1
        
        # metirc
        self.metric[-1,:] = (self.metric[-1,:] - 1.0) / (3.9 - 1.0) # freq
        self.metric[:-1,:] = self.metric[:-1,:] / 100 # topdowmetricn 
        
        self.metric = self.metric * 100
        
        
        
    def slcing(self, length):
        sliced_metric = []
        sliced_event = []
        for i in range(self.metric.shape[-1] - length + 1):
            sliced_metric.append(np.mean(self.metric[:, i:i+length], axis=-1))
            # sliced_event.append(np.round(np.mean(self.event[:, i:i+length], axis=-1)))
            sliced_event.append(self.event[:, i:i+length])
        
        sliced_metric = np.array(sliced_metric)
        sliced_event = np.array(sliced_event)    
        
        self.metric = sliced_metric
        self.event = sliced_event
        
        self.length = self.event.shape[0]
        
    
    def slcing_second(self, length):
        sliced_metric = []
        sliced_event = []
        for i in range(self.metric.shape[-1] - length + 1):
            tempdata = self.metric[:, i:i+length]
            sliced_metric.append(
                [np.mean(tempdata[:, :(length//2)], axis=-1), np.mean(tempdata[:, (length//2): ], axis=-1)]
                 )
            # sliced_event.append(np.round(np.mean(self.event[:, i:i+length], axis=-1)))
            sliced_event.append(self.event[:, i:i+length])
        
        sliced_metric = np.array(sliced_metric)
        sliced_event = np.array(sliced_event)    
        
        self.metric = sliced_metric
        self.event = sliced_event
        
        self.length = self.event.shape[0]
    
    
    def get_datashape(self, ):
        return self.metric.shape, self.event.shape
        
        
    def __len__(self, ):
        return self.length
    
    
    def __getitem__(self, idx):
        return self.event[idx], self.metric[idx]
    
    
    

class L2_topdown_dataset_doublemean(Dataset):
    def __init__(self, metric, event, length=128, second=False):
        self.metric = metric # timestep, dim
        self.event = event
        self.length = 0
        
        self.v_min = np.array([L2_EVENT_MIN_MAX[key][0] for key in L2_EVENT_MIN_MAX.keys()])[:, np.newaxis] 
        self.v_max = np.array([L2_EVENT_MIN_MAX[key][1] for key in L2_EVENT_MIN_MAX.keys()])[:, np.newaxis]
        
        
        self.norm()
        if second:
            self.slcing_second(length=length)
        else:
            self.slcing(length=length)
        
        # self.metric = self.metric / 100
        
        
    def norm(self ):
        self.metric = self.metric.transpose() # dim, timesteps
        self.event = self.event.transpose() #  dim, teimsteps
        
        # event
        self.event = (self.event - self.v_min) / (self.v_max - self.v_min)
        self.event[self.event < 0] = 0
        self.event[self.event > 1] = 1
        
        # metirc
        self.metric[-1,:] = (self.metric[-1,:] - 1.0) / (3.9 - 1.0) # freq
        self.metric[:-1,:] = self.metric[:-1,:] / 100 # topdowmetricn 
        
        self.metric = self.metric * 100
        
        
        
    def slcing(self, length):
        sliced_metric = []
        sliced_event = []
        for i in range(self.metric.shape[-1] - length + 1):
            prevmean = np.mean(self.metric[:, i:i+(length//2)], axis=-1)
            nextmean = np.mean(self.metric[:, i+(length//2):i+length], axis=-1)
            sliced_metric.append(np.concatenate([prevmean, nextmean], axis=-1))
            sliced_event.append(self.event[:, i:i+length])
        
        sliced_metric = np.array(sliced_metric)
        sliced_event = np.array(sliced_event)    
        
        self.metric = sliced_metric
        self.event = sliced_event
        
        self.length = self.event.shape[0]
        
    
    def slcing_second(self, length):
        sliced_metric = []
        sliced_event = []
        for i in range(self.metric.shape[-1] - length + 1):
            tempdata = self.metric[:, i:i+length]
            sliced_metric.append(
                [np.mean(tempdata[:, :(length//2)], axis=-1), np.mean(tempdata[:, (length//2): ], axis=-1)]
                 )
            # sliced_event.append(np.round(np.mean(self.event[:, i:i+length], axis=-1)))
            sliced_event.append(self.event[:, i:i+length])
        
        sliced_metric = np.array(sliced_metric)
        sliced_event = np.array(sliced_event)    
        
        self.metric = sliced_metric
        self.event = sliced_event
        
        self.length = self.event.shape[0]
    
    
    def get_datashape(self, ):
        return self.metric.shape, self.event.shape
        
        
    def __len__(self, ):
        return self.length
    
    
    def __getitem__(self, idx):
        return self.event[idx], self.metric[idx]
    





class L2_topdown_dataset_event(Dataset):
    def __init__(self, metric, event, length=128, second=False):
        self.metric = metric # timestep, dim
        self.event = event
        self.length = 0
        
        self.meta_event = None
        
        self.v_min = np.array([L2_EVENT_MIN_MAX[key][0] for key in L2_EVENT_MIN_MAX.keys()])[:, np.newaxis] 
        self.v_max = np.array([L2_EVENT_MIN_MAX[key][1] for key in L2_EVENT_MIN_MAX.keys()])[:, np.newaxis]
        
        
        self.norm()
        if second:
            self.slcing_second(length=length)
        else:
            self.slcing(length=length)
        
        # self.metric = self.metric / 100
        
        
    def norm(self ):
        self.metric = self.metric.transpose() # dim, timesteps
        self.event = self.event.transpose() #  dim, teimsteps
        
        # event
        self.event = (self.event - self.v_min) / (self.v_max - self.v_min)
        self.event[self.event < 0] = 0
        self.event[self.event > 1] = 1
        
        # 
        
        
        # # metirc
        # self.metric[-1,:] = (self.metric[-1,:] - 1.0) / (3.9 - 1.0) # freq
        # self.metric[:-1,:] = self.metric[:-1,:] / 100 # topdowmetricn 
        
        # self.metric = self.metric * 100
        
        
        
    def slcing(self, length):
        sliced_meta_event = []
        sliced_event = []
        for i in range(self.metric.shape[-1] - length + 1):
            sliced_meta_event.append(np.mean(self.event[:, i:i+length], axis=-1), )
            # sliced_event.append(np.round(np.mean(self.event[:, i:i+length], axis=-1)))
            sliced_event.append(self.event[:, i:i+length])
        
        sliced_meta_event = np.array(sliced_meta_event)
        sliced_event = np.array(sliced_event)    
        
        self.metric = sliced_meta_event
        self.event = sliced_event
        
        self.length = self.event.shape[0]
        
    
    def slcing_second(self, length):
        sliced_metric = []
        sliced_event = []
        for i in range(self.metric.shape[-1] - length + 1):
            tempdata = self.metric[:, i:i+length]
            sliced_metric.append(
                [np.mean(tempdata[:, :(length//2)], axis=-1), np.mean(tempdata[:, (length//2): ], axis=-1)]
                 )
            # sliced_event.append(np.round(np.mean(self.event[:, i:i+length], axis=-1)))
            sliced_event.append(self.event[:, i:i+length])
        
        sliced_metric = np.array(sliced_metric)
        sliced_event = np.array(sliced_event)    
        
        self.metric = sliced_metric
        self.event = sliced_event
        
        self.length = self.event.shape[0]
    
    
    def get_datashape(self, ):
        return self.metric.shape, self.event.shape
        
        
    def __len__(self, ):
        return self.length
    
    
    def __getitem__(self, idx):
        return self.event[idx], self.metric[idx]














    
    



def l2_data_gather(path, except_time=5):
    files = os.listdir(path) # ['544', '541' ...]
    exclude_step = except_time * 5
    merged_metric = pd.DataFrame()
    merged_event = pd.DataFrame()
    
    for file in files:        
        temp = os.path.join(path, file)
        metricfile = os.path.join(temp, "merged_topdown.csv")
        eventfile = os.path.join(temp, "merged_event.csv")
        
        len_m = pd.read_csv(metricfile).shape[0]
        df_m = pd.read_csv(metricfile, skiprows=range(1, exclude_step), nrows=len_m - exclude_step)
        merged_metric = pd.concat([merged_metric, df_m], ignore_index=True)
        
        
        len_e = pd.read_csv(eventfile).shape[0]
        df_e = pd.read_csv(eventfile, skiprows=range(1, exclude_step), nrows=len_e - exclude_step)
        merged_event = pd.concat([merged_event, df_e], ignore_index=True)
        
        
    
    merged_metric = merged_metric.to_numpy()
    merged_event = merged_event.to_numpy()
    
    return merged_metric, merged_event



# functions for check data range
def check_data(metrics, events):
    112




def get_data(length, mean=False):
    '''
    lenght : the timesteps for data
    mean : L2_topdown_dataset_mean if True else L2_topdown_dataset
    
    '''
    
    # trainset setting
    # datapath = ["/ssd/ssd3/ljy/stable_test/diffusionmodel/newdata/topdowndata", 
                # "/ssd/ssd3/ljy/stable_test/diffusionmodel/newdata/topdowndata2", 
                # "/ssd/ssd3/ljy/stable_test/diffusionmodel/newdata/topdowndata3",]
    datapath = ["/ssd/ssd3/ljy/stable_test/diffusionmodel/newdata/topdowndata", ]
    merged_metric, merged_event = None, None
    for dp in datapath:
        if merged_metric is not None:
            tp_m, tp_e = l2_data_gather(dp) # shape : [timesteps, dim]
            merged_metric = np.concatenate([merged_metric, tp_m])
            merged_event = np.concatenate([merged_event, tp_e])
        else: 
            merged_metric, merged_event = l2_data_gather(dp) # shape : [timesteps, dim]
            
    
    check_data(merged_metric, merged_event)
            
    
    # get datatset
    # l2_dataset = L2_topdown_dataset_mean(merged_metric, merged_event, length) # shape [batch, dim, timesteps]
    l2_dataset = L2_topdown_dataset_doublemean(merged_metric, merged_event, length) # shape [batch, dim, timesteps]
    
    
    return merged_metric, merged_event, l2_dataset




# for overfitting on small dataset -> just test whole process
def get_data_test(length, mean=False):
    datapath = ["/ssd/ssd3/ljy/stable_test/diffusionmodel/newdata/topdowndata_test", ]
    merged_metric, merged_event = None, None
    for dp in datapath:
        if merged_metric is not None:
            tp_m, tp_e = l2_data_gather(dp) # shape : [timesteps, dim]
            merged_metric = np.concatenate([merged_metric, tp_m])
            merged_event = np.concatenate([merged_event, tp_e])
        else: 
            merged_metric, merged_event = l2_data_gather(dp) # shape : [timesteps, dim]
    # merged_metric, merged_event = l2_data_gather("/ssd/ssd3/ljy/stable_test/diffusionmodel/newdata/topdowndata") # shape : [timesteps, dim]
    
    check_data(merged_metric, merged_event)
    l2_dataset = L2_topdown_dataset(merged_metric, merged_event, length) # shape [batch, dim, timesteps]
    
    
    return merged_metric, merged_event, l2_dataset
