import torch
from torch.utils.data import Dataset

import numpy as np

# L2_EVENT_MIN_MAX = {"idq_uops_not_delivered.core":[2400000.0, 1950000000.0],
#                     "cpu_clk_unhalted.thread_any":[510000000.0, 797000000.0],
#                     "uops_retired.retire_slots":[35000000.0, 2960000000.0],
#                     "uops_issued.any":[35000000.0, 2930000000.0],
#                     "int_misc.recovery_cycles_any":[28000.0, 110000000.0],
#                     "idq_uops_not_delivered.cycles_0_uops_deliv.core":[140000.0, 227000000.0],
#                     "uops_retired.macro_fused":[40000.0, 510000000.0],
#                     "inst_retired.any":[410000.0, 2870000000.0],
#                     "br_misp_retired.all_branches":[1000.0, 17000000.0],
#                     "machine_clears.count":[150.0, 2000000.0],
#                     "exe_activity.2_ports_util":[4200000.0, 330000000.0],
#                     "cycle_activity.stalls_mem_any":[300000.0, 688000000.0],
#                     "exe_activity.bound_on_stores":[5000.0, 760000000.0],
#                     "cycle_activity.stalls_total":[3100000.0, 750000000.0],
#                     "exe_activity.1_ports_util":[7300000.0, 310000000.0]
#                     }

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


class PowerModelingDataset(Dataset):
    def __init__(self, event, power):
        assert event.shape[-1] == power.shape[-1]
        
        self.v_min = np.array([L2_EVENT_MIN_MAX[key][0] for key in L2_EVENT_MIN_MAX.keys()])[:, np.newaxis] 
        self.v_max = np.array([L2_EVENT_MIN_MAX[key][1] for key in L2_EVENT_MIN_MAX.keys()])[:, np.newaxis]
        
        self.len = power.shape[-1]
        self.event = event
        self.power = power # label
        self.norm_data()
        
    
    def norm_data(self, ):
        '''min-max normalization'''
        # event (2d)
        # v_min = np.min(self.event, axis=1)[:, np.newaxis]
        # v_max = np.max(self.event, axis=1)[:, np.newaxis]
        self.event = (self.event - self.v_min) / (self.v_max - self.v_min)
        self.event[self.event < 0] = 0
        self.event[self.event > 1] = 1
        
        # power (1d)
        v_min = np.min(self.power)
        v_max = np.max(self.power)
        self.power = (self.power - v_min) / (v_max - v_min)
        
        self.p_min = v_min
        self.p_max = v_max
        
        self.power = self.power[np.newaxis, :]
        
            
    def __len__(self, ):
        return self.len
    
    
    def __getitem__(self, index):
        return self.event[:,index], self.power[:,index]
    
    
    def min_max(self, ):
        '''return min and max value of power'''
        return self.p_min, self.p_max
    

