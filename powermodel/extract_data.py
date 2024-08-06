import subprocess
import numpy as np
import os
import signal
import time
import argparse

##############################################################################################################################

# l2 events for power modeling
L2_EVENT_LIST = ["idq_uops_not_delivered.core",
                 "cpu_clk_unhalted.thread_any",
                 "uops_retired.retire_slots",
                 "uops_issued.any",
                 "int_misc.recovery_cycles_any",
                 "idq_uops_not_delivered.cycles_0_uops_deliv.core",
                 "uops_retired.macro_fused",
                 "inst_retired.any",
                 "br_misp_retired.all_branches",
                 "machine_clears.count",
                 "exe_activity.2_ports_util",
                 "cycle_activity.stalls_mem_any",
                 "exe_activity.bound_on_stores",
                 "cycle_activity.stalls_total",
                 "exe_activity.1_ports_util"]
# a core number
CORE = 7

# current dir
CURRENT_DIR = os.getcwd()

################################################################################################################################



parser = argparse.ArgumentParser()

parser.add_argument(
    "--path", type=str, default="0.001", help="path for executable benchmark file"
)
parser.add_argument(
    "--exe_cmd", type=str, default="1000", help="exe command in the path"
)
parser.add_argument(
    "--path_save", type=str, default="15", help="save path for data"
)



def data_extract(path, exe_cmd, path_save):
    PATH = path
    EXE_CMD = exe_cmd
    save_path = path_save
    # current dir
    CURRENT_DIR = os.getcwd()
    
    l2_events = ""
    event_save = list()
    power_save = list()

    for i, event in enumerate(L2_EVENT_LIST):
        l2_events += event
        if event != L2_EVENT_LIST[-1]:
            l2_events +=","
        event_save.append(list())
        
    perf_cmd = f"sudo perf stat -C {CORE} -x, --time 200 -e {l2_events} 2>&1" 
    power_cmd = "sudo cat /sys/class/powercap/intel-rapl/intel-rapl\:0/energy_uj"

    os.chdir(PATH)
    process = subprocess.Popen(
                f"sudo taskset -c {CORE} {EXE_CMD}",
                shell=True,
                preexec_fn=os.setsid  # Create a new process group to make it easier to kill the entire process tree
            )
    process_id = os.getpgid(process.pid)


    w_tic = time.time()


    while process.poll() == None: # process is still running
        tic = time.time()
        prev_energy = subprocess.check_output(power_cmd, shell=True)
        
        event_out = subprocess.check_output(perf_cmd, shell=True)
        
        after_energy = subprocess.check_output(power_cmd, shell=True)
        toc = time.time()
        
        prev_energy = prev_energy.decode()
        event_out = event_out.decode()
        after_energy = after_energy.decode()
        t = toc - tic
        
        event_out = event_out.split('\n')
        for i in range(len(event_save)):
            event_save[i].append(float(event_out[i].split(',')[0]))
            
        j_power = round((float(after_energy) - float(prev_energy) )/ (t * 1000000), 6) # uj -> j, Watt
        power_save.append(j_power)    

    w_toc = time.time()
    # os.killpg(os.getpgid(process_id), signal.SIGTERM)

    print("time consume to run benchmark : ", w_toc - w_tic)


    os.chdir(CURRENT_DIR)
    
    
    # save data
    event_save = np.array(event_save)
    power_save = np.array(power_save).reshape(1,-1)

    save_data = np.concatenate([event_save, power_save], axis=0)

    np.save(f"{save_path}", save_data)
    
    print("data saved")
    
    
    return




if __name__ == "__main__":
    args = parser.parse_args()
    # print(args.path, args.exe_cmd, args.path_save)
    data_extract(path=args.path, exe_cmd=args.exe_cmd, path_save=args.path_save)









# l2_events = "idq_uops_not_delivered.core,cpu_clk_unhalted.thread_any,uops_retired.retire_slots,uops_issued.any,int_misc.recovery_cycles_any,idq_uops_not_delivered.cycles_0_uops_deliv.core,uops_retired.macro_fused,inst_retired.any,br_misp_retired.all_branches,machine_clears.count,exe_activity.2_ports_util,cycle_activity.stalls_mem_any,exe_activity.bound_on_stores,cycle_activity.stalls_total,exe_activity.1_ports_util"
    