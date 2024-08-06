import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats
import math

# from helper import get_file_list


def save_data(path, save_path="./data/test.png"):
    data = np.load(path)
    event_data = data[:-1]
    power_data = data[-1]
    
    # # plot two graph (event, power)
    
    plt.subplot(2,1,1)
    for i, event in enumerate(event_data):
        x = np.arange(event.shape[-1])
        plt.plot(x, event)
    
    plt.subplot(2,1,2)
    x = np.arange(power_data.shape[-1])
    plt.plot(x, power_data)
    
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
    
    
def plot_data(path):
    data = np.load(path)
    power_data = data[-1]
    
    # plot two graph (event, power)
    
    plt.subplot(2,1,1)
    for i, event in enumerate(data):
        if (i != len(data)-1):
            x = np.arange(event.shape[-1])
            plt.plot(x, event)
    
    plt.subplot(2,1,2)
    x = np.arange(power_data.shape[-1])
    plt.plot(x, power_data)
    
    plt.legend()
    plt.plot()
    

def qq_plot(real, predicted):
    MIN = 30
    MAX = 80
    
    
    # 1:1 matching
    
    plt.figure()
    # plt.scatter(a, b)
    plt.xlim(MIN, MAX)
    plt.ylim(MIN, MAX)
    plt.xlabel("real")
    plt.ylabel("predicted")
    plt.scatter(real, predicted)
    plt.plot([MIN, MAX], [MIN, MAX])
    plt.savefig('/home/cell/ljy/DVFS_RAY/powermodel/data/powerdata_validation/matching.png')
    plt.close()
    
    
    # qqplot
    real = np.sort(real)
    predicted = np.sort(predicted)
    
    
    plt.figure()
    # plt.scatter(a, b)
    plt.xlim(MIN, MAX)
    plt.ylim(MIN, MAX)
    plt.xlabel("real")
    plt.ylabel("predicted")
    plt.scatter(real, predicted)
    plt.plot([MIN, MAX], [MIN, MAX])
    plt.savefig('/home/cell/ljy/DVFS_RAY/powermodel/data/powerdata_validation/qqplot.png')
    plt.close()



def mae_mape(real, predicted):
    real = real.reshape(-1)
    predicted = predicted.reshape(-1)
    N = real.shape[-1]
    
    mae = 0
    mape = 0
    for i, (re, pre) in enumerate(zip(real, predicted)):
        mae += abs(re - pre)
        mape += abs((re- pre) / re)
    
    mae = round((mae / N), 5)
    mape = round((mape / N) * 100, 5)
         
    return mae, mape
    


def r_squre(real, predicted):
    real = real.reshape(-1)
    predicted = predicted.reshape(-1)
    
    mean_real = np.mean(real)
    
    T = (real - mean_real) ** 2 # SST
    R = (mean_real - predicted) ** 2 # SSR
    E = (real - predicted) ** 2 # SSE
    
    T = np.sum(T)
    R = np.sum(R)
    E = np.sum(E)
    
    
    expl = round((R / T), 4)
    
    return expl


    

if __name__=='__main__':
    # save_data("/home/cell/ljy/DVFS_RAY/powermodel/data/modeldata/503_1.npy")
    
    
    '''
    sample power & predicted power
    '''
    real_power = np.array([1,2,3,4,5])
    predicted_power = np.array([6,7,8,9,10])
    
    
    '''
    plot powerdata
    '''
    # datapath = "/home/cell/ljy/DVFS_RAY/powermodel/data/powerdata"
    # files = get_file_list(datapath)
    # for i, file in enumerate(files):
    #     path = os.path.join(datapath, file)
    #     name = file[:-4]
    #     s_path = f"./data/powerdata_plot/{name}.png"
    #     save_data(path=path, save_path=s_path)
        
    
    # real_power = np.load('/home/cell/ljy/DVFS_RAY/powermodel/data/powerdata_validation/real_power.npy')
    # predicted_power = np.load('/home/cell/ljy/DVFS_RAY/powermodel/data/powerdata_validation/predicted_power.npy')
    
    # # plot qqplot
    # qq_plot(real_power, predicted_power) 
    
    # # calculate MAE and MAPE
    # mae, mape = mae_mape(real_power, predicted_power)
    # print("MAE : ", mae)
    # print("MAPE : ", mape)
    
    # calculate R squre
    # expl = r_squre(real_power, predicted_power)
    # print("r_squre : ", expl)
    
    
    a = np.load("/home/cell/ljy/DVFS_RAY/powermodel/data/powerdata/538.npy")
    
    print(a.shape)
    
    
