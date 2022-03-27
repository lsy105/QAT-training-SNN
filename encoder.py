import numpy as np

def RateEncoder(input, min_val, max_val, spike_time):
    rate = float(input - min_val) / (max_val - min_val)
    temp = np.random.rand(spike_time)
    res = np.zeros(spike_time)
    res[temp <= rate] = 1
    idx = np.where(res == 1)[0]
    return res

