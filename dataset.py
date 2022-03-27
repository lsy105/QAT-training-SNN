import torch
from torch.utils import data
import numpy as np
from encoder import *

class Dataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, noise=False):
        self.labels = labels.flatten()
        self.data = data.flatten()
        self.sequence_length = sequence_length
        self.all_data = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.data), axis=0)
        self.all_labels = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.labels), axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            X.append(self.all_data[i])
        y = self.all_labels[idx]
        return np.array(X), np.array(y)

class SpikingDataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, time_step):
        self.labels = labels.flatten()
        self.data = data.flatten()
        self.sequence_length = sequence_length
        self.all_data = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.data), axis=0)
        self.all_labels = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.labels), axis=0)
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp = RateEncoder(self.all_data[i], self.min, self.max, self.time_step)
            temp = np.array(temp)
            X.append(temp)
        X = np.stack(X, axis=1)
        y = np.array(self.all_labels[idx])
        return X, y

class NewSpikingDataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, time_step):
        self.labels = labels.flatten()
        self.data = data.flatten()
        self.sequence_length = sequence_length
        self.all_data = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.data), axis=0)
        self.all_labels = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.labels), axis=0)
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp = RateEncoder(self.all_data[i], self.min, self.max, self.time_step)
            temp = np.array(temp)
            X.append(temp)
        X = np.stack(X, axis=0)
        y = np.array(self.all_labels[idx])
        return X, y
    
    
class SpikingMNISTDataset(data.Dataset):
    def __init__(self, data, labels, time_step):
        self.all_data = data 
        self.all_labels = labels
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        # Select sample
        X = []
        for i in range(self.all_data.shape[1]):
            temp = RateEncoder(self.all_data[idx, i], self.min, self.max, self.time_step)
            X.append(temp)
        X = np.stack(X, axis=1)
        y = np.array(self.all_labels[idx])
        return X, y

class SpikingCIFAR10Dataset(data.Dataset):
    def __init__(self, data, labels, time_step):
        self.all_data = data 
        self.all_labels = labels
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        # Select sample
        X = []
        for i in range(self.all_data.shape[1]):
            temp = RateEncoder(self.all_data[idx, i], self.min, self.max, self.time_step)
            X.append(temp)
        X = np.stack(X, axis=1)
        y = np.array(self.all_labels[idx])
        return X, y
