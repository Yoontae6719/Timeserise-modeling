from __future__ import division
import numpy as np
import torch
import os
import logging
import click
from torch.utils.data import DataLoader, Dataset, Sampler


class TrainDataset(Dataset):
    def __init__(self, cnf, data_name):

        self.params = cnf.all_params
        self.data = np.load(os.path.join(self.params["data_path"], f'train_data_{data_name}.npy' ))
        self.label = np.load(os.path.join(self.params["data_path"], f'train_label_{data_name}.npy'))
        self.train_len = self.data.shape[0]

    def __len__(self):
        return self.train_len

    def __getitem__(self, ind):
        return (self.data[ind, :, : -1],
                int(self.data[ind, 0, -1]),
                self.label[ind])

class TestDataset(Dataset):
    def __init__(self, cnf, data_name):

        self.params = cnf.all_params

        self.data = np.load(os.path.join(self.params["data_path"], f'test_data_{data_name}.npy'))
        self.v = np.load(os.path.join(self.params["data_path"], f'test_v_{data_name}.npy'))
        self.label = np.load(os.path.join(self.params["data_path"], f'test_label_{data_name}.npy'))
        self.test_len = self.data.shape[0]

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]),self.v[index],self.label[index])

class WeightedSampler(Sampler):
    def __init__(self, cnf, data_name, replacement=True):

        self.params = cnf.all_params
        v = np.load(os.path.join(self.params["data_path"], f'train_v_{data_name}.npy'))
        self.weights = torch.as_tensor(np.abs(v[:,0])/np.sum(np.abs(v[:,0])), dtype=torch.double)
        self.num_samples = self.weights.shape[0]
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples



