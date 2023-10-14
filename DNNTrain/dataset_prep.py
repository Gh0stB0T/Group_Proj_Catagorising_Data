'''
Functions used to prepare the dataset for training ANNs
Kevin Zhang 2023
'''

import torch

import pandas as pd
import numpy as np

# build a dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, channel_size=None):
        '''
        prepare dataset for NN training, x and y can be either pd.DataFrames or np.array
        
        channel_size: If not None, y will be shaped into a 2D array with shape (channel, channel_size)
            If None, y will be a 1D array
        '''
        if isinstance(x, pd.DataFrame):
            self.x = x.to_numpy().astype(np.float32)
            self.x_headings = x.columns.values # saves the headings for referencing
            self.len = len(x.index)
        else:
            self.x = x.astype(np.float32)
            self.x_headings = None 
            self.len = len(x)
            
        if isinstance(y, pd.DataFrame):
            self.y = y.to_numpy().astype(np.float32)
            self.y_headings = y.columns.values
        else:
            self.y = y.astype(np.float32)
            self.y_headings = None
        
        assert len(self.x) == len(self.y)


    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    # check the dtype of the dataset for pytorch training
    def check_dtype(self):
        if self.x.dtype != np.float32:
            print(f'Detected {self.x.dtype} in dataset.x, converting it to np.float32')
            self.x = self.x.astype(np.float32)
        if self.y.dtype != np.float32:
            print(f'Detected {self.y.dtype} in dataset.y, converting it to np.float32')
            self.y = self.y.astype(np.float32)

def to_dense(sparse):
    dense = np.zeros([len(sparse), np.max(sparse)])
    dense[:, sparse-1] = 1
    return dense