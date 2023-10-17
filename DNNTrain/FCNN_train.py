'''
Trains the NN

Kaicheng Zhang 2023
'''

from sklearn.datasets import load_digits

from model import FCNN, one_layer_model
from trainer import Trainer 
from dataset_prep import Dataset, to_dense
from utils import set_seed_torch

import torch
torch.cuda.empty_cache()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json
import os

set_seed_torch(42)

train_logdir = './'

data_split = [0.7, 0.1, 0.2]

class TrainerConfig:
    device = 'cpu' # 'cpu' or 'gpu'
    max_epochs = 500
    batch_size = 1000 #70000 #256
    ini_lr = 5e-4 # initial learning rate
    fin_lr = 1e-6 # final learning rate, if using LinearLR
    gamma = 0.996 # multiplicative factor, if using ExponentialLR
    decayfactor = 0.5 # if using ReduceLROnPlateau
    # patience = 10 # if using ReduceLROnPlateau, not implemented
    betas = (0.9, 0.99)
    eps = 1e-8
    grad_norm_clip = 1.0
    weight_decay = 1e-10
    epoch_save_freq = 0 # set to 0 to only save at the end of the last epoch
    epoch_save_name = train_logdir+'FCNN_'
    num_workers = 0 # for DataLoader, only zero works
    progress_bar = False # whether to show a progress bar
    model_shape = [20]
    dropout = None
    dropout_on_last_layer = False
    

    
# ----------------------------------------------------------------------------

mnist = load_digits()
x = np.array(mnist.data)
y = to_dense(np.array(mnist.target))

assert np.allclose(np.sum(data_split), 1.)
split_idx = np.array(data_split)*len(x)
split_idx = split_idx.astype('int')

# ----------------------------------------------------------------------------

train_set = Dataset(x[:split_idx[0]], y[:split_idx[0]])
val_set = Dataset(x[split_idx[0]:split_idx[-1]], y[split_idx[0]:split_idx[-1]])
test_set = Dataset(x[split_idx[-1]:], y[split_idx[-1]:])

model = FCNN(train_set.x.shape[1], train_set.y.shape[1], TrainerConfig.model_shape 
             )

trainer = Trainer(model, TrainerConfig, train_set, test_set)

log = trainer.train()

test_result = trainer.test()

for key in log.keys():
    plt.figure()
    plt.title(str(key))
    print(key) # in case plt.title does not work
    plt.plot(log[key])
    plt.show()
    
# if test_result is not None:
#     print(f'test_loss: {test_result["loss"]} test_err: {test_result["err"]}')

# save logs, configs
# with open(TrainerConfig.epoch_save_name+'logs.pkl', 'wb') as f:
#     pickle.dump(log, f)
# with open(TrainerConfig.epoch_save_name+'config.json', 'w') as f:
#     f.write(str(dict(vars(TrainerConfig))))

print('logs & config saved')
