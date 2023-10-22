'''
Kaicheng Zhang 2023
'''

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random
import torch
import time

def set_seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def MAPE(pred, act):
    '''
    returns mean absolute percentage error for each pytorch batch
    '''
    return 100/torch.numel(act) * torch.sum(torch.abs((act-pred)/act))

def samplewise_MAPE(pred, act):
    '''
    returns mean absolute percentage error for each sample
    '''
    return 100/act.shape[1] * np.sum(np.abs((act-pred)/act), axis=1)

def torch_samplewise_MAPE(pred, act):
    '''
    returns mean absolute percentage error for each sample
    '''
    return 100/act.size()[1] * torch.sum(torch.abs((act-pred)/act), axis=1)

def samplewise_MSE(act, pred):
    '''
    returns mean squared error for each sample
    '''
    return np.sum((act-pred)**2, axis=1)/act.shape[1]

def samplewise_MAE(pred, act):
    '''
    returns mean absolute error for each sample
    '''
    return 1/act.shape[1] * np.sum(np.abs(act-pred), axis=1)

def speed_test(model, dim, num_test=1e5, precision='full', device='cuda'):
    test_tensor = torch.randn(int(num_test), int(dim))
    
    if device == 'cuda':
        model.to(device)
        test_tensor = test_tensor.to(device)
        
    if precision == 'half':
        model.half()
        test_tensor = test_tensor.half()
        
    t0 = time.time()
    model(test_tensor)
    dt = time.time() - t0
    
    print(f'Speed test run on {num_test:0.2e} samples,\nTook {dt} seconds in total, {dt/num_test:0.4e} seconds per sample')
    return dt
    