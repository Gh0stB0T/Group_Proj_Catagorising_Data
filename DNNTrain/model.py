'''
Define different models

Kevin Zhang 2023
'''

from torch import nn
import torch
import copy

import numpy as np

def model_summary(model):
    '''
    prints a summery of the model, mainly for debugging uses
    '''
    print(next(model.named_modules())[1])
    
    param_size = 0
    trainable_param_size = 0
    for name, param in model.named_parameters():
        nparam = param.nelement()
        print(f'{name}, {nparam}, trainable: {param.requires_grad}')
        param_size += nparam
        if param.requires_grad:
            trainable_param_size += nparam
    print(f'Total parameter = {param_size}')
    print(f'Total trainable parameter = {trainable_param_size}\n')
    # time.sleep(1)

def initialise_weights(layers, a, nonlinearity='leaky_relu'):
    '''
    Initialise weights for dense layers
    '''
    for layer in layers[:-1]: # skip the last layer, which is either Lrelu or nn.Linear linking the output layer
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, a=a, nonlinearity=nonlinearity)
        
        
class FCNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, 
                 alpha=0.3, 
                 dropout=None, dropout_on_last_layer=False,
                 ):
        '''
        fully connected neural net
        
        branching: the nth last layer at which the network splits in half
        alpha: sets the negative gradient for the leakyrelu activation
        dropout: None or float or list of float for nn.Dropout
        dropout_on_last_layer: whether to add Dropout right before the output
        activation_on_last_layer: whether to add nonlinear activation right before the output layer
        '''
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        hdim = np.array([input_dim]+hidden_dim)
        
        if isinstance(dropout, float):
            dropout = [dropout]*len(hdim)
        
        # assert len(dropout) == len(hdim)
        
        hidden_layers=[]
        for i in range(len(hidden_dim)):
            layer = []
            layer.append(nn.Linear(hdim[i], hdim[i+1]))
            layer.append(nn.LeakyReLU(negative_slope=alpha))
            if dropout is not None:
                layer.append(nn.Dropout(dropout[i]))
            hidden_layers.append(layer)
            
        if not dropout_on_last_layer and dropout is not None:
            del hidden_layers[-1][2]
            
        self.layers = nn.Sequential(*sum(hidden_layers, []),
                                    nn.Linear(hdim[-1], output_dim),
                                    nn.Sigmoid(),
                                    )

        initialise_weights(self.layers, a=alpha, nonlinearity='leaky_relu')
        
    def forward(self, x):
        h = self.layers(x)
        return h
        
class one_layer_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.layer(x)