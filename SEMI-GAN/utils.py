import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F    
    
def normalize(x, y):
        
    x_mean = np.mean(x, axis=0, dtype=np.float32)
    x_std = np.std(x, axis=0, dtype=np.float32)
        
    y_mean = np.mean(y, axis=0, dtype=np.float32)
    y_std = np.std(y, axis=0, dtype=np.float32)
        
    norm_x = ( x - x_mean ) / (x_std)
    norm_y = ( y - y_mean ) / (y_std)
        
    return norm_x, norm_y, y_mean, y_std

def init_params(model):
    for p in model.parameters():
        if (p.dim() > 1):
            nn.init.xavier_normal_(p)
        else:
            nn.init.uniform_(p, 0.1, 0.2)    
    
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
       
    
def sample_z(batch_size = 1, d_noise=100):    
    return torch.randn(batch_size, d_noise).cuda()