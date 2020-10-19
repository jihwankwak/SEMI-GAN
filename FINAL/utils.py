import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F    
from scipy import linalg
    
    
def train_mean_std(x, y):
        
    x_mean = np.mean(x[:,:3], axis=0, dtype=np.float64)
    x_std = np.std(x[:,:3], axis=0, dtype=np.float64)
    
    
    x_mean = x_mean.reshape(1, 3)
    x_std = x_std.reshape(1, 3)
    
    x_mean = np.hstack((x_mean, np.ones((x_mean.shape[0], 1))))
    x_std = np.hstack((x_std, np.ones((x_std.shape[0], 1))))
    
    x_mean = np.hstack((x_mean, np.zeros((x_mean.shape[0], 1))))
    x_std = np.hstack((x_std, np.zeros((x_std.shape[0], 1))))
        
    y_mean = np.mean(y, axis=0, dtype=np.float64)
    y_std = np.std(y, axis=0, dtype=np.float64)
    
    return x_mean, x_std, y_mean, y_std

def normalize_train(x, y, x_mean, x_std, y_mean, y_std):
    
    norm_x = (x - x_mean) / x_std
    norm_y = (y - y_meah) / y_std
    
    return norm_x, norm_y

def normalize(x, y):
        
    x_mean = np.mean(x, axis=0, dtype=np.float32)
    x_std = np.std(x, axis=0, dtype=np.float32)
        
    y_mean = np.mean(y, axis=0, dtype=np.float32)
    y_std = np.std(1e+10*y, axis=0, dtype=np.float32)

    norm_x = ( x - x_mean ) / (x_std)
    norm_y = ( y - y_mean )*1e+10 / (y_std)
       
    y_std = y_std / 1e+10
    return norm_x, norm_y, x_mean, x_std, y_mean, y_std

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



def FID_score(generated_samples, real_samples):
    # https://en.wikipedia.org/wiki/Sample_mean_and_covariance
    mu_g = np.mean(generated_samples, axis=0, keepdims=True).T
    mu_r = np.mean(real_samples, axis=0, keepdims=True).T
    cov_g = (generated_samples - np.ones((len(generated_samples),1)).dot(mu_g.T)).T.dot((generated_samples - np.ones((len(generated_samples),1)).dot(mu_g.T)))/(len(generated_samples)-1)
    cov_r = (real_samples - np.ones((len(real_samples),1)).dot(mu_r.T)).T.dot((real_samples - np.ones((len(real_samples),1)).dot(mu_r.T)))/(len(real_samples)-1)

    
    mean_diff = mu_g - mu_r
    cov_prod_sqrt = linalg.sqrtm(cov_g.dot(cov_r))
    
    #numerical instability of linalg.sqrtm
    #https://github.com/mseitzer/pytorch-fid/blob/master/pytorch_fid/fid_score.py
    eps=1e-6
    if not np.isfinite(cov_prod_sqrt).all():
        offset = np.eye(cov_g.shape[0]) * eps
        cov_prod_sqrt = linalg.sqrtm((cov_g + offset).dot(cov_r + offset))

    if np.iscomplexobj(cov_prod_sqrt):
        if not np.allclose(np.diagonal(cov_prod_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_prod_sqrt.imag))
            raise ValueError('Imaginary component {}'.format(m))
        cov_prod_sqrt = cov_prod_sqrt.real
    
    
    FID_score = np.sum(mean_diff**2) + np.trace(cov_g + cov_r -2*cov_prod_sqrt)
    
    
    return FID_score