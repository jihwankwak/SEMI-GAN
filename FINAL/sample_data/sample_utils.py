import torch
from torch.utils.data import DataLoader, TensorDataset

from torchvision import transforms
from scipy import linalg

from scipy import stats
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import cv2

import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.metrics import r2_score
import os
        
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

def FID_score_each_X(generated_samples, real_samples, num_in_gen, num_in_cycle):
    num_of_cycle = int(len(real_samples)/num_in_cycle)
    FID_score_list = []
    for i in range(num_of_cycle) :
        generated_samples_cycle = generated_samples[i*num_in_gen : (i+1)*num_in_gen]
        real_samples_cycle = real_samples[i*num_in_cycle : (i+1)*num_in_cycle]
        FID = FID_score(generated_samples_cycle, real_samples_cycle)
        FID_score_list.append(FID)
    return np.mean(np.array(FID_score_list)), FID_score_list

def KL(P,Q):
    epsilon = 1e-15
    P = P+epsilon
    Q = Q+epsilon

    KL_div = np.sum(P*np.log(P/Q))
    return KL_div

def KL_with_KDE(generated_samples, real_samples):
    # Fitting
    kde_real = stats.gaussian_kde(real_samples.T)
    kde_gen = stats.gaussian_kde(generated_samples.T)
    # Estimate distribution
    density_real = kde_real(real_samples.T)
    density_gen = kde_gen(generated_samples.T)
    # Compare
    KL_div = KL(density_real, density_gen)
    return KL_div

def KL_with_KDE_each_X(generated_samples, real_samples, num_in_cycle):
    num_of_cycle = int(len(generated_samples)/num_in_cycle)
    
    KL_div_list = []
    for i in range(num_of_cycle):
        generated_samples_cycle = generated_samples[i*num_in_cycle : (i+1)*num_in_cycle]
        real_samples_cycle = real_samples[i*num_in_cycle : (i+1)*num_in_cycle]
        KL_div = KL_with_KDE(generated_samples_cycle, real_samples_cycle)
        KL_div_list.append(KL_div)
        
    return np.mean(np.array(KL_div_list))

def EMD_each_X(generated_samples, real_samples, num_in_gen, num_in_cycle):
    generated_samples = np.float32(generated_samples)
    real_samples = np.float32(real_samples)
    num_of_cycle = int(len(real_samples)/num_in_cycle)
    EMD_score_list = []
    for i in  range(num_of_cycle):
        generated_samples_cycle = generated_samples[i*num_in_gen : (i+1)*num_in_gen]
        real_samples_cycle = real_samples[i*num_in_cycle : (i+1)*num_in_cycle]
        
#         for i in range(len(generated_samples_cycle)):
#             for j in range(len(generated_samples_cycle[i])):
#                 if generated_samples_cycle[i][j] < 0:
#                     print(generated_samples_cycle[i][j])
        
#         for i in range(len(real_samples_cycle)):
#             for j in range(len(real_samples_cycle[i])):
#                 if real_samples_cycle[i][j] < 0:
#                     print(real_samples_cycle[i][j])

        EMD, _, _ = cv2.EMD(generated_samples_cycle+100, real_samples_cycle+100, cv2.DIST_L2)
        EMD_score_list.append(EMD)
        
    return np.mean(np.array(EMD_score_list)), EMD_score_list

# EMD
def EMD_from_2d_kde(generated_samples, real_samples, index_x, index_y, num_coordinate): #각 joint pair 별로 2d sample을 kde로 fittining시키고 얻은 pdf값을 weight로 삼아 histogram 만들어서 EMD

    x_min = min(min(real_samples[:,index_x]), min(generated_samples[:,index_x]))
    x_max = max(max(real_samples[:,index_x]), max(generated_samples[:,index_x]))
    y_min = min(min(real_samples[:,index_y]), min(generated_samples[:,index_y]))
    y_max = max(max(real_samples[:,index_y]), max(generated_samples[:,index_y]))

    position_cartesian = np.array(np.meshgrid(np.linspace(x_min, x_max, num_coordinate), np.linspace(y_min, y_max, num_coordinate))).T.reshape(-1,2)
    
    real_samples_2d = real_samples[:,[index_x, index_y]]
    generated_samples_2d = generated_samples[:,[index_x, index_y]]
    
    kde_real = stats.gaussian_kde(real_samples_2d.T)
    kde_gen = stats.gaussian_kde(generated_samples_2d.T)
    
    # Estimate distribution
    density_real = kde_real(position_cartesian.T)
    density_gen = kde_gen(position_cartesian.T)
    
    real_weight_position = np.concatenate((density_real.reshape(-1,1), position_cartesian),axis=1).astype('float32')
    gen_weight_position = np.concatenate((density_gen.reshape(-1,1), position_cartesian),axis=1).astype('float32')

    # Compare
    #KL_div = KL(density_real, density_gen)
    #print('KL_div', KL_div)

    EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)
    
    return EMD_score


def EMD_from_1d_kde(generated_samples, real_samples, index_x, num_coordinate): #대각 성분에 대해서는 1d kde로..

    x_min = min(min(real_samples[:,index_x]), min(generated_samples[:,index_x]))
    x_max = max(max(real_samples[:,index_x]), max(generated_samples[:,index_x]))
    

    position_cartesian = np.linspace(x_min, x_max, num_coordinate).reshape(-1,1)
    
    real_samples_1d = real_samples[:,[index_x]]
    generated_samples_1d = generated_samples[:,[index_x]]
    
    kde_real = stats.gaussian_kde(real_samples_1d.T)
    kde_gen = stats.gaussian_kde(generated_samples_1d.T)
    
    # Estimate distribution
    density_real = kde_real(position_cartesian.T)
    density_gen = kde_gen(position_cartesian.T)
    
    real_weight_position = np.concatenate((density_real.reshape(-1,1), position_cartesian),axis=1).astype('float32')
    gen_weight_position = np.concatenate((density_gen.reshape(-1,1), position_cartesian),axis=1).astype('float32')

    # Compare
    #KL_div = KL(density_real, density_gen)
    #print('KL_div', KL_div)

    EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)
    
    return EMD_score

def EMD_all_pair(normalized_generated_samples, normalized_real_samples, num_coordinate): #하나의 X에 대해, 모든 pair(36가지) EMD를 list로 뱉는 함수.

    dim = normalized_real_samples.shape[1]
    EMD_score_list = []

    for i in range(dim):
        for j in range(dim):
            if i == j :
                EMD_score_list.append(EMD_from_1d_kde(normalized_generated_samples, normalized_real_samples, i, num_coordinate))
            else : 
                EMD_score_list.append(EMD_from_2d_kde(normalized_generated_samples, normalized_real_samples, i, j, num_coordinate))

    return np.array(EMD_score_list)

def EMD_all_pair_each_X_val(generated_samples, real_samples, num_coordinate, num_in_gen, num_in_real): #여러 X에 대해 각각 쪼개서, 모든 pair(36가지)에 대한 EMD list를 모아서 뱉는 함수.
    num_of_cycle = int(len(real_samples)/num_in_real)
    
    print(num_of_cycle)
    dim = real_samples.shape[1]
    
    EMD_score_list = []
    

    for i in range(num_of_cycle):
        generated_samples_cycle = generated_samples[i*num_in_gen : (i+1)*num_in_gen]
        real_samples_cycle = real_samples[i*num_in_real : (i+1)*num_in_real]
        
        xy_mean = np.mean(real_samples_cycle, axis=0, dtype=np.float32)
        xy_std = np.std(real_samples_cycle, axis=0, dtype=np.float32)
        
        normalized_generated_samples = (generated_samples_cycle-xy_mean)/xy_std
        normalized_real_samples = (real_samples_cycle-xy_mean)/xy_std
        
        EMD_score_list.append(EMD_all_pair(normalized_generated_samples, normalized_real_samples, num_coordinate))
        
    return np.array(EMD_score_list)

def EMD_all_pair_each_X_test(generated_samples, real_samples, num_coordinate, num_of_cycle, num_in_gen_list, num_in_real_list): #여러 X에 대해 각각 쪼개서, 모든 pair(36가지)에 대한 EMD list를 모아서 뱉는 함수.
    dim = real_samples.shape[1]
    
    EMD_score_list = []

    for i in range(num_of_cycle):
        generated_samples_cycle = generated_samples[sum(num_in_gen_list[:i]):sum(num_in_gen_list[:i])+num_in_gen_list[i]]
        real_samples_cycle = real_samples[sum(num_in_real_list[:i]):sum(num_in_real_list[:i])+num_in_real_list[i]]
        
        xy_mean = np.mean(real_samples_cycle, axis=0, dtype=np.float32)
        xy_std = np.std(real_samples_cycle, axis=0, dtype=np.float32)
        
        normalized_generated_samples = (generated_samples_cycle-xy_mean)/xy_std
        normalized_real_samples = (real_samples_cycle-xy_mean)/xy_std
        
        EMD_score_list.append(EMD_all_pair(normalized_generated_samples, normalized_real_samples, num_coordinate))
        
    return np.array(EMD_score_list)

# def EMD_all_pair_each_X_val(generated_samples, real_samples, num_coordinate, num_of_cycle, num_in_gen_list, num_in_real_list): #여러 X에 대해 각각 쪼개서, 모든 pair(36가지)에 대한 EMD list를 모아서 뱉는 함수.
#     dim = real_samples.shape[1]
    
#     sample_num = int(len(generated_samples)/num_of_cycle)
#     num_in_cycle = int(len(real_samples)/num_of_cycle)
    
    
#     EMD_score_list = []

#     for i in range(num_of_cycle):
#         generated_samples_cycle = generated_samples[i*sample_num:(i+1)*sample_num]
#         real_samples_cycle = real_samples[i*num_in_cycle:(i+1)*num_in_cycle]
        
#         xy_mean = np.mean(real_samples_cycle, axis=0, dtype=np.float32)
#         xy_std = np.std(real_samples_cycle, axis=0, dtype=np.float32)
        
#         normalized_generated_samples = (generated_samples_cycle-xy_mean)/xy_std
#         normalized_real_samples = (real_samples_cycle-xy_mean)/xy_std
        
#         EMD_score_list.append(EMD_all_pair(normalized_generated_samples, normalized_real_samples, num_coordinate))
        
#     return np.array(EMD_score_list)

    
        

    