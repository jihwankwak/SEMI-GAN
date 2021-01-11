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

# def EMD_from_2d_kde_integral(generated_samples, real_samples, index_x, index_y, num_coordinate): #각 joint pair 별로 2d sample을 kde로 fittining시키고 얻은 pdf값을 weight로 삼아 histogram 만들어서 EMD

#     x_min = min(min(real_samples[:,index_x]), min(generated_samples[:,index_x]))
#     x_max = max(max(real_samples[:,index_x]), max(generated_samples[:,index_x]))
#     y_min = min(min(real_samples[:,index_y]), min(generated_samples[:,index_y]))
#     y_max = max(max(real_samples[:,index_y]), max(generated_samples[:,index_y]))

#     position_cartesian = np.array(np.meshgrid(np.linspace(x_min, x_max, num_coordinate), np.linspace(y_min, y_max, num_coordinate))).T.reshape(-1,2)

#     real_samples_2d = real_samples[:,[index_x, index_y]]
#     generated_samples_2d = generated_samples[:,[index_x, index_y]]

#     kde_real = stats.gaussian_kde(real_samples_2d.T)
#     kde_gen = stats.gaussian_kde(generated_samples_2d.T)

#     # Estimate distribution
#     # density_real = kde_real(position_cartesian.T) #2by100 입력 -> 1by100 출력
#     # density_gen = kde_gen(position_cartesian.T)



#     # Estimate probability
#     interval_x = (x_max - x_min)/num_coordinate
#     interval_y = (y_max - y_min)/num_coordinate
#     interval = np.array([interval_x, interval_y])

#     integral_real = np.zeros((num_coordinate**2, 1))
#     integral_gen = np.zeros((num_coordinate**2, 1))
#     for i in range(num_coordinate**2):
#         integral_real[i] = kde_real.integrate_box(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)
#         integral_gen[i] = kde_gen.integrate_box(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)

#     #clipping
#     integral_real = np.maximum(integral_real,0)
#     integral_gen = np.maximum(integral_gen,0)


#     real_weight_position = np.concatenate((integral_real, position_cartesian),axis=1).astype('float32')
#     gen_weight_position = np.concatenate((integral_gen, position_cartesian),axis=1).astype('float32')


#     # Compare
#     #KL_div = KL(density_real, density_gen)
#     #print('KL_div', KL_div)

#     EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)

#     return EMD_score

# def EMD_from_1d_kde_integral(generated_samples, real_samples, index_x, num_coordinate): # 적분 weighting 버전

#     x_min = min(min(real_samples[:,index_x]), min(generated_samples[:,index_x]))
#     x_max = max(max(real_samples[:,index_x]), max(generated_samples[:,index_x]))


#     position_cartesian = np.linspace(x_min, x_max, num_coordinate).reshape(-1,1)

#     real_samples_1d = real_samples[:,[index_x]]
#     generated_samples_1d = generated_samples[:,[index_x]]

#     kde_real = stats.gaussian_kde(real_samples_1d.T)
#     kde_gen = stats.gaussian_kde(generated_samples_1d.T)

#     # Estimate distribution
#     #density_real = kde_real(position_cartesian.T)
#     #density_gen = kde_gen(position_cartesian.T)


#     # Estimate probability
#     interval = (x_max - x_min)/num_coordinate
#     integral_real = np.zeros((num_coordinate,1))
#     integral_gen = np.zeros((num_coordinate,1))
#     for i in range(num_coordinate):
#         integral_real[i] = kde_real.integrate_box_1d(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)
#         integral_gen[i] = kde_gen.integrate_box_1d(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)

#     #real_weight_position = np.concatenate((density_real.reshape(-1,1), position_cartesian),axis=1).astype('float32')
#     #gen_weight_position = np.concatenate((density_gen.reshape(-1,1), position_cartesian),axis=1).astype('float32')
#     real_weight_position = np.concatenate((integral_real, position_cartesian),axis=1).astype('float32')
#     gen_weight_position = np.concatenate((integral_gen, position_cartesian),axis=1).astype('float32')

#     # Compare
#     #KL_div = KL(density_real, density_gen)
#     #print('KL_div', KL_div)

#     EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)

#     return EMD_score

# def EMD_1d_2d_list_integral(normalized_generated_samples, normalized_real_samples, num_coordinate): #하나의 X에 대해, 모든 pair(36가지) EMD를 list로 뱉는 함수.

#     dim = normalized_real_samples.shape[1]
#     EMD_1d_score_list = []
#     EMD_2d_score_list = []

#     for i in range(dim):
#         for j in range(dim):
#             if i == j :
#                 EMD_1d_score_list.append(EMD_from_1d_kde_integral(normalized_generated_samples, normalized_real_samples, i, num_coordinate))
#             elif i < j : 
#                 EMD_2d_score_list.append(EMD_from_2d_kde_integral(normalized_generated_samples, normalized_real_samples, i, j, num_coordinate))

#     return np.array(EMD_1d_score_list), np.array(EMD_2d_score_list)


# def EMD_all_pair_each_X_integral(generated_samples, real_samples, num_coordinate, num_of_cycle, num_in_gen_list, num_in_real_list): #여러 X에 대해 각각 쪼개서, 모든 pair(36가지)에 대한 EMD list를 모아서 뱉는 함수.
#     dim = real_samples.shape[1]
    
#     #EMD_score_list = []
#     EMD_1d_score_list = []
#     EMD_2d_score_list = []

#     for i in range(num_of_cycle):
#         generated_samples_cycle = generated_samples[sum(num_in_gen_list[:i]):sum(num_in_gen_list[:i])+num_in_gen_list[i]]
#         real_samples_cycle = real_samples[sum(num_in_real_list[:i]):sum(num_in_real_list[:i])+num_in_real_list[i]]
        
#         xy_mean = np.mean(real_samples_cycle, axis=0, dtype=np.float32)
#         xy_std = np.std(real_samples_cycle, axis=0, dtype=np.float32)
        
#         normalized_generated_samples = (generated_samples_cycle-xy_mean)/xy_std
#         normalized_real_samples = (real_samples_cycle-xy_mean)/xy_std
        
#         EMD_1d_score_list.append(EMD_1d_2d_list_integral(normalized_generated_samples, normalized_real_samples, num_coordinate)[0])
#         EMD_2d_score_list.append(EMD_1d_2d_list_integral(normalized_generated_samples, normalized_real_samples, num_coordinate)[1])
        
#     return np.array(EMD_1d_score_list), np.array(EMD_2d_score_list)


# def EMD_from_2d_kde_pdf(generated_samples, real_samples, index_x, index_y, num_coordinate): #각 joint pair 별로 2d sample을 kde로 fittining시키고 얻은 pdf값을 weight로 삼아 histogram 만들어서 EMD

#     x_min = min(min(real_samples[:,index_x]), min(generated_samples[:,index_x]))
#     x_max = max(max(real_samples[:,index_x]), max(generated_samples[:,index_x]))
#     y_min = min(min(real_samples[:,index_y]), min(generated_samples[:,index_y]))
#     y_max = max(max(real_samples[:,index_y]), max(generated_samples[:,index_y]))

#     position_cartesian = np.array(np.meshgrid(np.linspace(x_min, x_max, num_coordinate), np.linspace(y_min, y_max, num_coordinate))).T.reshape(-1,2)
    
#     real_samples_2d = real_samples[:,[index_x, index_y]]
#     generated_samples_2d = generated_samples[:,[index_x, index_y]]
    
#     kde_real = stats.gaussian_kde(real_samples_2d.T)
#     kde_gen = stats.gaussian_kde(generated_samples_2d.T)
    
#     # Estimate distribution
#     density_real = kde_real(position_cartesian.T)
#     density_gen = kde_gen(position_cartesian.T)
    
#     real_weight_position = np.concatenate((density_real.reshape(-1,1), position_cartesian),axis=1).astype('float32')
#     gen_weight_position = np.concatenate((density_gen.reshape(-1,1), position_cartesian),axis=1).astype('float32')

#     # Compare
#     #KL_div = KL(density_real, density_gen)
#     #print('KL_div', KL_div)

#     EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)
    
#     return EMD_score


# def EMD_from_1d_kde_pdf(generated_samples, real_samples, index_x, num_coordinate): #대각 성분에 대해서는 1d kde로..

#     x_min = min(min(real_samples[:,index_x]), min(generated_samples[:,index_x]))
#     x_max = max(max(real_samples[:,index_x]), max(generated_samples[:,index_x]))
    

#     position_cartesian = np.linspace(x_min, x_max, num_coordinate).reshape(-1,1)
    
#     real_samples_1d = real_samples[:,[index_x]]
#     generated_samples_1d = generated_samples[:,[index_x]]
    
#     kde_real = stats.gaussian_kde(real_samples_1d.T)
#     kde_gen = stats.gaussian_kde(generated_samples_1d.T)
    
#     # Estimate distribution
#     density_real = kde_real(position_cartesian.T)
#     density_gen = kde_gen(position_cartesian.T)
    
#     real_weight_position = np.concatenate((density_real.reshape(-1,1), position_cartesian),axis=1).astype('float32')
#     gen_weight_position = np.concatenate((density_gen.reshape(-1,1), position_cartesian),axis=1).astype('float32')

#     # Compare
#     #KL_div = KL(density_real, density_gen)
#     #print('KL_div', KL_div)

#     EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)
    
#     return EMD_score

# def EMD_1d_2d_list_pdf(normalized_generated_samples, normalized_real_samples, num_coordinate): #하나의 X에 대해, 모든 pair(36가지) EMD를 list로 뱉는 함수.

#     dim = normalized_real_samples.shape[1]
#     EMD_1d_score_list = []
#     EMD_2d_score_list = []

#     for i in range(dim):
#         for j in range(dim):
#             if i == j :
#                 EMD_1d_score_list.append(EMD_from_1d_kde_pdf(normalized_generated_samples, normalized_real_samples, i, num_coordinate))
#             elif i < j : 
#                 EMD_2d_score_list.append(EMD_from_2d_kde_pdf(normalized_generated_samples, normalized_real_samples, i, j, num_coordinate))

#     return np.array(EMD_1d_score_list), np.array(EMD_2d_score_list)


# def EMD_all_pair_each_X_pdf(generated_samples, real_samples, num_coordinate, num_of_cycle, num_in_gen_list, num_in_real_list): #여러 X에 대해 각각 쪼개서, 모든 pair(36가지)에 대한 EMD list를 모아서 뱉는 함수.
#     dim = real_samples.shape[1]
    
#     #EMD_score_list = []
#     EMD_1d_score_list = []
#     EMD_2d_score_list = []

#     for i in range(num_of_cycle):
#         generated_samples_cycle = generated_samples[sum(num_in_gen_list[:i]):sum(num_in_gen_list[:i])+num_in_gen_list[i]]
#         real_samples_cycle = real_samples[sum(num_in_real_list[:i]):sum(num_in_real_list[:i])+num_in_real_list[i]]
        
#         xy_mean = np.mean(real_samples_cycle, axis=0, dtype=np.float32)
#         xy_std = np.std(real_samples_cycle, axis=0, dtype=np.float32)
        
#         normalized_generated_samples = (generated_samples_cycle-xy_mean)/xy_std
#         normalized_real_samples = (real_samples_cycle-xy_mean)/xy_std
        
#         EMD_1d_score_list.append(EMD_1d_2d_list_pdf(normalized_generated_samples, normalized_real_samples, num_coordinate)[0])
#         EMD_2d_score_list.append(EMD_1d_2d_list_pdf(normalized_generated_samples, normalized_real_samples, num_coordinate)[1])
        
#     return np.array(EMD_1d_score_list), np.array(EMD_2d_score_list)


def EMD_from_2d_kde_integral(normalized_generated_samples, normalized_real_samples, index_x, index_y, num_coordinate, normalized_x_min, normalized_x_max, normalized_y_min, normalized_y_max): #각 joint pair 별로 2d sample을 kde로 fittining시키고 얻은 pdf값을 weight로 삼아 histogram 만들어서 EMD

    position_cartesian = np.array(np.meshgrid(np.linspace(normalized_x_min, normalized_x_max, num_coordinate), np.linspace(normalized_y_min, normalized_x_max, num_coordinate))).T.reshape(-1,2)

    normalized_real_samples_2d = normalized_real_samples[:,[index_x, index_y]]
    normalized_generated_samples_2d = normalized_generated_samples[:,[index_x, index_y]]

    kde_real = stats.gaussian_kde(normalized_real_samples_2d.T)
    kde_gen = stats.gaussian_kde(normalized_generated_samples_2d.T)

    # Estimate distribution
    # density_real = kde_real(position_cartesian.T) #2by100 입력 -> 1by100 출력
    # density_gen = kde_gen(position_cartesian.T)



    # Estimate probability
    interval_x = (normalized_x_max - normalized_x_min)/num_coordinate
    interval_y = (normalized_y_max - normalized_y_min)/num_coordinate
    interval = np.array([interval_x, interval_y])

    integral_real = np.zeros((num_coordinate**2, 1))
    integral_gen = np.zeros((num_coordinate**2, 1))
    for i in range(num_coordinate**2):
        integral_real[i] = kde_real.integrate_box(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)
        integral_gen[i] = kde_gen.integrate_box(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)

    #clipping
    integral_real = np.maximum(integral_real,0)
    integral_gen = np.maximum(integral_gen,0)


    real_weight_position = np.concatenate((integral_real, position_cartesian),axis=1).astype('float32')
    gen_weight_position = np.concatenate((integral_gen, position_cartesian),axis=1).astype('float32')


    # Compare
    #KL_div = KL(density_real, density_gen)
    #print('KL_div', KL_div)

    EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)

    return EMD_score

def EMD_from_1d_kde_integral(normalized_generated_samples, normalized_real_samples, index_x, num_coordinate, normalized_x_min, normalized_x_max): # 적분 weighting 버전

    position_cartesian = np.linspace(normalized_x_min, normalized_x_max, num_coordinate).reshape(-1,1)

    normalized_real_samples_1d = normalized_real_samples[:,[index_x]]
    normalized_generated_samples_1d = normalized_generated_samples[:,[index_x]]

    kde_real = stats.gaussian_kde(normalized_real_samples_1d.T)
    kde_gen = stats.gaussian_kde(normalized_generated_samples_1d.T)

    # Estimate distribution
    #density_real = kde_real(position_cartesian.T)
    #density_gen = kde_gen(position_cartesian.T)


    # Estimate probability
    interval = (normalized_x_max - normalized_x_min)/num_coordinate
    integral_real = np.zeros((num_coordinate,1))
    integral_gen = np.zeros((num_coordinate,1))
    for i in range(num_coordinate):
        integral_real[i] = kde_real.integrate_box_1d(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)
        integral_gen[i] = kde_gen.integrate_box_1d(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)

    #real_weight_position = np.concatenate((density_real.reshape(-1,1), position_cartesian),axis=1).astype('float32')
    #gen_weight_position = np.concatenate((density_gen.reshape(-1,1), position_cartesian),axis=1).astype('float32')
    real_weight_position = np.concatenate((integral_real, position_cartesian),axis=1).astype('float32')
    gen_weight_position = np.concatenate((integral_gen, position_cartesian),axis=1).astype('float32')

    # Compare
    #KL_div = KL(density_real, density_gen)
    #print('KL_div', KL_div)

    EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)

    return EMD_score




def EMD_1d_2d_list_integral(normalized_generated_samples, normalized_real_samples, num_coordinate, noramlized_min_list, noramlized_max_list): #하나의 X에 대해, 모든 pair(36가지) EMD를 list로 뱉는 함수.

    dim = normalized_real_samples.shape[1]
    EMD_1d_score_list = []
    EMD_2d_score_list = []

    for i in range(dim):
        for j in range(dim):
            if i == j :
                EMD_1d_score_list.append(EMD_from_1d_kde_integral(normalized_generated_samples, normalized_real_samples, i, num_coordinate, noramlized_min_list[i], noramlized_max_list[i]))
            elif i < j : 
                EMD_2d_score_list.append(EMD_from_2d_kde_integral(normalized_generated_samples, normalized_real_samples, i, j, num_coordinate, noramlized_min_list[i], noramlized_max_list[i], noramlized_min_list[j], noramlized_max_list[j]))

    return np.array(EMD_1d_score_list), np.array(EMD_2d_score_list)


def EMD_all_pair_each_X_integral(generated_samples, real_samples, num_coordinate, num_of_cycle, num_in_gen_list, num_in_real_list, min_list, max_list): #여러 X에 대해 각각 쪼개서, 모든 pair(36가지)에 대한 EMD list를 모아서 뱉는 함수.
    dim = real_samples.shape[1]
    
    #EMD_score_list = []
    EMD_1d_score_list = []
    EMD_2d_score_list = []
    
    xy_mean = np.mean(real_samples, axis=0, dtype=np.float32)
    xy_std = np.std(real_samples, axis=0, dtype=np.float32)
    normalized_min_list = (min_list-xy_mean)/xy_std
    normalized_max_list = (max_list-xy_mean)/xy_std
    
    for i in range(num_of_cycle):
        generated_samples_cycle = generated_samples[sum(num_in_gen_list[:i]):sum(num_in_gen_list[:i])+num_in_gen_list[i]]
        real_samples_cycle = real_samples[sum(num_in_real_list[:i]):sum(num_in_real_list[:i])+num_in_real_list[i]]
        
        normalized_generated_samples = (generated_samples_cycle-xy_mean)/xy_std
        normalized_real_samples = (real_samples_cycle-xy_mean)/xy_std

        EMD_1d_score_list.append(EMD_1d_2d_list_integral(normalized_generated_samples, normalized_real_samples, num_coordinate, normalized_min_list, normalized_max_list)[0])
        EMD_2d_score_list.append(EMD_1d_2d_list_integral(normalized_generated_samples, normalized_real_samples, num_coordinate, normalized_min_list, normalized_max_list)[1])
        
    return np.array(EMD_1d_score_list), np.array(EMD_2d_score_list)
