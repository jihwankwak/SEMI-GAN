import torch
from torch.utils.data import DataLoader, TensorDataset

from torchvision import transforms
from scipy import linalg

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

def get_dataset(name, datatype):
    if name == 'LER_data_20191125.xlsx':
        return SEMI_data(name, datatype, num_input=4, num_output=8, num_in_cycle=10, num_of_cycle=270, num_train=230, num_val=20, num_test=20, x_cols="D:G", y_cols="K:S", header=2)
    elif name == 'LER_data_20191107.xlsx':
        return SEMI_data(name, datatype, num_input=4, num_output=8, num_in_cycle=10, num_of_cycle=185, num_train=150, num_val=15 , num_test=20, x_cols="D:G", y_cols="K:S", header=2)
    elif name == '2020_LER_20200529_V004.xlsx':
        return SEMI_data(name, datatype, num_input=4, num_output=6, num_in_cycle=50, num_of_cycle=72, num_train=50, num_val=10, num_test=12, x_cols="D:G", y_cols="H:P", header=0)
    elif name == '2020_LER_20200804_V006.xlsx':
        return SEMI_data(name, datatype, num_input=4, num_output=6, num_in_cycle=50, num_of_cycle=200, num_train=150, num_val=20, num_test=30, x_cols="B:G", y_cols="H:P", header=0)
    elif name == '2020_LER_20200922_V007_testset_edit.csv':
        return SEMI_data(name, datatype, num_input=4, num_output=6, num_in_cycle=50, num_of_cycle=236, num_train=88*2, num_val=15*2, num_test=15*2, x_cols=['PNMOS', 'amp.', 'corr.x', 'corr.y'], y_cols=['Ioff', 'IDSAT', 'IDLIN', 'VTSAT', 'VTLIN', 'SS'], header=0)
    
def get_dataset_test(name, datatype):
    if name == '2020_LER_20200922_testset.csv':
        return SEMI_sample_data(name, num_input=4, num_output=6, num_in_cycle=[232, 289, 277, 253, 255], num_of_cycle=5, x_cols=['PNMOS', 'amp.', 'corr.x', 'corr.y'], y_cols=['Ioff', 'IDSAT', 'IDLIN', 'VTSAT', 'VTLIN', 'SS'], header=0)
    elif name == '2020_LER_20200804_testset.csv':
        return SEMI_sample_data(name, num_input=4, num_output=6, num_in_cycle=[50, 50, 50, 50, 50], num_of_cycle=5, x_cols=['PNMOS', 'amp.', 'corr.x', 'corr.y'], y_cols=['Ioff', 'IDSAT', 'IDLIN', 'VTSAT', 'VTLIN', 'SS'], header=0)    

        
def load_data(file_path, datatype, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header):  
    """
    
     1) 20191107 기준 : num_input, num_output, num_of_cycle = 185, num_in_cycle=10, header=2, usecols="D:G" 확인 필수
     2) num_input, num_output, num_in_cycle, num_of_cycle 새로 추가함
    
    """
    num_total = num_of_cycle*num_in_cycle
    
    if datatype == 'none':
        data_x = pd.read_csv('../'+file_path, header=0, usecols=x_cols)
        data_y = pd.read_csv('../'+file_path, header=0, usecols=y_cols)
        
        # No one-hot encoding 
        X_all , Y_all = np.zeros((num_total, num_input)), np.zeros((num_total, num_output))
        X_per_cycle, Y_per_cycle = np.zeros((num_of_cycle, num_input)), np.zeros((num_of_cycle, num_output))
        
        # PANDAS TO NUMPY
        # X_per_cycle
        for i in range(num_of_cycle):
            X_per_cycle[i] = data_x[i*num_in_cycle+1:i*num_in_cycle+2].values

        # X_all
        X_all = np.repeat(X_per_cycle,num_in_cycle,axis=0)
        for i in range(X_all.shape[0]):
            print(X_all[i])


        # Y_all
        for i in range(num_total):
            Y_all[i] = data_y[i+1:i+2].values

        # Y_per_cycle
        for i in range(num_of_cycle):
            Y_per_cycle[i] = np.mean(Y_all[i*num_in_cycle:(i+1)*num_in_cycle],axis=0)

    else:
        data_x = pd.read_csv('../'+file_path, header=0, usecols=x_cols)
        data_y = pd.read_csv('../'+file_path, header=0, usecols=y_cols)

        # one-hot encoding (num_input +1)
        X_all , Y_all = np.zeros((num_total, num_input+1)), np.zeros((num_total, num_output))
        X_per_cycle, Y_per_cycle = np.zeros((num_of_cycle, num_input+1)), np.zeros((num_of_cycle, num_output))
 
        # DATA_X DATA_Y preprocessing

        # 1. N, P to 10, 01 (one-hot encoding)
        data_x =pd.get_dummies(data_x, columns=['PNMOS'], dtype=float)

#         # 2. Remove unrequired column ( Wfin [nm], alpha )
#         # X: Wfin, alpha
#         data_x = data_x.drop('Wfin [nm]', axis=1)
#         data_x = data_x.drop('alpha', axis=1)

#         # Y: IDLO, IDHI, DIBL
#         data_y = data_y.drop('IDLO', axis=1)
#         data_y = data_y.drop('IDHI', axis=1)
#         data_y = data_y.drop('DIBL(mV)', axis=1)
        
        # PANDAS TO NUMPY
        # X_per_cycle
        for i in range(num_of_cycle):    
            #print(data_x[i*num_in_cycle:i*num_in_cycle+1])
            X_per_cycle[i] = data_x[i*num_in_cycle:i*num_in_cycle+1].values

        # X_all
        X_all = np.repeat(X_per_cycle,num_in_cycle,axis=0)

        # DATA_Y preprcoessing

        # Y_all
        for i in range(num_total):
            Y_all[i] = data_y[i:i+1].values

        # Y_per_cycle    
        for i in range(num_of_cycle):
            Y_per_cycle[i] = np.mean(Y_all[i*num_in_cycle:(i+1)*num_in_cycle],axis=0)
        
    print("============ Data load =============")
    print("STEP 1: CHECK DATA SIZE")
    print("X data shape: ", X_all.shape, "X per cycle data shape:", X_per_cycle.shape)
    print("Y data shape: ", Y_all.shape, "Y per cycle data shape:", Y_per_cycle.shape)  
    print("STEP 2: CHECK NAN DATA")
    print("any nan in X?: ", np.argwhere(np.isnan(X_all)))
    print("any nan in Y?: ", np.argwhere(np.isnan(Y_all)))
    print()
    
    return X_all, Y_all, X_per_cycle, Y_per_cycle

def load_sample_data(file_path, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header):  
    """
    
     1) 20191107 기준 : num_input, num_output, num_of_cycle = 185, num_in_cycle=10, header=2, usecols="D:G" 확인 필수
     2) num_input, num_output, num_in_cycle, num_of_cycle 새로 추가함
    
    """
    num_total = sum(num_in_cycle)
    print(num_total)
    
    data_x = pd.read_csv('../'+file_path, header=0, usecols=x_cols)
    data_y = pd.read_csv('../'+file_path, header=0, usecols=y_cols)
        
    data_x =pd.get_dummies(data_x, columns=['PNMOS'])
    print(data_x)
    print(data_y)
   
    num_input = data_x.shape[1]
    
    X_all , Y_all = np.zeros((num_total, num_input)), np.zeros((num_total, num_output))
    X_per_cycle, Y_per_cycle = np.zeros((num_of_cycle, num_input+1)), np.zeros((num_of_cycle, num_output))
    
    X_all = data_x.values
    X_all = np.hstack([X_all, np.zeros((X_all.shape[0], 1))])
    Y_all = data_y.values
    
    # X_per_cycle
   
    idx = 0
    add = 0     
    for i in range(num_of_cycle):
        add = num_in_cycle[i]
        X_per_cycle[i] = np.mean(X_all[idx:idx+add], axis=0)
        Y_per_cycle[i] = np.mean(Y_all[idx:idx+add], axis=0)
        idx += add                
        
    print("============ Data load =============")
    print("X data shape: ", X_all.shape, "X per cycle data shape:", X_per_cycle.shape)
    print("Y data shape: ", Y_all.shape, "Y per cycle data shape:", Y_per_cycle.shape)  
    print("any nan in X?: ", np.argwhere(np.isnan(X_all)))
    print("any nan in Y?: ", np.argwhere(np.isnan(Y_all)))
      
    return X_all, Y_all, X_per_cycle, Y_per_cycle

def split_data(x, y, num_train, num_val, num_test):
       
    if len(x) == len(y):
        print("Same number of x data and y data")
        len_total = len(x)
    else:
        print("Different number of x data and y data")
    
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train+num_val], y[num_train:num_train+num_val]
    x_test, y_test = x[num_train+num_val:], y[num_train+num_val:]
    
    print("train X: {} train Y: {}".format(x_train.shape, y_train.shape))
    print("val X: {} val Y: {}".format(x_val.shape, y_val.shape))
    print("test X: {} test Y: {}".format(x_test.shape, y_test.shape))

    y_train_mean = np.mean(y_train, axis=0, dtype=np.float32)
    y_train_std = np.std(y_train, axis=0, dtype=np.float32)
    
    x_train_mean = np.mean(x_train, axis=0, dtype=np.float32)
    x_train_std = np.std(x_train, axis=0, dtype=np.float32)
    
    print("x mean, std: ", x_train_mean, x_train_std)
    print("y mean, std: ", y_train_mean, y_train_std)
    
    return x_train, y_train, x_val, y_val, x_test, y_test
    

class Dataset():   
    def __init__(self, name):
        
        self.train_X = None
        self.val_X = None
        self.test_X = None        
        
        self.train_Y = None
        self.val_Y = None
        self.test_Y = None
        
        self.train_X_per_cycle = None
        self.val_X_per_cycle = None
        self.test_X_per_cycle = None
              
        self.train_Y_per_cycle = None
        self.val_Y_per_cycle = None
        self.test_Y_per_cycle = None
        
        self.train_Y_mean= None
        self.val_Y_mean = None
        self.test_Y_mean = None
        
        self.train_Y_noise = None
        self.val_Y_noise = None
        self.test_Y_noise = None

class SEMI_data(Dataset):
    def __init__(self, name, datatype, num_input, num_output, num_in_cycle, num_of_cycle, num_train, num_val, num_test, x_cols, y_cols, header):
        super().__init__(name)
        
        # DATASET with no PN type (2019 datas)
     
        if datatype == 'none':
            X_all, Y_all, X_per_cycle, Y_per_cycle = load_data(name, datatype, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header)
        
            self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = split_data(X_all, Y_all, num_train*num_in_cycle, num_val*num_in_cycle, num_test*num_in_cycle)
            self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_per_cycle, num_train, num_val, num_test) 
            

        # DATASET with PN type (2020 datas)
        
        # use P type
        elif datatype == 'p':
            X_all_temp, Y_all_temp, X_per_cycle_temp, Y_per_cycle_temp = load_data(name, datatype, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header)
            
            X_all = np.empty(num_input+1)
            Y_all= np.empty(num_output)
            
            for i in range(num_of_cycle):
                X_all = np.vstack((X_all, X_all_temp[num_in_cycle*(2*i+1):num_in_cycle*(2*i+1)+num_in_cycle]))
                X_per_cycle = X_per_cycle_temp[1::2]
                Y_all = np.vstack((Y_all, Y_all_temp[num_in_cycle*(2*i+1):num_in_cycle*(2*i+1)+num_in_cycle]))
                Y_per_cycle = Y_per_cycle_temp[1::2]
                
            X_all = X_all[1:]
            Y_all = Y_all[1:]
        
            self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = split_data(X_all, Y_all, num_train*num_in_cycle//2, num_val*num_in_cycle//2, num_test*num_in_cycle//2)
            self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_per_cycle, num_train//2, num_val//2, num_test//2) 
            
        # use N type
        elif datatype == 'n': 
            X_all_temp, Y_all_temp, X_per_cycle_temp, Y_per_cycle_temp = load_data(name, datatype, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header)
            
            X_all = np.empty(num_input+1)
            Y_all= np.empty(num_output)
            
            for i in range(num_of_cycle):
                X_all = np.vstack((X_all, X_all_temp[num_in_cycle*(2*i):num_in_cycle*(2*i)+num_in_cycle]))
                X_per_cycle = X_per_cycle_temp[::2]
                Y_all = np.vstack((Y_all, Y_all_temp[num_in_cycle*(2*i):num_in_cycle*(2*i)+num_in_cycle]))
                Y_per_cycle = Y_per_cycle_temp[::2]
            
            X_all = X_all[1:]
            Y_all = Y_all[1:]                                       

            # print(X_all.shape, Y_all.shape, X_per_cycle.shape, Y_per_cycle.shape)
            
            print("============= Data split ==============")

            print("STEP1: split All data")
            self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = split_data(X_all, Y_all, num_train*num_in_cycle//2, num_val*num_in_cycle//2, num_test*num_in_cycle//2)
            print()
            print("STEP2: split cycle data")
            self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_per_cycle, num_train//2, num_val//2, num_test//2)        
        # use both P, N type
        else:
            
            X_all, Y_all, X_per_cycle, Y_per_cycle = load_data(name, datatype, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header)

            self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = split_data(X_all, Y_all, num_train*num_in_cycle, num_val*num_in_cycle, num_test*num_in_cycle)
            self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_per_cycle, num_train, num_val, num_test)             
            
class SEMI_sample_data(Dataset):
    def __init__(self, name, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header):
        super().__init__(name)
        
        X_all, Y_all, X_per_cycle, Y_per_cycle = load_sample_data(name, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header)
        
        self.test_X = X_all
        self.test_Y = Y_all
        self.test_X_per_cycle = X_per_cycle
        self.test_Y_per_cycle = X_per_cycle
        
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

    
        

    