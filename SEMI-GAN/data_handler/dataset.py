import torch
from torch.utils.data import DataLoader, TensorDataset

from torchvision import transforms


import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np

import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.metrics import r2_score
import os

def load_data(file_path, datatype, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header):  
    """
    
     1) 20191107 기준 : num_input, num_output, num_of_cycle = 185, num_in_cycle=10, header=2, usecols="D:G" 확인 필수
     2) num_input, num_output, num_in_cycle, num_of_cycle 새로 추가함
    
    """
    num_total = num_of_cycle*num_in_cycle
    
    if datatype == 'none':
        data_x = pd.read_excel('./'+file_path, sheet_name='uniformly sampled',usecols=x_cols, nrows=num_total+1, header=header)
        data_y = pd.read_excel('./'+file_path, sheet_name='uniformly sampled', usecols=y_cols, nrows=num_total+1, header=header)
        
        # No one-hot encoding 
        X_all , Y_all = np.zeros((num_total, num_input)), np.zeros((num_total, num_output))
        X_per_cycle, Y_per_cycle = np.zeros((num_of_cycle, num_input)), np.zeros((num_of_cycle, num_output))
        
        # PANDAS TO NUMPY
        # X_per_cycle
        for i in range(num_of_cycle):
            X_per_cycle[i] = data_x[i*num_in_cycle+1:i*num_in_cycle+2].values

        # X_all
        X_all = np.repeat(X_per_cycle,num_in_cycle,axis=0)

        # Y_all
        for i in range(num_total):
            Y_all[i] = data_y[i+1:i+2].values

        # Y_per_cycle
        for i in range(num_of_cycle):
            Y_per_cycle[i] = np.mean(Y_all[i*num_in_cycle:(i+1)*num_in_cycle],axis=0)

    else:
        data_x = pd.read_excel('./'+file_path, sheet_name='Generated DATAs',usecols=x_cols, nrows=num_total+1, header=header)
        data_y = pd.read_excel('./'+file_path, sheet_name='Generated DATAs', usecols=y_cols, nrows=num_total+1, header=header)

        # one-hot encoding (num_input +1)
        X_all , Y_all = np.zeros((num_total, num_input+1)), np.zeros((num_total, num_output))
        X_per_cycle, Y_per_cycle = np.zeros((num_of_cycle, num_input+1)), np.zeros((num_of_cycle, num_output))
 
        # DATA_X DATA_Y preprocessing

        # 1. N, P to 10, 01 (one-hot encoding)
        data_x =pd.get_dummies(data_x, columns=['PNMOS'], dtype=float)

        # 2. Remove unrequired column ( Wfin [nm], alpha )
        # X: Wfin, alpha
        data_x = data_x.drop('Wfin [nm]', axis=1)
        data_x = data_x.drop('alpha', axis=1)

        # Y: IDLO, IDHI, DIBL
        data_y = data_y.drop('IDLO', axis=1)
        data_y = data_y.drop('IDHI', axis=1)
        data_y = data_y.drop('DIBL(mV)', axis=1)
        
        # PANDAS TO NUMPY
        # X_per_cycle
        for i in range(num_of_cycle):    
#            print(data_x[i*num_in_cycle:i*num_in_cycle+1])
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
    print("X data shape: ", X_all.shape, "X per cycle data shape:", X_per_cycle.shape)
    print("Y data shape: ", Y_all.shape, "Y per cycle data shape:", Y_per_cycle.shape)  
    print("any nan in X?: ", np.argwhere(np.isnan(X_all)))
    print("any nan in Y?: ", np.argwhere(np.isnan(Y_all)))
      
    return X_all, Y_all, X_per_cycle, Y_per_cycle

def mean_cov(y_all, num_in_cycle, num_of_cycle, num_output):
    
    print("Y_all size: ", y_all.shape)
    
    y_mean_cov_num = int(2*num_output+((num_output)**2-(num_output))/2)
    # print(y_mean_cov_num)

    y_mean_cov = np.zeros((num_of_cycle, y_mean_cov_num))
    # print(y_mean_cov.shape)
    
    for i in range(num_of_cycle):
        #print(i)        
#        print(i*num_in_cycle, (i+1)*num_in_cycle)
        temp = y_all[i*num_in_cycle:(i+1)*num_in_cycle,:]
    
        mean_y = np.mean(temp, axis=0)
        cov_y = np.cov(temp.T)
        # print(cov_y)
        
        # mean
        y_mean_cov[i, :num_output] = mean_y
        # print(y_mean_cov)
        # diagonal
        y_mean_cov[i, num_output:num_output*2] = np.diagonal(cov_y)
        # print(y_mean_cov)
        
        # covariance
        cnt = num_output*2
        for j in range(1, num_output):
            y_mean_cov[i,cnt:cnt+j] = cov_y[j,:j]
            cnt += j
            
    return y_mean_cov
        
def split_data(x, y, num_train, num_val, num_test):
    
    print(x.shape)
    print(y.shape)
    
    if len(x) == len(y):
        print("Same number of x data and y data")
        len_total = len(x)
    else:
        print("Different number of x data and y data")
    
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train+num_val], y[num_train:num_train+num_val]
    x_test, y_test = x[num_train+num_val:], y[num_train+num_val:]
    
    print("============= Data split ==============")
    print("train X: {} train Y: {}".format(x_train.shape, y_train.shape))
    print("val X: {} val Y: {}".format(x_val.shape, y_val.shape))
    print("test X: {} test Y: {}".format(x_test.shape, y_test.shape))
    
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
        
class SEMI_gan_data(Dataset):
    def __init__(self, name, datatype, num_input, num_output, num_in_cycle, num_of_cycle, num_train, num_val, num_test, x_cols, y_cols, header):
        super().__init__(name)
        
        ##########################   DATASET with no PN type (2019 datas) ########################
     
        if datatype == 'none':
            X_all, Y_all, X_per_cycle, Y_per_cycle = load_data(name, datatype, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header)
        
            self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = split_data(X_all, Y_all, num_train*num_in_cycle, num_val*num_in_cycle, num_test*num_in_cycle)
            self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_per_cycle, num_train, num_val, num_test) 
            

        ##########################   DATASET with PN type (2020 datas) ########################
        
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
            
            self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = split_data(X_all, Y_all, num_train*num_in_cycle//2, num_val*num_in_cycle//2, num_test*num_in_cycle//2)
            self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_per_cycle, num_train//2, num_val//2, num_test//2)        
            

        # use both P, N type
        else:
            
            X_all, Y_all, X_per_cycle, Y_per_cycle = load_data(name, datatype, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header)

            self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = split_data(X_all, Y_all, num_train*num_in_cycle, num_val*num_in_cycle, num_test*num_in_cycle)
            self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_per_cycle, num_train, num_val, num_test)             
        
        # STEP 2: Split data
        
        # OPTIONAL: Split data for Y_mean, Y_noise
        
        self.train_Y_mean = np.repeat(self.train_Y_per_cycle, num_in_cycle, axis=0)
        self.val_Y_mean = np.repeat(self.val_Y_per_cycle, num_in_cycle, axis=0)
        self.test_Y_mean = np.repeat(self.test_Y_per_cycle, num_in_cycle, axis=0)
        
        print("train_Y_mean shape", self.train_Y_mean.shape)
        print("val_Y_mean shape", self.val_Y_mean.shape)
        print("test_Y_mean shape", self.test_Y_mean.shape)
        
        self.train_Y_noise = self.train_Y - self.train_Y_mean
        self.val_Y_noise = self.val_Y - self.val_Y_mean
        self.test_Y_noise = self.test_Y - self.test_Y_mean
        
        print("train_Y_noise shape", self.train_Y_noise.shape)
        print("val_Y_noise shape", self.val_Y_noise.shape)
        print("test_Y_noise shape", self.test_Y_noise.shape)    

"""
class SEMI_gaussian_data(Dataset):
    def __init__(self, name, datatype, num_input, num_output, num_in_cycle, num_of_cycle, num_train, num_val, num_test, x_cols, y_cols, header):
        super().__init__(name)
        
        ##########################   DATASET with no PN type (2019 datas) ########################
     
        if datatype == 'none':
            X_all, Y_all, X_per_cycle, Y_per_cycle = load_data(name, datatype, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header)
        
            self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = split_data(X_all, Y_all, num_train*num_in_cycle, num_val*num_in_cycle, num_test*num_in_cycle)
            self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_per_cycle, num_train, num_val, num_test) 
            

        ##########################   DATASET with PN type (2020 datas) ########################
        
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
            
            # Changing Target Y variable : ( N, num_output) => ( N, mean + covariance )
            Y_mean_cov = mean_cov(Y_all, num_in_cycle, num_of_cycle, num_output)
        
            self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_mean_cov, num_train//2, num_val//2, num_test//2) 
            
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

            # Changing Target Y variable : ( N, num_output) => ( N, mean + covariance )
            Y_mean_cov = mean_cov(Y_all, num_in_cycle, num_of_cycle//2, num_output)

            self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_mean_cov, num_train//2, num_val//2, num_test//2)        
            
        # use both P, N type
        else:
            
            X_all, Y_all, X_per_cycle, Y_per_cycle = load_data(name, datatype, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header)
            
            # Changing Target Y variable : ( N, num_output) => ( N, mean + covariance )
            Y_mean_cov = mean_cov(Y_all, num_in_cycle, num_of_cycle, num_output)

            self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_mean_cov, num_train, num_val, num_test)             
        
        # STEP 2: Split data
        
        # OPTIONAL: Split data for Y_mean, Y_noise
        

"""

        