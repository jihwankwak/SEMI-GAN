# import torch
# from torch.utils.data import DataLoader, TensorDataset

# from torchvision import transforms

import pandas as pd
# from pandas import ExcelWriter
# from pandas import ExcelFile
import numpy as np

# import matplotlib.pyplot as plt
# import math
# import sklearn
# from sklearn.metrics import r2_score
import os

def load_xlsx_data(file_path, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header):  
    """
    
     1) 20201008 기준 : num_input, num_output, num_of_cycle = 127, num_in_cycle=50, header=0, usecols="D:F" 확인 필수
     2) num_input, num_output, num_in_cycle, num_of_cycle 새로 추가함
    
    """
    num_total = num_of_cycle*num_in_cycle
    
    # load npy data file 
    if os.path.isfile(file_path+'.npy'):
        print(file_path, "alreday exists. Let's load from npy")            
        data = np.load(file_path+'.npy', allow_pickle=True)
    
        X_all, Y_all, X_per_cycle, Y_per_cycle = data[0], data[1], data[2], data[3]
        
    # load excel file        
    else:
        data_x = pd.read_excel(file_path, sheet_name='Generated DATAs', usecols=x_cols, nrows=num_total+1, header=header)
        data_y = pd.read_excel(file_path, sheet_name='Generated DATAs', usecols=y_cols, nrows=num_total+1, header=header)
        
        X_all , Y_all = np.zeros((num_total, num_input)), np.zeros((num_total, num_output))
        X_per_cycle, Y_per_cycle = np.zeros((num_of_cycle, num_input)), np.zeros((num_of_cycle, num_output))
        
        # DATA_X DATA_Y preprocessing

        # Y: IDLO, IDHI, DIBL
        data_y = data_y.drop('IDLO', axis=1)
        data_y = data_y.drop('IDHI', axis=1)
        data_y = data_y.drop('DIBL(mV)', axis=1)

        for i in range(num_of_cycle):         
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

        data = []
        data.append(X_all)
        data.append(Y_all)
        data.append(X_per_cycle)
        data.append(Y_per_cycle)

        data = np.array(data)
        np.save(file_path, data)
        print(file_path, "npy file saved!!")
        
        print("============ Data load =============")
        print("X data shape: ", X_all.shape, "X per cycle data shape:", X_per_cycle.shape)
        print("Y data shape: ", Y_all.shape, "Y per cycle data shape:", Y_per_cycle.shape)  
        print("any nan in X?: ", np.argwhere(np.isnan(X_all)))
        print("any nan in Y?: ", np.argwhere(np.isnan(Y_all)))
        
    return X_all, Y_all, X_per_cycle, Y_per_cycle

def load_sample_data(file_path, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header):  

    num_total = sum(num_in_cycle)

    # load npy data file   
    if os.path.isfile(file_path+'.npy'):
        print(file_path, "alreday exists. Let's load from npy")            
        
        data = np.load(file_path+'.npy', allow_pickle=True)
    
        X_all, Y_all, X_per_cycle, Y_per_cycle = data[0], data[1], data[2], data[3]
        
    # load excel file        
    else:
        
        num_total = sum(num_in_cycle)

        data_x = pd.read_excel(file_path, sheet_name='Generated DATAs', usecols=x_cols, nrows=num_total+1, header=header)
        data_y = pd.read_excel(file_path, sheet_name='Generated DATAs', usecols=y_cols, nrows=num_total+1, header=header)
        
        # DATA_X preprocessing
        # 1. Remove unrequired column  (Y)
        data_y = data_y.drop('IDLO', axis=1)
        data_y = data_y.drop('IDHI', axis=1)
        data_y = data_y.drop('DIBL(mV)', axis=1)
            
        X_all , Y_all = np.zeros((num_total, num_input)), np.zeros((num_total, num_output))
        X_per_cycle, Y_per_cycle = np.zeros((num_of_cycle, num_input)), np.zeros((num_of_cycle, num_output))   

        X_all = data_x.values
        Y_all = data_y.values

        # X_per_cycle

        idx = 0
        add = 0     
        for i in range(num_of_cycle):
            add = num_in_cycle[i]
            
            X_per_cycle[i] = X_all[idx:idx+1]
            Y_per_cycle[i] = np.mean(Y_all[idx:idx+add], axis=0)
            
            temp = X_per_cycle[i].reshape(1, num_input)
            
            X_all[idx:idx+add] = np.repeat(temp, add, axis=0)
            idx += add
        
        data = []
        data.append(X_all)
        data.append(Y_all)
        data.append(X_per_cycle)
        data.append(Y_per_cycle)

        data = np.array(data)
        np.save(file_path, data)
        print(file_path, "npy file saved!!")
        
    print("============ Data load =============")
    print("X data shape: ", X_all.shape, "X per cycle data shape:", X_per_cycle.shape)
    print("Y data shape: ", Y_all.shape, "Y per cycle data shape:", Y_per_cycle.shape)  
    print("any nan in X?: ", np.argwhere(np.isnan(X_all)))
    print("any nan in Y?: ", np.argwhere(np.isnan(Y_all)))
      
    return X_all, Y_all, X_per_cycle, Y_per_cycle

        
def split_data(x, y, num_train, num_val):
    
    print(x.shape)
    print(y.shape)
    
    if len(x) == len(y):
        print("Same number of x data and y data")
        len_total = len(x)
    else:
        print("Different number of x data and y data")
    
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:], y[num_train:]
    
    print("============= Data split ==============")
    print("train X: {} train Y: {}".format(x_train.shape, y_train.shape))
    print("val X: {} val Y: {}".format(x_val.shape, y_val.shape))
    
    return x_train, y_train, x_val, y_val

class Dataset():   
    def __init__(self, name):
        
        self.train_X = None
        self.val_X = None
        
        self.train_Y = None
        self.val_Y = None
        
        self.train_X_per_cycle = None
        self.val_X_per_cycle = None
              
        self.train_Y_per_cycle = None
        self.val_Y_per_cycle = None
        
        self.train_Y_mean= None
        self.val_Y_mean = None
        
        self.train_Y_noise = None
        self.val_Y_noise = None    
    
class SEMI_gan_data(Dataset):
    def __init__(self, name, num_input, num_output, num_in_cycle, num_of_cycle, num_train, num_val, x_cols, y_cols, header):
        super().__init__(name)
        
        # STEP 1: Data load
        X_all, Y_all, X_per_cycle, Y_per_cycle = load_xlsx_data(name, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header)
        
        # STEP2: split data ( K - fold if neccesary )
        self.train_X, self.train_Y, self.val_X, self.val_Y = split_data(X_all, Y_all, num_train*num_in_cycle, num_val*num_in_cycle)
        self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle = split_data(X_per_cycle, Y_per_cycle, num_train, num_val)                 
                
        # OPTIONAL: Split data for Y_mean, Y_noise
        
        self.train_Y_mean = np.repeat(self.train_Y_per_cycle, num_in_cycle, axis=0)
        self.val_Y_mean = np.repeat(self.val_Y_per_cycle, num_in_cycle, axis=0)
        
        print("train_Y_mean shape", self.train_Y_mean.shape)
        print("val_Y_mean shape", self.val_Y_mean.shape)
        
        self.train_Y_noise = self.train_Y - self.train_Y_mean
        self.val_Y_noise = self.val_Y - self.val_Y_mean
        
        print("train_Y_noise shape", self.train_Y_noise.shape)
        print("val_Y_noise shape", self.val_Y_noise.shape)

class SEMI_sample_data(Dataset):
    def __init__(self, name, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header):
        super().__init__(name)
        
        X_all, Y_all, X_per_cycle, Y_per_cycle = load_sample_data(name, num_input, num_output, num_in_cycle, num_of_cycle, x_cols, y_cols, header)
        
        self.test_X = X_all
        self.test_Y = Y_all
        self.test_X_per_cycle = X_per_cycle
        self.test_Y_per_cycle = X_per_cycle