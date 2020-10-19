import torch

import pandas as pd
import numpy as np

import os

def load_data(name):
    
    data = np.load('./data_handler/'+name+'.npy', allow_pickle=True)
    
    X_all, Y_all, X_per_cycle, Y_per_cycle = data[0], data[1], data[2], data[3]
    
    print("============ Data load =============")
    print("X data shape: ", X_all.shape, "X per cycle data shape:", X_per_cycle.shape)
    print("Y data shape: ", Y_all.shape, "Y per cycle data shape:", Y_per_cycle.shape)  
    print("any nan in X?: ", np.argwhere(np.isnan(X_all)))
    print("any nan in Y?: ", np.argwhere(np.isnan(Y_all)))

    return X_all, Y_all, X_per_cycle, Y_per_cycle
        
def split_data(x, y, num_train, num_val):
    
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
    def __init__(self, name, num_in_cycle, num_of_cycle, num_train, num_val):
        super().__init__(name)
        
        # STEP 1: load data
        
        X_all, Y_all, X_per_cycle, Y_per_cycle = load_data(name)
        
        # STEP 2: Split data
       
        self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = split_data(X_all, Y_all, num_train*num_in_cycle, num_val*num_in_cycle)
        self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_per_cycle, num_train, num_val)
 
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

class SEMI_sample_data(Dataset):
    def __init__(self, name):
        super().__init__(name)
        
        data = np.load('./data_handler/'+name+'.npy', allow_pickle=True)
    
        X_all, Y_all, X_per_cycle, Y_per_cycle = data[0], data[1], data[2], data[3]
        
        print("============ Data load =============")
        print("test X data shape: ", X_all.shape, "test X per cycle data shape:", X_per_cycle.shape)
        print("test Y data shape: ", Y_all.shape, "test Y per cycle data shape:", Y_per_cycle.shape)  
        print("any nan in test X?: ", np.argwhere(np.isnan(X_all)))
        print("any nan in test Y?: ", np.argwhere(np.isnan(Y_all)))
                
        self.test_X = X_all
        self.test_Y = Y_all
        self.test_X_per_cycle = X_per_cycle
        self.test_Y_per_cycle = Y_per_cycle