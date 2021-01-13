import torch
import torch.utils.data as td
import numpy as np


class SemiLoader(td.Dataset):
    def __init__(self, args, data_x, data_y, x_mean, x_std, y_mean, y_std):
                   
        self.data_x = data_x
        self.data_y = data_y
        
        # normalization
        
        temp_x = (data_x[:,:args.num_of_input-2] - x_mean[:, :args.num_of_input-2]) / x_std[:, :args.num_of_input-2]
        temp_y = (data_y - y_mean) / y_std
        
        temp_x = np.hstack((temp_x, np.ones((temp_x.shape[0], 1))))
        temp_x = np.hstack((temp_x, np.zeros((temp_x.shape[0], 1))))
        
        self.data_x = temp_x
        self.data_y = temp_y
        
    def __len__(self):
        return len(self.data_x)
           
    def __getitem__(self, index):
        
        x = self.data_x[index]
        y = self.data_y[index]
            
        x = torch.from_numpy(x).float().cuda()
        y = torch.from_numpy(y).float().cuda()
        
        # print(y)
                
        # x[3] = 1.0
        # x[4] = 0.0
        
        return x, y
        