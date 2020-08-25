import torch
import torch.utils.data as td

class SemiLoader(td.Dataset):
    def __init__(self, data_x, data_y, normalize = None):
                   
        self.data_x = data_x
        self.data_y = data_y
        
        if normalize is not None:
            temp_data_x, self.data_y, self.data_y_mean, self.data_y_std = normalize(data_x[:,:3], data_y)    
            self.data_x[:,:3] = temp_data_x
            
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
        