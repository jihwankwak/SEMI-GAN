import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, hidden_dim, num_of_input, num_of_output):
        super(Net, self).__init__()
        print(num_of_input, hidden_dim, num_of_output)
        self.fc1 = nn.Linear(num_of_input, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_of_output)
        
    def forward(self, x):
        x=torch.tanh(self.fc1(x))
        x=torch.tanh(self.fc2(x))
        x=torch.tanh(self.fc3(x))
        x=self.fc4(x)
        #x=F.sigmoid(x)

        return x    