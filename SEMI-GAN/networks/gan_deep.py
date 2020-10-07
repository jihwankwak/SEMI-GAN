import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        
        self.fc1 = nn.Linear(inplanes, planes)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(planes, planes)

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.leaky_relu(out)

        out = self.fc2(out)

        out += identity
        out = self.leaky_relu(out)

        return out

class gen1(nn.Module):
    def __init__(self, d_noise_num_of_input, gan_hidden_dim, num_of_output):
        super(gen1, self).__init__()
        self.inplanes = gan_hidden_dim
        self.fc1 = nn.Linear(d_noise_num_of_input, gan_hidden_dim)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, gan_hidden_dim, 4)
        self.fc2 = nn.Linear(gan_hidden_dim, num_of_output)
        
    def _make_layer(self, block, planes, blocks):
        
        layers = []
        layers.append(block(self.inplanes, planes))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, noise, x):
        gen_input = torch.cat((noise, x), axis=1)
        x = self.leaky_relu(self.fc1(gen_input))
        x = self.layer1(x)
        x = self.fc2(x)
        return x
        
class dis1(nn.Module):
    def __init__(self, num_of_output, gan_hidden_dim):
        super(dis1, self).__init__()
        
        self.inplanes = gan_hidden_dim
        self.fc1 = nn.Linear(num_of_output, gan_hidden_dim)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, gan_hidden_dim, 4)
        self.fc2 = nn.Linear(gan_hidden_dim, 1)
        
    def _make_layer(self, block, planes, blocks):
        
        layers = []
        layers.append(block(self.inplanes, planes))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
        
    def forward(self, y, x):
        dis_input = torch.cat((y, x), axis=1)
        
        x = self.leaky_relu(self.fc1(dis_input))
        x = self.layer1(x)
        
        x = torch.sigmoid(self.fc2(x))
        
        return x