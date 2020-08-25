import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, noise_d, hidden_dim, num_of_output, num_of_input):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(num_of_output + num_of_input, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, noise_d)
        self.fc32 = nn.Linear(hidden_dim, noise_d)
        # decoder part
        self.fc4 = nn.Linear(noise_d + num_of_input, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, num_of_output)
        
    def encoder(self, y, x):
        z = torch.cat((y,x), dim=1)
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z, x):
        z = torch.cat((z, x), dim=1)
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return self.fc6(h)
    
    def forward(self, y, x):
        mu, log_var = self.encoder(y, x)
        z = self.sampling(mu, log_var)
        return self.decoder(z, x), mu, log_var
    
    def forward(self, y, x):
        mu, log_var = self.encoder(y, x)
        z = self.sampling(mu, log_var)
        return self.decoder(z, x), mu, log_var        