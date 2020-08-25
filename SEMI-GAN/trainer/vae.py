from __future__ import print_function

import networks
import trainer
import utils

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

class VAETrainer(trainer.vae_GenericTrainer):
    def __init__(self, noise_trainer_iterator, noise_val_iterator, model, optimizer, exp_vae_lr_scheduler, noise_d):
        super().__init__(noise_trainer_iterator, noise_val_iterator, model, optimizer, exp_vae_lr_scheduler, noise_d)
                
    def train(self):
        
        loss_sum = 0.
        batch_num = 0
        
        self.model.train()
        
        for i, data in enumerate(self.train_iterator):
            
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            mini_batch_size = len(data_x)
            
            # Train VAE
            recon_y, mu, log_var = self.model(data_y, data_x)
            loss = self.loss_fn(recon_y, data_y, mu, log_var)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_sum += loss / mini_batch_size
            batch_num += 1
            
        self.prob['loss_train'].append(loss_sum / batch_num)
            
        for param_group in self.optimizer.param_groups:
            self.current_d_lr = param_group['lr']
        self.exp_lr_scheduler.step()
        
        return loss_sum / batch_num
                    
    def evaluate(self):
        
        loss = 0.
        batch_num = 0
        
        self.model.eval()
        
        for i, data in enumerate(self.val_iterator):
            
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            
            mini_batch_size = len(data_x)
            
            z = utils.sample_z(mini_batch_size, self.noise_d)
            
            with torch.autograd.no_grad():
                recon_y, mu, log_var = self.model(data_y, data_x)
                loss += self.loss_fn(recon_y, data_y, mu, log_var) / mini_batch_size
                gen_y = self.model.decoder(z, data_x)
                
            batch_num += 1
            
        loss /= batch_num
        
        self.prob['loss_val'].append(loss)
        
        return loss
    
    def loss_fn(self, recon_y, y, mu, log_var):
        BCE = F.mse_loss(recon_y, y, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD