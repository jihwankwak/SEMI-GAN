from __future__ import print_function

import networks
import trainer
import utils

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

class GanTrainer(trainer.gan_GenericTrainer):
    def __init__(self, noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, noise_d, clipping, kernel_sigma, kappa, threshold_type): #nonzero_soft_weight_threshold=1e-3
        super().__init__(noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, noise_d, clipping, kernel_sigma, kappa, threshold_type) #nonzero_soft_weight_threshold=1e-3
        self.clipping = None
        
    def train(self):
        print('1.new_epoch')
        nonzero_soft_weight_threshold = torch.tensor(1e-3)
        
        p_real_list = []
        p_fake_list = []
        
        self.G.train()
        self.D.train()
        
        train_labels = torch.from_numpy(self.train_iterator.dataset.data_x).type(torch.float).cuda() #[:,:-3] ##LER+onehot
        train_labels_sub = train_labels[:,:-3]
        train_labels_dummpy = train_labels[:,-3:]
        train_samples = torch.from_numpy(self.train_iterator.dataset.data_y).type(torch.float).cuda() ##random variation
    
        
        num_of_output = train_labels_sub.shape[1]
        max_x = torch.max(train_labels_sub, dim=0)[0]
        min_x = torch.min(train_labels_sub, dim=0)[0]
        
        min_max_normalized_train_labels_sub = (train_labels_sub - min_x)/(max_x - min_x)
        min_max_normalized_train_labels = torch.cat([min_max_normalized_train_labels_sub,train_labels_dummpy], dim=1)
        
        
        for i, data in enumerate(self.train_iterator):
            data_x, data_y = data #unnormalized
            data_x, data_y = data_x.cuda(), data_y.cuda() #unnormalized? Gaussiannormalized?
            batch_labels_dummpy = data_x[:,-3:]#.cuda()
            
            mini_batch_size = len(data_x)
            
            ############### ccgan data gathering
            batch_epsilons = torch.from_numpy(np.random.normal(0, self.kernel_sigma, mini_batch_size)).type(torch.float).cuda() ##iteration 마다 랜덤한 margin 선택
            batch_target_labels_sub = (data_x[:,:-3] - min_x)/(max_x - min_x) + batch_epsilons.view(-1,1) ## (normalize 해야함?)
            batch_real_indx = torch.zeros(mini_batch_size, dtype=int)
            batch_fake_labels = torch.zeros_like(batch_target_labels_sub)
            
            for j in range(mini_batch_size):
                #print('3.new_example')
                if self.threshold_type == "hard":
                    indx_real_in_vicinity = torch.where(torch.sum(torch.abs(min_max_normalized_train_labels_sub-batch_target_labels_sub[j]), dim=1) <= self.kappa)[0] ## hard margin
                else:
                    raise "not implemented"
                    # reverse the weight function for SVDL
                    #indx_real_in_vicinity = torch.where(torch.sum((min_max_normalized_train_labels_sub-batch_target_labels_sub[j])**2, dim=1)<=-torch.log(nonzero_soft_weight_threshold)/self.kappa)[0]
            
                ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
                while len(indx_real_in_vicinity)<1: ## 한 샘플도 없다면.. 임시로 margin을 키워서 데이터 모으자
                    #print('4.new_while')
                    batch_epsilons_j = torch.from_numpy(np.random.normal(0, self.kernel_sigma, 1)).type(torch.float).cuda()
                    batch_target_labels_sub[j] = batch_target_labels_sub[j] + batch_epsilons_j.view(-1,1)
                    
                    batch_target_labels_sub = torch.clamp(batch_target_labels_sub, 0.0, 1.0)
                    ## index for real images
                    if self.threshold_type == "hard":
                        indx_real_in_vicinity = torch.where(torch.sum(torch.abs(min_max_normalized_train_labels_sub-batch_target_labels_sub[j]), dim=1) <= self.kappa)[0]
                    else:
                        raise "not implemented"
                        # reverse the weight function for SVDL
                        #indx_real_in_vicinity = np.where(np.sum((min_max_normalized_train_labels_sub-batch_target_labels_sub[j])**2, axis=1) <= -np.log(nonzero_soft_weight_threshold)/self.kappa)[0]
                #end while len(indx_real_in_vicinity)<1

                assert len(indx_real_in_vicinity)>=1 # 만족을 하면 에러 안뜸
            
                batch_real_indx[j] = torch.from_numpy(np.random.choice(indx_real_in_vicinity.cpu(), size=1)).type(torch.float)#.cuda() #모은 셋에서 하나 뽑아서 X 바꾸기 -> iteration 별로 다르게 사용되니까.. 합리적, target은 ? iteration별로 uniform random하게 뽑음 -> imbalance 고려 안해도 됨
                
                ## labels for fake images generation
                if self.threshold_type == "hard":
                    lb = batch_target_labels_sub[j] - self.kappa * num_of_output
                    ub = batch_target_labels_sub[j] + self.kappa * num_of_output
                else:
                    raise "not implemented"
                    #lb = batch_target_labels_sub[j] - np.sqrt(-np.log(nonzero_soft_weight_threshold)/self.kappa)
                    #ub = batch_target_labels_sub[j] + np.sqrt(-np.log(nonzero_soft_weight_threshold)/self.kappa)
                
                for k in range(num_of_output):
                    lb[k] = max(0.0, lb[k])
                    ub[k] = min(ub[k], 1.0)
                    assert lb[k]<=ub[k]
                    assert lb[k]>=0 and ub[k]>=0
                    assert lb[k]<=1 and ub[k]<=1
                    batch_fake_labels[j,k] = torch.from_numpy(np.random.uniform(lb[k].cpu(), ub[k].cpu(), size=1)).type(torch.float)#.cuda()
            #end for j
            
            ## draw the real image batch from the training set
            batch_real_samples = train_samples[batch_real_indx] ## index로 부터 X 가져오기
            batch_real_labels = train_labels[batch_real_indx] ## index에 대응되는 true label y 가져오기
            batch_real_samples = batch_real_samples.cuda()
            batch_real_labels = batch_real_labels.cuda()

            ## generate the fake image batch
            batch_fake_labels = batch_fake_labels*(max_x - min_x) + min_x #revert the normalization
            batch_fake_labels_sub = batch_fake_labels
            
            
            batch_fake_labels = torch.cat((batch_fake_labels_sub, batch_labels_dummpy), dim=1).cuda()
            z = torch.randn(mini_batch_size, self.noise_d, dtype=torch.float).cuda() ## noise
            data_x, data_y = data_x.cuda(), data_y.cuda() #unnormalized? Gaussiannormalized?
            batch_fake_samples = self.G(z, data_x)

            ## target labels on gpu
            batch_target_labels_sub = batch_target_labels_sub*(max_x - min_x) + min_x
            batch_target_labels = torch.cat((batch_target_labels_sub, batch_labels_dummpy), dim=1).type(torch.float).cuda() ## 요건 margin 고려헀을때 label (X에 대응되는 y_hat)

            ## weight vector
            if self.threshold_type == "soft":
                raise "not implemented"
                #real_weights = torch.exp(-self.kappa*(batch_real_labels_sub-batch_target_labels_sub)**2).to(device)
                #fake_weights = torch.exp(-self.kappa*(batch_fake_labels_sub-batch_target_labels_sub)**2).to(device)
            else:
                real_weights = torch.ones(mini_batch_size, dtype=torch.float).cuda()
                fake_weights = torch.ones(mini_batch_size, dtype=torch.float).cuda()
            #end if threshold type
    
            # forward pass
            p_real_D = self.D(batch_real_samples, batch_target_labels) ## feature
            p_fake_D = self.D(batch_fake_samples.detach(), batch_target_labels) ## feature

            d_loss = - torch.mean(real_weights.view(-1) * torch.log(p_real_D.view(-1)+1e-20)) - torch.mean(fake_weights.view(-1) * torch.log(1 - p_fake_D.view(-1)+1e-20))

            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()
            
            
            ############### GENERATOR
            
            batch_epsilons = torch.from_numpy(np.random.normal(0, self.kernel_sigma, mini_batch_size)).type(torch.float).cuda() ##iteration 마다 랜덤한 margin 선택
            batch_target_labels_sub = (data_x[:,:-3] - min_x)/(max_x - min_x) + batch_epsilons.view(-1,1) ## (normalize 해야함?)
            batch_target_labels_sub = torch.clamp(batch_target_labels_sub, 0.0, 1.0)
            batch_target_labels_sub = batch_target_labels_sub
            
            batch_target_labels = torch.cat((batch_target_labels_sub, batch_labels_dummpy), dim=1).cuda()
            
            z = utils.sample_z(mini_batch_size, self.noise_d).cuda()
            batch_fake_samples = self.G(z, batch_target_labels)
            
            
            # loss
            p_fake = self.D(batch_fake_samples, batch_target_labels)
            g_loss = - torch.mean(torch.log(p_fake+1e-20))

            
            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()
            
            
        self.prob['p_real_train'].append(p_real_D)
        self.prob['p_fake_train'].append(p_fake_D)
            
        for param_group in self.optimizer_D.param_groups:
            self.current_d_lr = param_group['lr']
        self.exp_gan_lr_scheduler.step()
        
        return p_real_D, p_fake_D
                    
    def evaluate(self):
        
        p_real, p_fake = 0., 0.
        batch_num = 0
        
        self.G.eval()
        self.D.eval()
        
        for i, data in enumerate(self.val_iterator):
            
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            
            mini_batch_size = len(data_x)
            
            z = utils.sample_z(mini_batch_size, self.noise_d)
            
            with torch.autograd.no_grad():
                p_real += torch.sum(self.D(data_y, data_x)/mini_batch_size)
                
                gen_y = self.G(z, data_x)
                
                p_fake += torch.sum(self.D(gen_y, data_x)/mini_batch_size)
                
            batch_num += 1
            
        p_real /= batch_num
        p_fake /= batch_num
        
        self.prob['p_real_val'].append(p_real)
        self.prob['p_fake_val'].append(p_fake)
        
        return p_real, p_fake