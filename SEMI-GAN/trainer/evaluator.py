import torch
import numpy as np
import utils

class EvaluatorFactory():
    def __init__(self):
        pass
    
    @staticmethod
    def get_evaluator(sample_num, testType='gan'):
        if testType == 'gan':
            return gan_evaluator(sample_num)
        elif testType == 'vae':
            return vae_evaluator(sample_num)
        elif testType == 'gaussian':
            return gaussian_evaluator(sample_num)
        
class gan_evaluator():
    
    def __init__(self, sample_num):
        
        self.sample_num = sample_num
        self.prob = {}
        
    def train_eval(self, generator, discriminator, val_loader, noise_d):
        
        p_real, p_fake = 0., 0.
        batch_num = 0
        
        generator.eval()
        discriminator.eval()
        
        for i, data in enumerate(val_loader):
            print(i)
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            
            mini_batch_size = len(data_x)
            
            z = utils.sample_z(mini_batch_size, noise_d)
            # utils.sample_z
            
            with torch.autograd.no_grad():
                p_real += torch.sum(discriminator(data_y, data_x)/mini_batch_size)
                
                gen_y = generator(z, data_x)
                
                p_fake += torch.sum(discriminator(gen_y, data_x)/mini_batch_size)
                
            batch_num += 1
            
        p_real /= batch_num
        p_fake /= batch_num
        
        self.prob['p_real_val'].append(p_real)
        self.prob['p_fake_val'].append(p_fake)
        
        return p_real, p_fake
        
    def mean_sample(self, mean_model, train_mean, train_std, test_iterator):
        
        mean_result = []
        total = 0

        mean_model.eval()
        
        for i, data in enumerate(test_iterator):
            print(i)
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            print(data_x)
            
            mini_batch_size = len(data_x)
            print(mini_batch_size)
            
            with torch.autograd.no_grad():
                mean = mean_model(data_x)
                
                mean = mean.data.cpu().numpy()
                
                
                mean = mean*train_std + train_mean
                mean_result.append(mean[0].tolist())
                
                total += mini_batch_size
        
        mean_result = np.array(mean_result)
            
        print("mean_result size: ", mean_result.shape)

        return mean_result, total
              
    def noise_sample(self, mean_result, generator, train_mean, train_std, test_loader, num_of_input, num_of_output, noise_d):
        
        noise_result = []
        total = 0
        
        generator.eval()
        
        for i, data in enumerate(test_loader):
            
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            
            mini_batch_size = len(data_x)
            
            with torch.autograd.no_grad():
                
                data_x_sample = data_x.repeat(1, self.sample_num).view(-1, num_of_input)
                              
                # print("batch * sample_num: (expected)", mini_batch_size*self.sample_num, "(result)", data_x_sample.shape)
                
                z = utils.sample_z(mini_batch_size*self.sample_num, noise_d)
                noise = generator(z, data_x_sample)
                
                noise = noise.data.cpu().numpy()
                
                noise = noise*train_std + train_mean
                print(noise.shape)
                
                noise_result.append(noise.tolist())
                
                total += mini_batch_size
            
        
        noise_result = np.array(noise_result)
        noise_result = noise_result.reshape(-1, num_of_output)
        
        print("noise_result_size: ", noise_result.shape)
        
        return noise_result, total