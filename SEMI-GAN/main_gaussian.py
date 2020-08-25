from arguments import get_args

import torch
import data_handler
import networks
import trainer
import utils
import numpy as np

import os, time
import scipy.io as sio
from torch.optim import lr_scheduler


# Arguments
args = get_args()

log_name = 'date_{}_data_{}_seed_{}_batch_size_{}_sample_num_{}'.format(
    args.date,
    args.data_type,    
    args.seed,   
    args.batch_size,
    args.sample_num
    )
    
if 'gan' in args.trainer:
    log_name += '_model_{}_{}'.format(args.mean_model_type, args.gan_model_type)
    log_name += '_lr_{}_{}_{}'.format(args.mean_lr, args.g_lr, args.d_lr)
    log_name += '_hidden_dim_{}_{}'.format(args.mean_hidden_dim, args.gan_hidden_dim)
    log_name += '_epoch_{}_{}'.format(args.mean_nepochs, args.gan_nepochs)
    log_name += '_noise_d_{}'.format(arg.noise_d)
    
if 'gaussian' in args.trainer:
    log_name += '_model_{}'.format(args.mean_model_type)
    log_name += '_lr_{}'.format(args.mean_lr)
    log_name += '_hidden_dim_{}'.format(args.mean_hidden_dim)
    log_name += '_epoch_{}'.format(args.mean_nepochs)                                     
                               
print(log_name)

print("="*100)
print("Arguments =")
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print("="*100)

# Dataset
dataset = data_handler.DatasetFactory.get_dataset(args)

# loss result
result_dict = {}

kwargs = {'num_workers': args.workers}

print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
print("Inits...")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# ==================================================================================================
#                                         1. model train
# ==================================================================================================

train_loader = data_handler.SemiLoader(dataset.train_X_per_cycle, 
                                                    dataset.train_Y_per_cycle, 
                                                    utils.normalize)

val_loader = data_handler.SemiLoader(dataset.val_X_per_cycle, 
                                                  dataset.val_Y_per_cycle,
                                                  utils.normalize)

test_loader = data_handler.SemiLoader(dataset.test_X_per_cycle, 
                                                   dataset.test_Y_per_cycle, 
                                                   utils.normalize)

# Dataloader

train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, **kwargs)

val_iterator = torch.utils.data.DataLoader(val_loader, batch_size=args.batch_size, shuffle=False, **kwargs)

test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=1, shuffle=False, **kwargs)

# model

model = networks.ModelFactory.get_mean_model(args)

# weight initialization

model.apply(utils.init_normal)
model.cuda()

print(model)

# optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=args.mean_lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) # scheduler

# trainer

mytrainer = trainer.TrainerFactory.get_trainer(train_iterator, val_iterator, model, args, optimizer, exp_lr_scheduler)

for epoch in range(args.mean_nepochs):
    
    train_loss = mytrainer.train()
    
    val_loss, val_r2 = mytrainer.evaluate()
    
    current_lr = mytrainer.current_lr
    
    if((epoch+1)% 10 == 0):
        print("epoch:{:2d}, lr:{:.6f}, || train_loss:{:.6f}, val_loss:{:.6f}, r2_score:{:.6f}".format(epoch, current_lr, train_loss, val_loss, val_r2))

result_dict['train_loss'] = mytrainer.loss['train_loss']
result_dict['val_loss'] = mytrainer.loss['val_loss']
        
best_model = mytrainer.best_model

if not os.path.exists('./gaussian/result_loss'):
    os.makedirs('./gaussian/result_loss')
np.save('./gaussian/result_loss/'+log_name, result_dict)

# net.state_dict()
if not os.path.exists('./gaussian/model'):
    os.makedirs('./gaussian/model')
torch.save(best_model.state_dict(), './gaussian/model'+log_name)

# ==================================================================================================
#                                          2. Generate sample
# ==================================================================================================

if args.trainer == 'gaussian':
    testType = 'gaussian'

t_classifier = trainer.EvaluatorFactory.get_evaluator(args.sample_num, testType)

# mean_cov result

train_mean = train_loader.data_y_mean
train_std = train_loader.data_y_std

print(train_mean, train_std)

mean_cov_result, total = t_classifier.mean_cov_sample(best_model, train_mean, train_std, test_iterator)

# sample

result = t_classifier.sample(mean_cov_result, args.num_of_output, args.num_of_cycle) 

if not os.path.exists('./sample_data'):
    os.makedirs('./sample_data')
np.save('./sample_data/'+'total_'+log_name, result)