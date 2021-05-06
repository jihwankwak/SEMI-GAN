from arguments_mean import get_args

import torch
import data_handler
import networks
import trainer
import utils
import numpy as np

import os, time
import scipy.io as sio
from torch.optim import lr_scheduler
import pickle


# Arguments
args = get_args()

log_name = 'date_{}_data_{}_model_{}_seed_{}_{}_lr_{}_hidden_dim_{}_batch_size_{}'.format(
    args.date,
    args.dataset,
    args.mean_model_type,
    args.seed,
    args.relu_type,
    args.mean_lr,  
    args.mean_hidden_dim,
    args.batch_size,  
)

utils.set_seed(args.seed)

# Mean model architecture ( naming for training & sampling )
mean_model_spec = 'data_{}_batch_{}_model_{}_lr_{}_hidden_dim_{}_relu_{}'.format(args.dataset, args.batch_size, args.mean_model_type, args.mean_lr, args.mean_hidden_dim, args.relu_type)

if args.pdrop is not None:
    mean_model_spec += '_pdrop_{}'.format(args.pdrop)
    log_name += '_pdrop_{}'.format(args.pdrop)

print(log_name)

print("="*100)
print("Arguments =")
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print("="*100)

# Dataset
dataset = data_handler.DatasetFactory.get_dataset(args)

# Test specific dataset
dataset_test = data_handler.DatasetFactory.get_test_dataset(args)

# loss result
result_dict = {}

kwargs = {'num_workers': args.workers}

print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
print("Inits...")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# ==================================================================================================
#                                          1. Predict mean 
# ==================================================================================================

### 상수설정
X_train_mean, X_train_std, Y_train_mean, Y_train_std = utils.train_mean_std(dataset.train_X, dataset.train_Y)
# X_train_min, X_train_max, Y_train_min, Y_train_max = utils.train_min_max(dataset.train_X, dataset.train_Y)

print(" Assign mean, std for Training data ")
print("X train mean, std", X_train_mean, X_train_std)
print("Y train mean, std", Y_train_mean, Y_train_std)

# print("X train min, max", X_train_min, X_train_max)
# print("Y train min, max", Y_train_min, Y_train_max)


mean_train_dataset_loader = data_handler.SemiLoader(dataset.train_X_per_cycle, 
                                                    dataset.train_Y_per_cycle, 
                                                    X_train_mean, X_train_std, Y_train_mean, Y_train_std)

mean_val_dataset_loader = data_handler.SemiLoader(dataset.val_X_per_cycle, 
                                                  dataset.val_Y_per_cycle,
                                                  X_train_mean, X_train_std, Y_train_mean, Y_train_std)

# mean_train_dataset_loader = data_handler.SemiLoader(dataset.train_X_per_cycle, 
#                                                     dataset.train_Y_per_cycle, 
#                                                     X_train_min, X_train_max, Y_train_min, Y_train_max)

# mean_val_dataset_loader = data_handler.SemiLoader(dataset.val_X_per_cycle, 
#                                                   dataset.val_Y_per_cycle,
#                                                   X_train_min, X_train_max, Y_train_min, Y_train_max)



# Dataloader

mean_train_iterator = torch.utils.data.DataLoader(mean_train_dataset_loader, batch_size=args.batch_size, shuffle=True, **kwargs)

mean_val_iterator = torch.utils.data.DataLoader(mean_val_dataset_loader, batch_size=1, shuffle=False, **kwargs)


# model

mean_model = networks.ModelFactory.get_mean_model(args)

# weight initialization

mean_model.apply(utils.init_normal)
mean_model.cuda()

print(mean_model)

# optimizer

optimizer = torch.optim.Adam(mean_model.parameters(), lr=args.mean_lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) # scheduler

mean_mytrainer = trainer.TrainerFactory.get_mean_trainer(mean_train_iterator, mean_val_iterator, mean_model, args, optimizer, exp_lr_scheduler)

res = {}
res['current_lr'] = []
res['train_loss'] = []
res['val_loss'] = []
res['val_r2'] = []
# trainer
if args.mode == 'train' and not os.path.isfile('./models/mean/'+mean_model_spec):

    for epoch in range(args.mean_nepochs):

        train_loss = mean_mytrainer.train()
        val_loss, val_r2 = mean_mytrainer.evaluate()
        current_lr = mean_mytrainer.current_lr

        if((epoch+1)% 10 == 0):
            print("epoch:{:2d}, lr:{:.6f}, || train_loss:{:.6f}, val_loss:{:.6f}, r2_score:{:.6f}".format(epoch, current_lr, train_loss, val_loss, val_r2))
            
        res['current_lr'].append(current_lr)
        res['train_loss'].append(train_loss)
        res['val_loss'].append(val_loss)
        res['val_r2'].append(val_r2)

    mean_best_model = mean_mytrainer.best_model
    torch.save(mean_best_model.state_dict(), './models/mean/'+mean_model_spec)
    
else:    
    print()
    print('Load mean model----------------')
    print()
    mean_mytrainer.model.load_state_dict(torch.load('./models/mean/'+mean_model_spec))
    mean_best_model = mean_mytrainer.model


path = log_name + '.pkl'
with open('sample_data/' + path, "wb") as f:
    pickle.dump(res, f)