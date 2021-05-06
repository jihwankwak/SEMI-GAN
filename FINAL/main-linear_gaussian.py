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

log_name = 'linear_gaussian_date_{}_data_{}_seed_{}_lr_{}_batch_size_{}_sample_num_{}_tr_num_in_cycle_{}'.format(
    args.date,
    args.dataset,
    args.seed,
    args.lr,    
    args.batch_size, 
    args.sample_num, args.tr_num_in_cycle
)

utils.set_seed(args.seed)

model_spec = 'linear_gaussian_date_{}_data_{}_seed_{}_lr_{}_batch_size_{}_sample_num_{}_tr_num_in_cycle_{}'.format(
    args.date,
    args.dataset,
    args.seed,
    args.lr,    
    args.batch_size, 
    args.sample_num, args.tr_num_in_cycle
)

if args.pdrop is not None:
    model_spec += '_pdrop_{}'.format(args.pdrop)
    log_name += '_pdrop_{}'.format(args.pdrop)

print(log_name)

print("="*100)
print("Arguments =")
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print("="*100)

# Dataset
dataset = data_handler.DatasetFactory.get_dataset(args) #
 
# Test specific dataset
dataset_test = data_handler.DatasetFactory.get_test_dataset(args) #

# loss result
result_dict = {}

kwargs = {'num_workers': args.workers}

print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
print("Inits...")
torch.set_default_tensor_type('torch.cuda.FloatTensor')



### 상수설정
X_train_mean, X_train_std, Y_train_mean, Y_train_std = utils.train_mean_std(args, dataset.train_X_per_cycle, dataset.train_Y_mean_cov) # 6(mean)+21(cov) ####dataset 만들기

print('Y_train_mean',Y_train_mean)
print('Y_train_std',Y_train_std)
print(" Assign mean, std for Training data ")
print("X train mean, std", X_train_mean, X_train_std) #

train_dataset_loader = data_handler.SemiLoader(args, dataset.train_X_per_cycle, 
                                                     dataset.train_Y_mean_cov, 
                                                     X_train_mean, X_train_std, Y_train_mean, Y_train_std) #

val_dataset_loader = data_handler.SemiLoader(args, dataset.val_X_per_cycle, 
                                                   dataset.val_Y_mean_cov, 
                                                   X_train_mean, X_train_std, Y_train_mean, Y_train_std) #

# Dataloader
train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True, **kwargs) #

val_iterator = torch.utils.data.DataLoader(val_dataset_loader, batch_size=1, shuffle=False, **kwargs) #


# model
model = networks.ModelFactory.get_gaussian_model(args) # linear gaussian model 만들기


# weight initiailzation
model.apply(utils.init_params)


model.cuda()
print(model)


# scheduler
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)


exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


# trainer
linear_gaussian_trainer = trainer.TrainerFactory.get_trainer(train_iterator, val_iterator, model, args, optimizer, exp_lr_scheduler) #


if args.mode == 'train' and not os.path.isfile('./models/linear_gaussian/'+model_spec):
    for epoch in range(args.gaussian_nepochs):

        linear_gaussian_trainer.train()
        val_loss, _ = linear_gaussian_trainer.evaluate()
        current_lr = linear_gaussian_trainer.current_lr

        if((epoch+1)% 10 == 0):
            print("epoch:{:2d}, lr_d:{:.6f}, || val_loss:{:.6f}".format(epoch, current_lr, val_loss))
            
    # net.state_dict()
    torch.save(model.state_dict(), './models/linear_gaussian/'+model_spec)
    
else:
    print()
    print('Load mean model----------------')
    print()
    linear_gaussian_trainer.model.load_state_dict(torch.load('./models/linear_gaussian/'+model_spec))


# ==================================================================================================
#                                          3. Generate sample
# ==================================================================================================

if args.model_type == 'linear_gaussian':
    testType = 'gaussian'

t_classifier = trainer.EvaluatorFactory.get_evaluator(args.sample_num, args.num_of_output, testType) #

# mean result
print("mean_train mean, std for scaling: ", Y_train_mean, Y_train_std)    



if args.mode == 'eval':
  #< for past dataset that did not have seperate test datset >

#     test_dataset_loader = data_handler.SemiLoader(args, dataset.test_X_per_cycle, 
#                                                        dataset.test_Y_per_cycle, 
#                                                        X_train_mean, X_train_std, Y_train_mean, Y_train_std)    

    test_X_dataset_loader = data_handler.SemiLoader(args, dataset_test.test_X_per_cycle, 
                                                       dataset_test.test_Y_per_cycle, 
                                                       X_train_mean, X_train_std, 0, 1)
    
    
    test_X_iterator = torch.utils.data.DataLoader(test_X_dataset_loader, batch_size=1, shuffle=False)
    
# if args.mode == 'train':
#     mean_result, mean_total = t_classifier.mean_sample(mean_best_model, Y_train_mean, Y_train_std, mean_train_iterator)
# else:
#     mean_result, mean_total = t_classifier.mean_sample(mean_best_model, Y_train_mean, Y_train_std, test_mean_iterator)
    
    
if args.mode == 'train':
    mean_cov_result, total = t_classifier.mean_cov_sample(model, val_iterator)
    
    #num_of_cycle = len(dataset_test.train_X_per_cycle)
    total_result = t_classifier.sample(mean_cov_result, Y_train_mean, Y_train_std, total)
    
    
else:
    mean_cov_result, total = t_classifier.mean_cov_sample(model, test_X_iterator)
    #num_of_cycle = len(dataset_test.test_X_per_cycle)
    total_result = t_classifier.sample(mean_cov_result, Y_train_mean, Y_train_std, total)


# # 3: num_of_input
# # 6: num_of_output

# mean_result = np.repeat(mean_result, args.sample_num, axis=0)
# print('mean_result',mean_result)
# total_result = noise_result + mean_result
    
if args.mode == 'train':
    np.save('./sample_data/'+log_name, total_result)

else:
#     np.save('./sample_data/'+'old_test_specific'+log_name, total_result)
    np.save('./sample_data/'+'test_specific'+log_name, total_result)