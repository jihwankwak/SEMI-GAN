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

log_name = 'naive_date_{}_data_{}_model_{}_{}_seed_{}_lr_{}_{}_{}_hidden_dim_{}_{}_batch_size_{}_noise_d_{}_sample_num_{}_tr_num_in_cycle_{}'.format(
    args.date,
    args.dataset,
    args.mean_model_type,
    args.gan_model_type,    
    args.seed,
    args.mean_lr,  
    args.g_lr,      
    args.d_lr,        
    args.mean_hidden_dim,
    args.gan_hidden_dim,
    args.batch_size,  
    args.noise_d,
    args.sample_num, args.tr_num_in_cycle
)

utils.set_seed(args.seed)

# # Mean model architecture ( naming for training & sampling )
# mean_model_spec = 'date_{}_data_{}_batch_{}_model_{}_lr_{}_tr_num_in_cycle_{}'.format(args.date, args.dataset, args.batch_size, args.mean_model_type, args.mean_lr, args.tr_num_in_cycle)

# gan model architecture ( naming for training & sampling )
gan_model_spec = 'naive_date_{}_data_{}_batch_{}_model_{}_noise_d_{}_hidden_dim_{}_lr_g_{}_d_{}_tr_num_in_cycle_{}_seed_{}'.format(args.date, args.dataset, args.batch_size, args.gan_model_type, args.noise_d, args.gan_hidden_dim, args.g_lr, args.d_lr, args.tr_num_in_cycle, args.seed)

if args.pdrop is not None:
    gan_model_spec += '_pdrop_{}'.format(args.pdrop)
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
print(dataset_test)

# loss result
result_dict = {}

kwargs = {'num_workers': args.workers}

print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
print("Inits...")
torch.set_default_tensor_type('torch.cuda.FloatTensor')



# # ==================================================================================================
# #                                          1. Predict mean 
# # ==================================================================================================

# ### 상수설정
# #X_train_mean, X_train_std, Y_train_mean, Y_train_std = utils.train_mean_std(args, dataset.train_X, dataset.train_Y)
# X_train_mean, X_train_std, Y_train_mean, Y_train_std = utils.train_mean_std(args, dataset.train_X_per_cycle, dataset.train_Y_per_cycle)

# print(" Assign mean, std for Training data ")
# print("X train mean, std", X_train_mean, X_train_std)
# print("Y train mean, std", Y_train_mean, Y_train_std)


# mean_train_dataset_loader = data_handler.SemiLoader(args, dataset.train_X_per_cycle, 
#                                                     dataset.train_Y_per_cycle, 
#                                                     X_train_mean, X_train_std, Y_train_mean, Y_train_std)

# mean_val_dataset_loader = data_handler.SemiLoader(args, dataset.val_X_per_cycle, 
#                                                   dataset.val_Y_per_cycle,
#                                                   X_train_mean, X_train_std, Y_train_mean, Y_train_std)

# # Dataloader

# mean_train_iterator = torch.utils.data.DataLoader(mean_train_dataset_loader, batch_size=args.batch_size, shuffle=True, **kwargs)

# mean_val_iterator = torch.utils.data.DataLoader(mean_val_dataset_loader, batch_size=1, shuffle=False, **kwargs)


# # model

# mean_model = networks.ModelFactory.get_mean_model(args)

# # weight initialization

# mean_model.apply(utils.init_normal)
# mean_model.cuda()

# print(mean_model)

# # optimizer

# optimizer = torch.optim.Adam(mean_model.parameters(), lr=args.mean_lr)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) # scheduler

# mean_mytrainer = trainer.TrainerFactory.get_mean_trainer(mean_train_iterator, mean_val_iterator, mean_model, args, optimizer, exp_lr_scheduler)

# # trainer
# if args.mode == 'train' and not os.path.isfile('./models/mean/'+mean_model_spec):

#     for epoch in range(args.mean_nepochs):

#         train_loss = mean_mytrainer.train()
#         val_loss, val_r2 = mean_mytrainer.evaluate()
#         current_lr = mean_mytrainer.current_lr

#         if((epoch+1)% 10 == 0):
#             print("epoch:{:2d}, lr:{:.6f}, || train_loss:{:.6f}, val_loss:{:.6f}, r2_score:{:.6f}".format(epoch, current_lr, train_loss, val_loss, val_r2))

#     mean_best_model = mean_mytrainer.best_model
#     torch.save(mean_best_model.state_dict(), './models/mean/'+mean_model_spec)
    
# else:    
#     print()
#     print('Load mean model----------------')
#     print()
#     mean_mytrainer.model.load_state_dict(torch.load('./models/mean/'+mean_model_spec))
#     mean_best_model = mean_mytrainer.model
    

# ==================================================================================================
#                                          2. Predict Y_hat
# ==================================================================================================

### 상수설정
X_train_mean, X_train_std, Y_train_mean, Y_train_std = utils.train_mean_std(args, dataset.train_X, dataset.train_Y) #

print("@@@@@@@debug@@@@@@@@", X_train_mean, X_train_std, Y_train_mean, Y_train_std)

print(" Assign mean, std for Training data ")
print("X train mean, std", X_train_mean, X_train_std) #
print("Y train mean, std", Y_train_mean, Y_train_std) #

train_dataset_loader = data_handler.SemiLoader(args, dataset.train_X, 
                                                     dataset.train_Y, 
                                                     X_train_mean, X_train_std, Y_train_mean, Y_train_std) #

print("val")

val_dataset_loader = data_handler.SemiLoader(args, dataset.val_X_per_cycle, 
                                                    dataset.val_Y_per_cycle, 
                                                    X_train_mean, X_train_std, Y_train_mean, Y_train_std)


# val_dataset_loader = data_handler.SemiLoader(args, dataset.val_X, 
#                                                    dataset.val_Y, 
#                                                    X_train_mean, X_train_std, Y_train_mean, Y_train_std) #

# Dataloader

train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True, **kwargs) #

val_iterator = torch.utils.data.DataLoader(val_dataset_loader, batch_size=1, shuffle=True, **kwargs)

# val_iterator = torch.utils.data.DataLoader(val_dataset_loader, batch_size=1, shuffle=False, **kwargs) #

# model

generator, discriminator = networks.ModelFactory.get_gan_model(args)

# weight initiailzation

generator.apply(utils.init_params)
discriminator.apply(utils.init_params)

generator.cuda()
discriminator.cuda()

print(generator, discriminator)

# scheduler

optimizer_g = torch.optim.Adam(generator.parameters(), lr = args.g_lr)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = args.d_lr)

exp_gan_lr_scheduler = lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.5)

# trainer

gan_mytrainer = trainer.TrainerFactory.get_gan_trainer(train_iterator, val_iterator, generator, discriminator, args, optimizer_g, optimizer_d, exp_gan_lr_scheduler) #

if args.mode == 'train' and not os.path.isfile('./models/generator/'+gan_model_spec):
    
    t_start = time.time()
    
    for epoch in range(args.gan_nepochs):

        gan_mytrainer.train()
        p_real, p_fake = gan_mytrainer.evaluate()
        current_d_lr = gan_mytrainer.current_d_lr

        if((epoch+1)% 10 == 0):
            print("epoch:{:2d}, lr_d:{:.6f}, || p_real:{:.6f}, p_fake:{:.6f}".format(epoch, current_d_lr, p_real, p_fake))
            
    t_end = time.time()
    # net.state_dict()
    torch.save(generator.state_dict(), './models/generator/'+gan_model_spec)
    torch.save(discriminator.state_dict(), './models/discriminator/'+gan_model_spec)
else:
    print()
    print('Load mean model----------------')
    print()
    gan_mytrainer.G.load_state_dict(torch.load('./models/generator/'+gan_model_spec))


# ==================================================================================================
#                                          3. Generate Noise
# ==================================================================================================

if args.gan_model_type == 'gan1' or 'wgan' or 'gan2' or 'gan3' or 'gan4' or 'wgan_gp':
    testType = 'naive_gan'

print(testType)
t_classifier = trainer.EvaluatorFactory.get_evaluator(args.sample_num, args.num_of_output, testType)

# mean result
print("mean_train mean, std for scaling: ", Y_train_mean, Y_train_std)    



if args.mode == 'eval':
  #< for past dataset that did not have seperate test datset >

#     test_dataset_loader = data_handler.SemiLoader(args, dataset.test_X_per_cycle, 
#                                                        dataset.test_Y_per_cycle, 
#                                                        X_train_mean, X_train_std, Y_train_mean, Y_train_std)    

    test_dataset_loader = data_handler.SemiLoader(args, dataset_test.test_X_per_cycle, 
                                                       dataset_test.test_Y_per_cycle, 
                                                       X_train_mean, X_train_std, Y_train_mean, Y_train_std)
    
    
    test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=1, shuffle=False)
    
# if args.mode == 'train':
#     mean_result, mean_total = t_classifier.mean_sample(mean_best_model, Y_train_mean, Y_train_std, mean_train_iterator)
# else:
#     mean_result, mean_total = t_classifier.mean_sample(mean_best_model, Y_train_mean, Y_train_std, test_mean_iterator)
    
    
if args.mode == 'train':
    total_result, total_num = t_classifier.sample(generator, Y_train_mean, Y_train_std, val_iterator, args.num_of_input, args.num_of_output, args.noise_d)
else:
    t_gen_start = time.time()
    total_result, total_num = t_classifier.sample(generator, Y_train_mean, Y_train_std, test_iterator, args.num_of_input, args.num_of_output, args.noise_d)
    t_gen_end = time.time()


# # 3: num_of_input
# # 6: num_of_output

# mean_result = np.repeat(mean_result, args.sample_num, axis=0)
# print('mean_result',mean_result)
# total_result = noise_result + mean_result
    
if args.mode == 'train':
    print("train time: ", t_end-t_start)

    np.save('./sample_data/'+log_name, total_result)

else:
    print("gen time: ", t_gen_end-t_gen_start)

#     np.save('./sample_data/'+'old_test_specific'+log_name, total_result)
    np.save('./sample_data/'+'test_specific'+log_name, total_result)