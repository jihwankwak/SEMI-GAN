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

log_name = 'date_{}_data_{}_model_{}_{}_seed_{}_lr_{}_{}_{}_hidden_dim_{}_{}_batch_size_{}_epoch_{}_{}_noise_d_{}_sample_num_{}'.format(
    args.date,
    args.data_type,
    args.mean_model_type,
    args.gan_model_type,    
    args.seed,
    args.mean_lr,  
    args.g_lr,      
    args.d_lr,        
    args.mean_hidden_dim,
    args.gan_hidden_dim,
    args.batch_size,  
    args.mean_nepochs,
    args.gan_nepochs,
    args.noise_d,
    args.sample_num
    )

if args.pdrop is not None:
    log_name += '_pdrop_{}'.format(args.pdrop)

print(log_name)

print("="*100)
print("Arguments =")
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print("="*100)

# Dataset
dataset = data_handler.DatasetFactory.get_dataset(args)
dataset_test = data_handler.DatasetFactory.get_test_dataset(args)

# loss result
result_dict = {}

kwargs = {'num_workers': 4}

print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
print("Inits...")

# ==================================================================================================
#                                          1. Predict mean 
# ==================================================================================================

mean_train_dataset_loader = data_handler.SemiLoader(dataset.train_X_per_cycle, 
                                                    dataset.train_Y_per_cycle, 
                                                    utils.normalize)

mean_val_dataset_loader = data_handler.SemiLoader(dataset.val_X_per_cycle, 
                                                  dataset.val_Y_per_cycle,
                                                  utils.normalize)

mean_test_dataset_loader = data_handler.SemiLoader(dataset.test_X_per_cycle, 
                                                   dataset.test_Y_per_cycle, 
                                                   utils.normalize)

# Dataloader

mean_train_iterator = torch.utils.data.DataLoader(mean_train_dataset_loader, batch_size=args.batch_size, shuffle=True, **kwargs)

mean_val_iterator = torch.utils.data.DataLoader(mean_val_dataset_loader, batch_size=args.batch_size, shuffle=False, **kwargs)

mean_test_iterator = torch.utils.data.DataLoader(mean_test_dataset_loader, batch_size=1, shuffle=False, **kwargs)

# model

mean_model = networks.ModelFactory.get_mean_model(args)

# weight initialization

mean_model.apply(utils.init_normal)
mean_model.cuda()

print(mean_model)

# optimizer

optimizer = torch.optim.Adam(mean_model.parameters(), lr=args.mean_lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) # scheduler

# trainer

mean_mytrainer = trainer.TrainerFactory.get_mean_trainer(mean_train_iterator, mean_val_iterator, mean_model, args, optimizer, exp_lr_scheduler)

if args.mode == 'train':
    for epoch in range(args.mean_nepochs):

        train_loss = mean_mytrainer.train()

        val_loss, val_r2 = mean_mytrainer.evaluate()

        current_lr = mean_mytrainer.current_lr

        if((epoch+1)% 10 == 0):
            print("epoch:{:2d}, lr:{:.6f}, || train_loss:{:.6f}, val_loss:{:.6f}, r2_score:{:.6f}".format(epoch, current_lr, train_loss, val_loss, val_r2))

    result_dict['train_loss'] = mean_mytrainer.loss['train_loss']
    result_dict['val_loss'] = mean_mytrainer.loss['val_loss']

    mean_best_model = mean_mytrainer.best_model
    if not os.path.exists('./mean_models'):
        os.makedirs('./mean_models')
    torch.save(mean_best_model.state_dict(), './mean_models/'+log_name)
else :
    print('Load mean model----------------')
    mean_mytrainer.model.load_state_dict(torch.load('./mean_models/'+args.model_path))
    mean_best_model = mean_mytrainer.model

# ==================================================================================================
#                                          2. Predict noise
# ==================================================================================================

noise_train_dataset_loader = data_handler.SemiLoader(dataset.train_X, 
                                                     dataset.train_Y_noise, 
                                                     utils.normalize)

noise_val_dataset_loader = data_handler.SemiLoader(dataset.val_X, 
                                                   dataset.val_Y_noise, 
                                                   utils.normalize)

noise_test_dataset_loader = data_handler.SemiLoader(dataset.test_X,
                                                    dataset.test_Y_noise, 
                                                    utils.normalize)

# Dataloader

noise_train_iterator = torch.utils.data.DataLoader(noise_train_dataset_loader, batch_size=args.batch_size, shuffle=True, **kwargs)

noise_val_iterator = torch.utils.data.DataLoader(noise_val_dataset_loader, batch_size=args.batch_size, shuffle=False, **kwargs)

noise_test_iterator = torch.utils.data.DataLoader(noise_test_dataset_loader, batch_size=1, shuffle=False, **kwargs)

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

gan_mytrainer = trainer.TrainerFactory.get_gan_trainer(noise_train_iterator, noise_val_iterator, generator, discriminator, args, optimizer_g, optimizer_d, exp_gan_lr_scheduler)

if args.mode == 'train':
    for epoch in range(args.gan_nepochs):

        gan_mytrainer.train()

        p_real, p_fake = gan_mytrainer.evaluate()

        current_d_lr = gan_mytrainer.current_d_lr

        if((epoch+1)% 10 == 0):
            print("epoch:{:2d}, lr_d:{:.6f}, || p_real:{:.6f}, p_fake:{:.6f}".format(epoch, current_d_lr, p_real, p_fake))

    result_dict['p_real_train'] = gan_mytrainer.prob['p_real_train']
    result_dict['p_fake_train'] = gan_mytrainer.prob['p_fake_train']
    result_dict['p_real_val'] = gan_mytrainer.prob['p_real_val']
    result_dict['p_fake_val'] = gan_mytrainer.prob['p_fake_val']

    if not os.path.exists('./result_loss'):
        os.makedirs('./result_loss')
    np.save('./result_loss/'+log_name, result_dict)

    # net.state_dict()
    if not os.path.exists('./gen_models'):
        os.makedirs('./gen_models')
    torch.save(generator.state_dict(), './gen_models/'+log_name)
    if not os.path.exists('./dis_models'):
        os.makedirs('./dis_models')
    torch.save(discriminator.state_dict(), './dis_models/'+log_name)
else : 
    gan_mytrainer.G.load_state_dict(torch.load('./gen_models/'+args.model_path))
#    gan_mytrainer.D.load_state_dict(torch.load('./gen_models/'+args.model_path))


# ==================================================================================================
#                                          3. Generate Noise
# ==================================================================================================
if args.mode == 'eval':
    test_mean_dataset_loader = data_handler.SemiLoader(dataset_test.test_X_per_cycle, 
                                                       dataset_test.test_Y_per_cycle, 
                                                       utils.normalize)

    test_mean_iterator = torch.utils.data.DataLoader(test_mean_dataset_loader, batch_size=1, shuffle=False, **kwargs)

if args.gan_model_type == 'gan1' or 'wgan' or 'gan2' or 'gan3':
    testType = 'gan'
    
t_classifier = trainer.EvaluatorFactory.get_evaluator(args.sample_num, testType)

# mean result
mean_train_mean = mean_train_dataset_loader.data_y_mean
mean_train_std = mean_train_dataset_loader.data_y_std

print(mean_train_mean, mean_train_std)    

if args.mode == 'train':
    mean_result, mean_total = t_classifier.mean_sample(mean_best_model, mean_train_mean, mean_train_std, mean_test_iterator)
else:
    mean_result, mean_total = t_classifier.mean_sample(mean_best_model, mean_train_mean, mean_train_std, test_mean_iterator)


# noise result
noise_train_mean = noise_train_dataset_loader.data_y_mean
noise_train_std = noise_train_dataset_loader.data_y_std

if args.mode == 'train':
    noise_result, noise_total = t_classifier.noise_sample(mean_result, generator, noise_train_mean, noise_train_std, mean_test_iterator, 5, 6, args.noise_d)
else:
    noise_result, noise_total = t_classifier.noise_sample(mean_result, generator, noise_train_mean, noise_train_std, test_mean_iterator, 5, 6, args.noise_d)
    
# 5: num_of_input + 1
# 6: num_of_output

mean_result = np.repeat(mean_result, args.sample_num, axis=0)
total_result = noise_result + mean_result

if not os.path.exists('./sample_data'):
    os.makedirs('./sample_data')
if args.mode == 'train':
    np.save('./sample_data/'+log_name, total_result)
else:
    np.save('./sample_data/'+'test_specific'+log_name, total_result)