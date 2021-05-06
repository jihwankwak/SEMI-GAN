import argparse

def get_args():
    parser = argparse.ArgumentParser(description='SEMI')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--dataset', default='2020_LER_20201008_V008.xlsx', type=str, required=False,
                        choices=['LER_data_20191125.xlsx',
                                 'LER_data_20191107.xlsx',
                                 '2020_LER_20200529_V004.xlsx',
                                '2020_LER_20200804_V006.xlsx', 
                                '2020_LER_20200922_V007_testset_edit.csv',
                                '2020_LER_20201008_V008.xlsx'], 
                        help='(default=%(default)s)')
    parser.add_argument('--dataset_test', default='2020_LER_20201021_testset.xlsx', type=str, required=False,
                        choices=['2020_LER_20200922_testset.xlsx',
                                 '2020_LER_20201021_testset.xlsx'
                                ],
                        help='(default=%(default)s)')
    parser.add_argument('--mean_model_type', required=False, type=str, default = 'mlp',
                        choices=['mlp'], 
                        help='(default=%(default)s)')
    parser.add_argument('--relu_type',default = 'relu', required=False, type=str,
                        choices=['relu', 'leaky_relu', 'prelu'], 
                        help='(default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seeds values to be used; seed introduces randomness by changing order of classes')
    parser.add_argument('--mean_lr', type=float, default=5e-5,
                        help='learning rate (default: 5e-5. Note that mean_lr is decayed by args.gamma parameter args.schedule ')
    parser.add_argument('--mean_hidden_dim', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--mean_nepochs', type=int, default=1000, help='Number of epochs for each mean increment')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers in Dataloaders')
    parser.add_argument('--num_of_input', type=int, default=5, help='Number of input for data')
    parser.add_argument('--num_of_output', type=int, default=6, help='Number of output for data')
    parser.add_argument('--pdrop', type=float, default=0.9, help='dropout rate')
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])


    
#     parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()

    return args