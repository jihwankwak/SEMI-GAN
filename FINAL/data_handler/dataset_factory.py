import data_handler.dataset as data

class DatasetFactory:
    def __init__(self):
        pass
    
    def get_dataset(args):
        if args.dataset == '2020_LER_20200804_V006.xlsx':
            return data.SEMI_gan_data(args.dataset, num_in_cycle=50, num_of_cycle=100, num_train=75, num_val=10)
        elif args.dataset == '2020_LER_20201008_V008.xlsx':
            return data.SEMI_gan_data(args.dataset, num_in_cycle=50, num_of_cycle=127, num_train=112, num_val=15)
        
    def get_test_dataset(args):
        if args.dataset_test == '2020_LER_20200922_testset.xlsx':
            return data.SEMI_sample_data(args.dataset_test)