import data_handler.dataset as data
import data_handler.dataset_n as data_n

class DatasetFactory:
    def __init__(self):
        pass
    
    def get_dataset(args):
        if args.dataset == 'LER_data_20191125.xlsx':
            return data.SEMI_gan_data(args.dataset, args.data_type, num_input=4, num_output=6, num_in_cycle=10, num_of_cycle=270, num_train=230, num_val=20, num_test=20, x_cols="D:G", y_cols="K:S", header=2)
        elif args.dataset == 'LER_data_20191107.xlsx':
            return data.SEMI_gan_data(args.dataset, args.data_type, num_input=4, num_output=6, num_in_cycle=10, num_of_cycle=185, num_train=150, num_val=15 , num_test=20, x_cols="D:G", y_cols="K:S", header=2)
        elif args.dataset == '2020_LER_20200529_V004.xlsx':
            return data.SEMI_gan_data(args.dataset, args.data_type, num_input=4, num_output=6, num_in_cycle=50, num_of_cycle=72, num_train=50, num_val=10, num_test=12, x_cols="D:G", y_cols="H:P", header=0)
        elif args.dataset == '2020_LER_20200804_V006.xlsx':
            return data.SEMI_gan_data(args.dataset, args.data_type, num_input=4, num_output=6, num_in_cycle=50, num_of_cycle=200, num_train=150, num_val=20, num_test=30, x_cols="B:G", y_cols="H:P", header=0)
        elif args.dataset == '2020_LER_20200922_V007_testset_edit.csv':
            return data.SEMI_gan_data(args.dataset, args.data_type, num_input=4, num_output=6, num_in_cycle=50, num_of_cycle=236, num_train=88*2, num_val=15*2, num_test=15*2, x_cols=['PNMOS', 'amp.', 'corr.x', 'corr.y'], y_cols=['Ioff', 'IDSAT', 'IDLIN', 'VTSAT', 'VTLIN', 'SS'], header=0)
        # dataset change
        # 1) consider only n datatype
        elif args.dataset == '2020_LER_20201008_V008.xlsx':
            return data_n.SEMI_gan_data(args.dataset, num_input=3, num_output=6, num_in_cycle=50, num_of_cycle=127, num_train=75, num_val=15, x_cols="D:F", y_cols="H:P", header=0)
        
    def get_test_dataset(args):
        if args.dataset_test == '2020_LER_20200922_testset.csv':
            return data.SEMI_sample_data(args.dataset_test, num_input=3, num_output=6, num_in_cycle=[232, 289, 277, 253, 255], num_of_cycle=5, x_cols=['PNMOS', 'amp.', 'corr.x', 'corr.y'], y_cols=['Ioff', 'IDSAT', 'IDLIN', 'VTSAT', 'VTLIN', 'SS'], header=0)
        elif args.dataset_test == '2020_LER_20200922_testset.xlsx':
            return data_n.SEMI_sample_data(args.dataset_test, num_input=3, num_output=6, num_in_cycle=[232, 289, 277, 253, 255], num_of_cycle=5, x_cols="D:F", y_cols="H:P", header=0)