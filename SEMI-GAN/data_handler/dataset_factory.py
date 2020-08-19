import data_handler.dataset as data

class DatasetFactory:
    def __init__(self):
        pass
    
    @staticmethod
    def get_dataset(args):
        if args.dataset == 'LER_data_20191125.xlsx':
            return data.SEMI_data(args.dataset, args.data_type, num_input=4, num_output=6, num_in_cycle=10, num_of_cycle=270, num_train=230, num_val=20, num_test=20, x_cols="D:G", y_cols="K:S", header=2)
        elif args.dataset == 'LER_data_20191107.xlsx':
            return data.SEMI_data(args.dataset, args.data_type, num_input=4, num_output=6, num_in_cycle=10, num_of_cycle=185, num_train=150, num_val=15 , num_test=20, x_cols="D:G", y_cols="K:S", header=2)
        elif args.dataset == '2020_LER_20200529_V004.xlsx':
            return data.SEMI_data(args.dataset, args.data_type, num_input=4, num_output=6, num_in_cycle=50, num_of_cycle=72, num_train=50, num_val=10, num_test=12, x_cols="D:G", y_cols="H:P", header=0)
        elif args.dataset == '2020_LER_20200804_V006.xlsx':
            return data.SEMI_data(args.dataset, args.data_type, num_input=4, num_output=6, num_in_cycle=50, num_of_cycle=200, num_train=150, num_val=20, num_test=30, x_cols="B:G", y_cols="H:P", header=0)
        