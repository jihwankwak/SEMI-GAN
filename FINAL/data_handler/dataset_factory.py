import data_handler.dataset as data

class DatasetFactory:
    def __init__(self):
        pass
    
    def get_dataset(args):
        if args.trainer == 'linear_gaussian' or args.trainer == 'mlp_gaussian':
            
            # LER
            if args.dataset == '2020_LER_20201008_V008.xlsx':
                return data.SEMI_gaussian_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=127, num_train=112, num_val=15)
            elif args.dataset == '2020_LER_20201008_V008.xlsx_generated_part' or args.dataset == '2020_LER_20201008_V008.xlsx_generated_all':
                return data.SEMI_gaussian_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=250, num_train=249, num_val=1)
            # RDF, WFV, RDF+WFV
            elif args.dataset == 'rdfwfv_train2020_RDFWFV_20201222_V10.xlsx':
                return data.SEMI_gaussian_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=100, num_train=90, num_val=10)
            elif 'wfv_train2020_RDFWFV_20201222_V10.xlsx' or 'rdf_train2020_RDFWFV_20201222_V10.xlsx':
                return data.SEMI_gaussian_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=50, num_train=45, num_val=5)
            elif args.dataset == 'rdfwfv_wfv_train2020_RDFWFV_20201222_V10.xlsx':
                return data.SEMI_gaussian_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=150, num_train=135, num_val=15)
            elif args.dataset == 'rdfwfv_wfv_rdf_train2020_RDFWFV_20201222_V10.xlsx':
                return data.SEMI_gaussian_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=200, num_train=175, num_val=25) 
                
                
                
           
        elif args.trainer == 'gan' or 'wgan':
            # LER
            if args.dataset == '2020_LER_20200804_V006.xlsx':
                return data.SEMI_gan_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=100, num_train=75, num_val=10)
            elif args.dataset == '2020_LER_20201008_V008.xlsx':
                return data.SEMI_gan_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=127, num_train=112, num_val=15)
            elif args.dataset == '2020_LER_20201008_V008.xlsx_generated_part' or args.dataset == '2020_LER_20201008_V008.xlsx_generated_all':
                return data.SEMI_gan_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=250, num_train=249, num_val=1)
            # RDF, WFV, RDF+WFV
            elif args.dataset == 'rdfwfv_train2020_RDFWFV_20201222_V10.xlsx':
                return data.SEMI_gan_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=100, num_train=90, num_val=10)
            elif args.dataset == 'wfv_train2020_RDFWFV_20201222_V10.xlsx':
                print(2)
                return data.SEMI_gan_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=50, num_train=45, num_val=5)
            
            elif args.dataset == 'rdf_train2020_RDFWFV_20201222_V10.xlsx':
                print(1)
                return data.SEMI_gan_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=50, num_train=45, num_val=5)
            
            elif args.dataset == 'rdfwfv_wfv_train2020_RDFWFV_20201222_V10.xlsx':
                return data.SEMI_gan_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=150, num_train=135, num_val=15)
            elif args.dataset == 'rdfwfv_wfv_rdf_train2020_RDFWFV_20201222_V10.xlsx':
                return data.SEMI_gan_data(args.dataset, num_in_cycle=args.tr_num_in_cycle, num_of_cycle=200, num_train=175, num_val=25) 
            
            
        
    def get_test_dataset(args):
        if args.dataset_test == '2020_LER_20200922_testset.xlsx':
            return data.SEMI_sample_data(args.dataset_test)
        elif args.dataset_test == '2020_LER_20201021_testset.xlsx':
            return data.SEMI_sample_data(args.dataset_test)
        elif args.dataset_test == '2020_LER_20201102_testset_V04.xlsx':
            return data.SEMI_sample_data(args.dataset_test)
        elif args.dataset_test == '2020_LER_20201008_V008.xlsxadd_x_all':
            return data.SEMI_sample_data(args.dataset_test)
        # RDF, WFV, RDF+WFV
        elif args.dataset_test == 'rdfwfv_test2020_RDFWFV_20201222_V10.xlsx' or 'rdf_train2020_RDFWFV_20201222_V10.xlsx' or 'wfv_train2020_RDFWFV_20201222_V10.xlsx' or 'rdfwfv_wfv_train2020_RDFWFV_20201222_V10.xlsx' or 'rdfwfv_wfv_rdf_train2020_RDFWFV_20201222_V10.xlsx':
            return data.SEMI_sample_data(args.dataset_test)
