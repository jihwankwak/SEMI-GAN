import torch

class ModelFactory():
    def __init__(self):
        pass
    
    @staticmethod
    def get_mean_model(args):
        
        num_of_input = args.num_of_input
        
        if args.data_type == 'n' or args.data_type == 'p':
            num_of_input += 1
        elif args.data_type == 'none':
            pass
        else:
            num_of_input += 2
        
        if args.mean_model_type == 'mlp':
            
            import networks.mean_mlp as mean_mlp
            return mean_mlp.Net(mean_hidden_dim=args.mean_hidden_dim, num_of_input=num_of_input, num_of_output=args.num_of_output)
            
    def get_gan_model(args):
        
        num_of_input = args.num_of_input
        
        if args.data_type == 'n' or args.data_type == 'p':
            num_of_input += 1
        elif args.data_type == 'none':
            pass 
        else:
            num_of_input += 2
        
        if args.gan_model_type == 'gan1':
            
            import networks.gan1 as gan
            return gan.gen1(args.noise_d+num_of_input, args.gan_hidden_dim, args.num_of_output), gan.dis1(args.num_of_output+num_of_input, args.gan_hidden_dim)
        
        elif args.gan_model_type == 'wgan':
            
            import networks.wgan as gan
            return gan.wgan_gen(args.noise_d+num_of_input, args.gan_hidden_dim, args.num_of_output), gan.wgan_dis(args.num_of_output+num_of_input, args.gan_hidden_dim)
    
    def get_vae_model(args):
        
        num_of_input = args.num_of_input
        num_of_output = args.num_of_output
        hidden_dim = args.vae_hidden_dim
        noise_d = args.noise_d
        
        if args.data_type == 'n' or args.data_type == 'p':
            num_of_input += 1
        elif args.data_type == 'none':
            pass 
        else:
            num_of_input += 2
        
        if args.gan_model_type == 'vae1':
            
            import networks.vae1 as gan
            return gan.VAE(noise_d, hidden_dim, num_of_output, num_of_input)

    def get_gaussian_model(args):
        
        num_of_input = args.num_of_input
        num_of_output = ((args.num_of_output)**2 - args.num_of_output)//2 + args.num_of_output*2
        
        if args.data_type == 'n' or args.data_type == 'p':
            num_of_input += 1
        elif args.data_type == 'none':
            pass 
        else:
            num_of_input += 2
            
        if args.trainer == 'gaussian':
            
            import networks.mean_mlp as gaussian
            return gaussian.Net(mean_hidden_dim=args.mean_hidden_dim, num_of_input=num_of_input, num_of_output=num_of_output)