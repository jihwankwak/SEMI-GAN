#! /bin/bash


# def set_seed(seed):
# torch.manual_seed(seed)
# # torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


# python3 main.py --date "taeeon_gan1_test" --dataset "2020_LER_20201008_V008.xlsx" --trainer "gan" --mean_model_type "mlp" --gan_model_type "gan1" --batch_size 25 --noise_d 100 --gan_hidden_dim 100 --mean_nepochs 100 --gan_nepochs 10

# python3 main-renewal.py --date "notseparated_gan1_default" --dataset "2020_LER_20201008_V008.xlsx" -
# -trainer "gan" --mean_model_type "mlp" --gan_model_type "gan1" --batch_size 25 --noise_d 100 --gan_hidden_dim 100

# python3 main.py --date "gan1_cyclenormalization" --dataset "2020_LER_20201008_V008.xlsx" --trainer "gan" --mean_model_type "mlp" --gan_model_type "gan1" --batch_size 25 --noise_d 100 --gan_hidden_dim 100



# python3 main-naive_gan.py --date "naive_gan" --dataset "2020_LER_20201008_V008.xlsx" --trainer "gan" --mean_model_type "mlp" --gan_model_type "gan1" --batch_size 25 --noise_d 100 --gan_hidden_dim 100 --mean_nepochs 100 --gan_nepochs 20



# python3 main-naive_gan.py --date "naive_gan_default" --dataset "2020_LER_20201008_V008.xlsx" --trainer "gan" --mean_model_type "mlp" --gan_model_type "gan1"


# python3 main-linear_gaussian.py --date "linear_gaussian_tmp" --dataset "2020_LER_20201008_V008.xlsx" --trainer "linear_gaussian" --batch_size 25  --gaussian_nepochs 100


# python3 main-mlp_gaussian.py --date "mlp_gaussian_v1" --dataset "2020_LER_20201008_V008.xlsx" --trainer "mlp_gaussian" --batch_size 25  --gaussian_nepochs 200 --lr 1e-5 --trainer "mlp_gaussian" --model_type "mlp_gaussian" --mean_hidden_dim 100

# python3 main-mlp_gaussian.py --date "mlp_gaussian_v1" --dataset "2020_LER_20201008_V008.xlsx" --trainer "mlp_gaussian" --batch_size 25  --gaussian_nepochs 2000 --lr 1e-5 --trainer "mlp_gaussian" --model_type "mlp_gaussian" --mean_hidden_dim 100

python3 main-mlp_gaussian.py --date "mlp_gaussian_v1" --dataset "2020_LER_20201008_V008.xlsx" --trainer "mlp_gaussian" --batch_size 25  --gaussian_nepochs 100 --lr 1e-5 --trainer "mlp_gaussian" --model_type "mlp_gaussian" --mean_hidden_dim 100

python3 main-mlp_gaussian.py --date "mlp_gaussian_v1" --dataset "2020_LER_20201008_V008.xlsx" --trainer "mlp_gaussian" --batch_size 25  --gaussian_nepochs 100 --lr 1e-5 --trainer "mlp_gaussian" --model_type "mlp_gaussian" --mean_hidden_dim 100 --mode eval




python3 main-mlp_gaussian.py --date "mlp_gaussian_v1" --dataset "2020_LER_20201008_V008.xlsx" --trainer "mlp_gaussian" --batch_size 25  --gaussian_nepochs 200 --lr 1e-5 --trainer "mlp_gaussian" --model_type "mlp_gaussian" --mean_hidden_dim 100

python3 main-mlp_gaussian.py --date "mlp_gaussian_v1" --dataset "2020_LER_20201008_V008.xlsx" --trainer "mlp_gaussian" --batch_size 25  --gaussian_nepochs 200 --lr 1e-5 --trainer "mlp_gaussian" --model_type "mlp_gaussian" --mean_hidden_dim 100 --mode eval




python3 main-mlp_gaussian.py --date "mlp_gaussian_v1" --dataset "2020_LER_20201008_V008.xlsx" --trainer "mlp_gaussian" --batch_size 25  --gaussian_nepochs 2000 --lr 1e-5 --trainer "mlp_gaussian" --model_type "mlp_gaussian" --mean_hidden_dim 100

python3 main-mlp_gaussian.py --date "mlp_gaussian_v1" --dataset "2020_LER_20201008_V008.xlsx" --trainer "mlp_gaussian" --batch_size 25  --gaussian_nepochs 2000 --lr 1e-5 --trainer "mlp_gaussian" --model_type "mlp_gaussian" --mean_hidden_dim 100 --mode eval



python3 main-mlp_gaussian.py --date "mlp_gaussian_v1" --dataset "2020_LER_20201008_V008.xlsx" --trainer "mlp_gaussian" --batch_size 25  --gaussian_nepochs 20000 --lr 1e-5 --trainer "mlp_gaussian" --model_type "mlp_gaussian" --mean_hidden_dim 100

python3 main-mlp_gaussian.py --date "mlp_gaussian_v1" --dataset "2020_LER_20201008_V008.xlsx" --trainer "mlp_gaussian" --batch_size 25  --gaussian_nepochs 20000 --lr 1e-5 --trainer "mlp_gaussian" --model_type "mlp_gaussian" --mean_hidden_dim 100 --mode eval



python3 main-mlp_gaussian.py --date "mlp_gaussian_v1" --dataset "2020_LER_20201008_V008.xlsx" --trainer "mlp_gaussian" --batch_size 25  --gaussian_nepochs 200000 --lr 1e-5 --trainer "mlp_gaussian" --model_type "mlp_gaussian" --mean_hidden_dim 100

python3 main-mlp_gaussian.py --date "mlp_gaussian_v1" --dataset "2020_LER_20201008_V008.xlsx" --trainer "mlp_gaussian" --batch_size 25  --gaussian_nepochs 200000 --lr 1e-5 --trainer "mlp_gaussian" --model_type "mlp_gaussian" --mean_hidden_dim 100 --mode eval


python3 ../../../run_slacker.py --name "finishied"