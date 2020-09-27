#! /bin/bash

#python3 main_gan.py --date "20200905" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1' --batch_size 10 --sample_num 500
#python3 main_gan.py --date "20200905" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1' --sample_num 500
#python3 main_gan.py --date "20200905" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1' --batch_size 50 --sample_num 500
#python3 main_gan.py --date "20200905" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1' --noise_d 500 --sample_num 500
#python3 main_gan.py --date "20200905" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1' --gan_hidden_dim 50 --sample_num 500
#python3 main_gan.py --date "20200909" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan2' --pdrop 0.5 --sample_num 50
python3 main_gan.py --date "20200927" --dataset "2020_LER_20200922_V007_testset_edit.csv" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1'
#python3 main_gan.py --date "20200927" --dataset "2020_LER_20200922_V007(testset_edit)" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1' --batch_size 10 --sample_num 150
#python3 main_gan.py --date "20200927" --dataset "2020_LER_20200922_V007(testset_edit)" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1' --batch_size 10 --sample_num 500

#python3 main_gan.py --date "20200927" --dataset "2020_LER_20200922_V007(testset_edit)" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1' --pdrop 0.5 --sample_num 500
#python3 main_gan.py --date "20200927" --dataset "2020_LER_20200922_V007(testset_edit)" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1' --pdrop 0.8 --sample_num 50
#python3 main_gan.py --date "20200927" --dataset "2020_LER_20200922_V007(testset_edit)" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1' --pdrop 0.8 --sample_num 150
#python3 main_gan.py --date "20200927" --dataset "2020_LER_20200922_V007(testset_edit)" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1' --pdrop 0.8 --sample_num 500

python send_email.py finished