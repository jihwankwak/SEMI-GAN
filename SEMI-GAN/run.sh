#! /bin/bash

python3 main_gan.py --date "20200822" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1' --batch_size 10
python3 main_gan.py --date "20200822" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1'--batch_size 25
python3 main_gan.py --date "20200822" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1'--batch_size 50
python3 main_gan.py --date "20200822" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1'--noise_d 10
python3 main_gan.py --date "20200822" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1'--noise_d 100
python3 main_gan.py --date "20200822" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1'--noise_d 500
python3 main_gan.py --date "20200822" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1'--noise_d 1000
python3 main_gan.py --date "20200822" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1'--noise_d 5000
python3 main_gan.py --date "20200822" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1'--gan_hidden_dim 50
python3 main_gan.py --date "20200822" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1'--gan_hidden_dim 100
python3 main_gan.py --date "20200822" --dataset "2020_LER_20200804_V006.xlsx" --data_type "n" --trainer 'gan' --mean_model_type 'mlp' --gan_model_type 'gan1'--gan_hidden_dim 500
python ~/send_email.py finished