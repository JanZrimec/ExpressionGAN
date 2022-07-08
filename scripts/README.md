# Scripts

The script 'train_gan.py' to train a generative model of regulatory DNA can be run via:

```python train_gan.py --data_loc "SeqsData" --log_name "gan_bal_200d" --train_iters 300000 --max_seq_len 1000 --latent_dim 200 --balanced_bins --seed 111```

where the folder SeqsData contains the training, validation and test sequence datasets in plain .txt files (e.g. train_data.txt, test_data.txt, valid_data.txt). Please see the script for further details on possible parameters.

The script 'optimize_gan.py' to optimize a trained generator with a predictive model to obtain an ExpressionGAN  can be run via:

```python optimize_gan.py --log_name "gan_bal_200d_opt" --generator "gen_path/trained_gan.ckpt.meta" --predictor "pred_path" --iterations 100000 --target 'max' --seed 222```

Please see the script for further details on possible parameters.
