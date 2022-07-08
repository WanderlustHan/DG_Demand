
# PEMS-BAY
# python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5 --horizon=12
#python dcrnn_train_pytorch.py --config_filename='data/model/dcrnn_la.yaml' --pre_k=2
python -m utils.generate_samples --eigen_k 15
python -m utils.generate_samples --eigen_k 30
python -m utils.generate_samples --eigen_k 50
python -m utils.generate_samples --eigen_k 100
