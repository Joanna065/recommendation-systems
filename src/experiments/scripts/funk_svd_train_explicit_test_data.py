import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.experiments.collaborative_filtering.funk_svd_exp import (run_funk_svd_experiment,
                                                                  save_logs, save_model)
from src.settings import DATA_DIR, RESULT_DIR

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# load test and train data
TEST_DATA_PATH = Path(DATA_DIR) / 'datasets' / 'compare_split' / 'ratings_test_1k_users.csv'
test_df = pd.read_csv(TEST_DATA_PATH)
test_df.drop(columns=['timestamp'], inplace=True)
test_df.rename(columns={'movieId': 'i_id', 'userId': 'u_id'}, inplace=True)

TRAIN_DATA_PATH = Path(DATA_DIR) / 'datasets' / 'compare_split' / 'ratings_train_1k_users.csv'
train_df = pd.read_csv(TRAIN_DATA_PATH)
train_df.drop(columns=['timestamp'], inplace=True)
train_df.rename(columns={'movieId': 'i_id', 'userId': 'u_id'}, inplace=True)

# specify save paths
LOGS_DIR = os.path.join(RESULT_DIR, 'logs', 'funk_svd')
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
MODEL_DIR = os.path.join(RESULT_DIR, 'models', 'funk_svd')
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

date = datetime.today()
date_now = date.strftime('%Y-%m-%d_%H-%M')

params = {
    'latent_factors': 45,
    'lr': 0.01,
    'regularization': 0.02,
    'epochs': 100,
    'min_rate': 0.5,
    'max_rate': 5.0,
    'shuffle': True
}

# run training with early stopping
model, model_dump_dict, logs_dict, trained_epochs = run_funk_svd_experiment(train_df,
                                                                            params,
                                                                            train_split=0.9,
                                                                            val=True,
                                                                            test_with_val=False,
                                                                            seed=25)

exp_name = f'{date_now}_FunkSVD-explicit-test-data'
save_model(MODEL_DIR, exp_name, model_dump_dict)
save_logs(LOGS_DIR, exp_name, logs_dict)

pred_rates = model.predict(test_df)

spearman, p_val = spearmanr(test_df.rating.values, pred_rates)
print(f'Spearman correlation: {spearman}')

correlation_dict = {
    'spearman': (spearman, p_val),
    'true_rates': test_df.rating.values.tolist(),
    'pred_rates': pred_rates
}

# save correlation results
with open(Path(LOGS_DIR) / f'{date_now}_FunkSVD_test_data_spearman.pkl', 'wb') as f:
    pickle.dump(correlation_dict, f)
