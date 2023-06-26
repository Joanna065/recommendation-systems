import itertools
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from experiments.collaborative_filtering.funk_svd_exp import (run_funk_svd_experiment, save_logs,
                                                              save_model)
from settings import DATA_DIR, RESULT_DIR

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
SAVE = True

# load data
MERGED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'merged_ml25m_kaggle')
ratings_df = pd.read_csv(os.path.join(MERGED_DATA_PATH, 'ratings_merged.csv'))
ratings_df.drop(columns=['timestamp'], inplace=True)
ratings_df.rename(columns={'movieId': 'i_id', 'userId': 'u_id'}, inplace=True)

date = datetime.today()
date_now = date.strftime('%Y-%m-%d_%H-%M')

train_params = {
    'latent_factors': [45, 50, 55, 60, 65],
    'lr': [0.005, 0.01],
    'regularization': [0.01, 0.02],
    'epochs': [50],
    'min_rate': [0.5],
    'max_rate': [5.0],
    'shuffle': [True],
}

for params in itertools.product(*train_params.values()):
    params = dict(zip(train_params.keys(), params))

    # run training with early stopping on val loss
    model, model_dump_dict, logs_dict, trained_epochs = run_funk_svd_experiment(ratings_df, params,
                                                                                train_split=0.8,
                                                                                val=True, seed=25)

    # save experiment model and logs
    if SAVE:
        exp_name = f'{date_now}_FunkSVD-{params["latent_factors"]}-factors-' \
                   f'{params["lr"]}-lr-{params["regularization"]}-reg-' \
                   f'{trained_epochs}-epochs-{params["shuffle"]}-shuffle'

        logs_dir = os.path.join(RESULT_DIR, 'logs', 'funk_svd')
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        model_dir = os.path.join(RESULT_DIR, 'models', 'funk_svd')
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        save_model(model_dir, exp_name, model_dump_dict)
        save_logs(logs_dir, exp_name, logs_dict)
