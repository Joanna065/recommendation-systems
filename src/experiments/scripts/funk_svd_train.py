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

# load ratings data saved in format for funk svd library
MERGED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'merged_ml25m_kaggle')
ratings = pd.read_csv(os.path.join(MERGED_DATA_PATH, 'ratings_funk_svd_format.csv'))

logs_dir = os.path.join(RESULT_DIR, 'logs', 'funk_svd')
Path(logs_dir).mkdir(parents=True, exist_ok=True)
model_dir = os.path.join(RESULT_DIR, 'models', 'funk_svd')
Path(model_dir).mkdir(parents=True, exist_ok=True)

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

# run experiment with validation and test split (early stopping)
_, model_dump_dict, logs_dict, trained_epochs = run_funk_svd_experiment(ratings, params,
                                                                        train_split=0.8,
                                                                        val=True, seed=25)

exp_name = f'{date_now}_FunkSVD-{params["latent_factors"]}-factors-' \
           f'{params["lr"]}-lr-{params["regularization"]}-reg-' \
           f'{trained_epochs}-epochs-{params["shuffle"]}-shuffle'

save_model(model_dir, exp_name, model_dump_dict)
save_logs(logs_dir, exp_name, logs_dict)

# run experiment with test split (fixed amount of epochs)
_, model_dump_dict, logs_dict, trained_epochs = run_funk_svd_experiment(ratings, params,
                                                                        train_split=0.9,
                                                                        seed=25)

exp_name = f'{date_now}_FunkSVD-{params["latent_factors"]}-factors-' \
           f'{params["lr"]}-lr-{params["regularization"]}-reg-' \
           f'{trained_epochs}-epochs-{params["shuffle"]}-shuffle'

save_model(model_dir, exp_name, model_dump_dict)
save_logs(logs_dir, exp_name, logs_dict)
