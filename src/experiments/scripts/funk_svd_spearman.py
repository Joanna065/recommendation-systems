import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.data_processing.process.users_selection import remove_user_rates, select_user_ids
from src.experiments.collaborative_filtering.funk_svd_exp import (run_funk_svd_experiment,
                                                                  save_logs, save_model)
from src.settings import DATA_DIR, RESULT_DIR

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# load ratings data saved in format for funk svd library
MERGED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'merged_ml25m_kaggle')
ratings = pd.read_csv(os.path.join(MERGED_DATA_PATH, 'ratings_merged.csv'))
ratings.drop(columns=['timestamp'], inplace=True)
ratings.rename(columns={'movieId': 'i_id', 'userId': 'u_id'}, inplace=True)

LOGS_DIR = os.path.join(RESULT_DIR, 'logs', 'funk_svd')
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
MODEL_DIR = os.path.join(RESULT_DIR, 'models', 'funk_svd')
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

date = datetime.today()
date_now = date.strftime('%Y-%m-%d_%H-%M')

train_params = {
    'latent_factors': 45,
    'lr': 0.01,
    'regularization': 0.02,
    'epochs': 100,
    'min_rate': 0.5,
    'max_rate': 5.0,
    'shuffle': True
}

USERS_AMOUNT = 100
DROP_RATES = [0.1, 0.3, 0.5, 0.7]

# choose random users
user_ids = select_user_ids(ratings, users_amount=1000, user_colname='u_id', min_rate_amount=400,
                           min_rate_avg=2.75, max_rate_avg=4.75)

correlation_dict = {}

for drop_rate in DROP_RATES:
    # remove some user ratings from data based on drop rate
    new_ratings, drop_rates_df = remove_user_rates(ratings, user_ids, user_colname='u_id',
                                                   drop_rate=drop_rate)

    # run training with early stopping
    model, model_dump_dict, logs_dict, trained_epochs = run_funk_svd_experiment(new_ratings,
                                                                                train_params,
                                                                                train_split=0.9,
                                                                                val=True,
                                                                                test_with_val=False,
                                                                                seed=25)
    exp_name = f'{date_now}_FunkSVD-{train_params["latent_factors"]}-factors-' \
               f'{train_params["lr"]}-lr-{train_params["regularization"]}-reg-' \
               f'{trained_epochs}-epochs-{drop_rate}-DROP_RATE'

    save_model(MODEL_DIR, exp_name, model_dump_dict)
    save_logs(LOGS_DIR, exp_name, logs_dict)

    # calculate Spearman correlations for each selected user
    spearman_values = []
    true_rates_list = []
    pred_rates_list = []

    for user_id in user_ids:
        user_drop_rates = drop_rates_df[drop_rates_df['u_id'] == user_id]
        user_drop_rates = user_drop_rates.sort_values(by=['i_id'])

        true_rates = user_drop_rates['rating'].values
        true_rates_list.extend(true_rates)

        user_drop_rates = user_drop_rates.drop(['rating'], axis=1)
        pred_rates = model.predict(user_drop_rates)
        pred_rates_list.extend(pred_rates)

        spearman, p_val = spearmanr(true_rates, pred_rates)
        spearman_values.append((spearman, p_val))

    correlation_dict[f'drop_{drop_rate}'] = {}
    correlation_dict[f'drop_{drop_rate}']['spearman'] = spearman_values
    correlation_dict[f'drop_{drop_rate}']['true_rates'] = true_rates_list
    correlation_dict[f'drop_{drop_rate}']['pred_rates'] = pred_rates_list

# save correlation results
with open(os.path.join(LOGS_DIR, f'{date_now}_FunkSVD_spearman-drop-rates.pkl'), 'wb') as f:
    pickle.dump(correlation_dict, f)
