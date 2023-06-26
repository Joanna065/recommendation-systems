import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from data_processing.process.users_selection import remove_user_rates, select_user_ids
from models.collaborative_filtering.knn import KNNModel
from settings import DATA_DIR, RESULT_DIR

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

MERGED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'merged_ml25m_kaggle')
ratings = pd.read_csv(os.path.join(MERGED_DATA_PATH, 'ratings_funk_svd_format.csv'))

LOGS_DIR = os.path.join(RESULT_DIR, 'logs', 'cf')
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

date = datetime.today()
date_now = date.strftime('%Y-%m-%d_%H-%M')

USERS_AMOUNT = 1000
DROP_RATES = [0.1, 0.3, 0.5, 0.7]

# choose random users
user_ids = select_user_ids(ratings, users_amount=USERS_AMOUNT, user_colname='u_id',
                           min_rate_amount=400,
                           min_rate_avg=2.75, max_rate_avg=4.75)

correlation_dict = {}

movies_path = os.path.join(DATA_DIR, 'processed', 'merged_ml25m_kaggle', 'movies_merged.csv')
ratings_path = os.path.join(DATA_DIR, 'processed', 'merged_ml25m_kaggle', 'ratings_merged.csv')
knn_model = KNNModel(movies_path, ratings_path, 3.5, 'cosine')
knn_model.preprocess_data()

for drop_rate in DROP_RATES:
    # remove some user ratings from data based on drop rate
    new_ratings, drop_rates_df = remove_user_rates(ratings, user_ids, user_colname='u_id',
                                                   drop_rate=drop_rate)

    # calculate Spearman correlations for each selected user
    true_rates_list = []

    spearman_knn_values = []
    pred_rates_knn_list = []

    for user_id in tqdm(user_ids, desc=f"Drop rate: {drop_rate}"):
        user_drop_rates = drop_rates_df[drop_rates_df['u_id'] == user_id]
        user_drop_rates = user_drop_rates.sort_values(by=['i_id'])

        user_left_rates = new_ratings[new_ratings['u_id'] == user_id].query('rating > 3.5')

        true_rates = user_drop_rates['rating'].values
        true_rates_list.extend(true_rates)

        movie_ids = user_drop_rates['i_id'].values.tolist()

        pred_knn_recomendations = knn_model.get_recommendations_for_movies_from_id(
            user_left_rates['i_id'].values, 1000000)

        pred_knn_recomendations = [t for t in pred_knn_recomendations if t[1] in movie_ids]

        pred_knn_recomendations.sort(key=lambda tup: tup[1])
        pred_knn_distances = [t[2] for t in pred_knn_recomendations]

        pred_rates_knn_list.extend(pred_knn_distances)

        spearman, p_val = spearmanr(true_rates, pred_knn_distances)
        spearman_knn_values.append((spearman, p_val))

    correlation_dict[f'drop_{drop_rate}'] = {}
    correlation_dict[f'drop_{drop_rate}']['spearman'] = spearman_knn_values
    correlation_dict[f'drop_{drop_rate}']['true_rates'] = true_rates_list
    correlation_dict[f'drop_{drop_rate}']['pred_rates'] = pred_rates_knn_list

# save correlation results
with open(os.path.join(LOGS_DIR, f'{date_now}_KNN_spearman.pkl'), 'wb') as f:
    pickle.dump(correlation_dict, f)
