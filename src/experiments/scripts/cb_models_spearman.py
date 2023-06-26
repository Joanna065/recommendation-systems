import logging
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from src.data_processing.process.users_selection import remove_user_rates, select_user_ids
from src.models.content_based.tag_model import TagModel
from src.models.content_based.text_model import TextModel
from src.settings import DATA_DIR, RESULT_DIR

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

RATINGS_PATH = Path(DATA_DIR) / 'processed' / 'merged_ml25m_kaggle' / 'ratings_merged.csv'
MOVIES_PATH = Path(DATA_DIR) / 'processed' / 'merged_ml25m_kaggle' / 'movies_merged.csv'
ratings = pd.read_csv(RATINGS_PATH)

LOGS_DIR = Path(RESULT_DIR) / 'logs' / 'cb'
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(RESULT_DIR) / 'models' / 'cb'
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

date = datetime.today()
date_now = date.strftime('%Y-%m-%d_%H-%M')

USERS_AMOUNT = 1000
DROP_RATES = [0.1, 0.3, 0.5, 0.7]
USER_COLNAME, MOVIE_COLNAME = 'userId', 'movieid'

# choose random users
user_ids = select_user_ids(ratings, users_amount=USERS_AMOUNT, user_colname=USER_COLNAME,
                           min_rate_amount=400,
                           min_rate_avg=2.75, max_rate_avg=4.75)

correlation_dict = {'text': {}, 'tag': {}}

text_model = TextModel(ratings_path=RATINGS_PATH, movies_path=MOVIES_PATH, rate_treshold=3.5)
text_model.preprocess_data()
tag_model = TagModel(ratings_path=RATINGS_PATH, movies_path=MOVIES_PATH, rate_treshold=3.5)
tag_model.preprocess_data()

for drop_rate in DROP_RATES:
    # remove some user ratings from data based on drop rate
    new_ratings, drop_rates_df = remove_user_rates(ratings, user_ids, user_colname=USER_COLNAME,
                                                   drop_rate=drop_rate)

    # calculate Spearman correlations for each selected user
    spearman_text_values = []
    true_rates_list = []
    pred_rates_text_list = []

    spearman_tag_values = []
    pred_rates_tag_list = []

    for user_id in tqdm(user_ids, desc=f"Drop rate: {drop_rate}"):
        user_drop_rates = drop_rates_df[drop_rates_df[USER_COLNAME] == user_id]
        user_drop_rates = user_drop_rates.sort_values(by=[MOVIE_COLNAME])

        user_left_rates = new_ratings[new_ratings[USER_COLNAME] == user_id].query('rating > 3.5')

        true_rates = user_drop_rates['rating'].values
        true_rates_list.extend(true_rates)

        movie_ids = user_drop_rates[MOVIE_COLNAME].values.tolist()

        pred_text_recomendations = text_model.get_recommendations_for_movies_from_id(
            user_left_rates[MOVIE_COLNAME].values, 1000000)
        pred_tag_recomendations = tag_model.get_recommendations_for_movies_from_id(
            user_left_rates[MOVIE_COLNAME].values, 1000000)

        pred_text_recomendations = [t for t in pred_text_recomendations if t[1] in movie_ids]
        pred_tag_recomendations = [t for t in pred_tag_recomendations if t[1] in movie_ids]

        pred_text_recomendations.sort(key=lambda tup: tup[1])
        pred_tag_recomendations.sort(key=lambda tup: tup[1])
        pred_text_distances = [t[2] for t in pred_text_recomendations]
        pred_tag_distances = [t[2] for t in pred_tag_recomendations]

        pred_rates_text_list.extend(pred_text_distances)
        pred_rates_tag_list.extend(pred_tag_distances)

        spearman, p_val = spearmanr(true_rates, pred_text_distances)
        spearman_text_values.append((spearman, p_val))
        spearman, p_val = spearmanr(true_rates, pred_tag_distances)
        spearman_tag_values.append((spearman, p_val))

    correlation_dict['text'][f'drop_{drop_rate}'] = {}
    correlation_dict['text'][f'drop_{drop_rate}']['spearman'] = spearman_text_values
    correlation_dict['text'][f'drop_{drop_rate}']['true_rates'] = true_rates_list
    correlation_dict['text'][f'drop_{drop_rate}']['pred_rates'] = pred_rates_text_list

    correlation_dict['tag'][f'drop_{drop_rate}'] = {}
    correlation_dict['tag'][f'drop_{drop_rate}']['spearman'] = spearman_tag_values
    correlation_dict['tag'][f'drop_{drop_rate}']['true_rates'] = true_rates_list
    correlation_dict['tag'][f'drop_{drop_rate}']['pred_rates'] = pred_rates_tag_list

# save correlation results
with open(Path(LOGS_DIR) / f'{date_now}_CB_spearman.pkl', 'wb') as f:
    pickle.dump(correlation_dict, f)
