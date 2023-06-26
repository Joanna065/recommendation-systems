import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.process.users_selection import remove_user_rates, select_user_ids
from src.settings import DATA_DIR

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# load data
movies_df = pd.read_csv(Path(DATA_DIR) / 'processed' / 'merged_ml25m_kaggle' / 'movies_merged.csv')
ratings_df = pd.read_csv(
    Path(DATA_DIR) / 'processed' / 'merged_ml25m_kaggle' / 'ratings_merged.csv')

USER_AMOUNT = 1000
USER_COLUMN = 'userId'
MOVIE_COLUMN = 'movieId'
SAVE_DIR = Path(DATA_DIR) / 'datasets' / 'compare_split'
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# select users and filter data
user_ids = select_user_ids(ratings_df, users_amount=USER_AMOUNT, user_colname=USER_COLUMN,
                           min_rate_amount=400, min_rate_avg=2.75, max_rate_avg=4.75)

train_ratings, test_ratings = remove_user_rates(ratings_df, user_ids, user_colname=USER_COLUMN,
                                                drop_rate=0.2)

test_movie_indices = np.unique(test_ratings[MOVIE_COLUMN].values)
test_movies = movies_df[movies_df[MOVIE_COLUMN].isin(test_movie_indices)]
print(f'Test movies amount: {len(test_movie_indices)}')

print('Removing movies outside test data from train ratings...')
train_ratings = train_ratings[train_ratings[MOVIE_COLUMN].isin(test_movie_indices)]

print(f'Test ratings amount: {len(test_ratings)}')
print(f'Train ratings amount: {len(train_ratings)}')

print('Saving train and test split')
test_movies.to_csv(Path(SAVE_DIR) / f'movies_test_{int(USER_AMOUNT / 1000)}k_users.csv',
                   index=False)
test_ratings.to_csv(Path(SAVE_DIR) / f'ratings_test_{int(USER_AMOUNT / 1000)}k_users.csv',
                    index=False)
train_ratings.to_csv(Path(SAVE_DIR) / f'ratings_train_{int(USER_AMOUNT / 1000)}k_users.csv',
                     index=False)
