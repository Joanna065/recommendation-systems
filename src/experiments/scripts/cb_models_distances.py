import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.process.users_selection import select_user_ids
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

date = datetime.today()
date_now = date.strftime('%Y-%m-%d_%H-%M')

USERS_AMOUNT = 1000
DROP_RATES = [0.1, 0.3, 0.5, 0.7]
USER_COLNAME, MOVIE_COLNAME = 'userId', 'movieid'

# choose random users
user_ids = select_user_ids(ratings, users_amount=USERS_AMOUNT, user_colname=USER_COLNAME,
                           min_rate_amount=400, min_rate_avg=2.75, max_rate_avg=4.75)

text_model = TextModel(ratings_path=RATINGS_PATH, movies_path=MOVIES_PATH, rate_treshold=3.5)
text_model.preprocess_data()
tag_model = TagModel(ratings_path=RATINGS_PATH, movies_path=MOVIES_PATH, rate_treshold=3.5)
tag_model.preprocess_data()

text_rows = []
tag_rows = []

total_movies = len(text_model.movies_transformed)


def upper_tri_indexing(A):
    m = A.shape[0]
    r, c = np.triu_indices(m, 1)
    return A[r, c]


text_cos = np.copy(text_model.cosine_sim)
tag_cos = np.copy(tag_model.cosine_sim)

del text_model
del tag_model

final_text_row = upper_tri_indexing(text_cos)
final_tag_row = upper_tri_indexing(tag_cos)

# save correlation results
np.savez(os.path.join(LOGS_DIR, f'{date_now}_CB_distances'), text=final_text_row, tag=final_tag_row)
