import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr

from datasets.neumf_dataset import get_separate_users_movies_ratings
from src.experiments.collaborative_filtering.neumf_params import get_params
from src.models.collaborative_filtering.neural_mf import F1Score, Precision, Recall, create_model
from src.settings import DATA_DIR, RESULT_DIR

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# load test and train ratings split
test_ratings = pd.read_csv(
    Path(DATA_DIR) / 'datasets' / 'compare_split' / 'ratings_test_1k_users.csv')
train_ratings = pd.read_csv(
    Path(DATA_DIR) / 'datasets' / 'compare_split' / 'ratings_train_1k_users.csv')

# data constants
ITEM_COLUMN, USER_COLUMN, RATING_COLUMN = 'movieId', 'userId', 'rating'
num_movies = np.unique(train_ratings[ITEM_COLUMN].values).size
num_users = np.unique(train_ratings[USER_COLUMN].values).size

# map users and movies ids to range (0, N-1)
unique_user_ids = np.unique(train_ratings[USER_COLUMN].values)
unique_movie_ids = np.unique(train_ratings[ITEM_COLUMN].values)
user_ids_dict = {id: counter for counter, id in enumerate(unique_user_ids)}
movie_ids_dict = {id: counter for counter, id in enumerate(unique_movie_ids)}

test_ratings[USER_COLUMN] = test_ratings[USER_COLUMN].apply(lambda x: user_ids_dict[x])
test_ratings[ITEM_COLUMN] = test_ratings[ITEM_COLUMN].apply(lambda x: movie_ids_dict[x])

num_test_ratings = test_ratings.shape[0]
test_ratings = np.concatenate(
    (np.array(test_ratings[USER_COLUMN], dtype=pd.Series).reshape(num_test_ratings, 1),
     np.array(test_ratings[ITEM_COLUMN], dtype=pd.Series).reshape(
         num_test_ratings, 1),
     np.array(test_ratings[RATING_COLUMN], dtype=pd.Series).reshape(num_test_ratings, 1)),
    axis=1)
test_ratings = test_ratings.astype(np.float64)
test_users, test_movies, y_test = get_separate_users_movies_ratings(test_ratings)

MODEL_NAME = '2020-06-02_17-25_NeuFM_compare_split'
MODEL_WEIGHTS_PATH = os.path.join(RESULT_DIR, 'checkpoints', f'{MODEL_NAME}/model_weights.ckpt')
LOGS_DIR = os.path.join(RESULT_DIR, 'logs', 'neural_mf', MODEL_NAME)
TEST_DATA_PATH = os.path.join(DATA_DIR, 'datasets', 'ratings_test_split_mapped_idx.npz')

correlation_dict = {}

# create model
params = get_params(num_users=num_users, num_items=num_movies)
model = create_model(params)

# define optimizer and callbacks
optimizer = tf.keras.optimizers.Adam(lr=params["learning_rate"],
                                     beta_1=params["beta1"],
                                     beta_2=params["beta2"],
                                     epsilon=params["epsilon"])

model.compile(loss=params['loss'],
              metrics=[Precision(), Recall(), F1Score()],
              optimizer=optimizer)

# load best weights
model.load_weights(MODEL_WEIGHTS_PATH)

pred_ratings = np.clip(model.predict([test_users, test_movies]), 0.5, 5.0)
true_ratings = y_test
print(f'Min pred: {np.min(pred_ratings)}, max pred: {np.max(pred_ratings)}, '
      f'avg pred: {np.mean(pred_ratings)}')
print(pred_ratings.shape)
print(true_ratings.shape)

correlation_dict['pred_rates'] = pred_ratings
correlation_dict['true_rates'] = true_ratings

spearman, p_val = spearmanr(true_ratings, pred_ratings)
print(f'Spearman corr: {spearman}, p_val: {p_val}')

correlation_dict['spearman'] = (spearman, p_val)

loss_and_metrics = model.evaluate(x=[test_users, test_movies], y=y_test,
                                  batch_size=params['batch_size'], verbose=1)
correlation_dict['test_loss'] = loss_and_metrics[0]
correlation_dict['metrics'] = loss_and_metrics[1:]

names = model.metrics_names
for name, value in zip(names, loss_and_metrics):
    print('{0:s} : {1:1.4f}'.format(name, value))

# save correlation results
with open(os.path.join(LOGS_DIR, 'spearman.pkl'), 'wb') as f:
    pickle.dump(correlation_dict, f)
