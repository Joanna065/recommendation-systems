import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from datasets.neumf_dataset import get_separate_users_movies_ratings, split_data
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
train_ratings[USER_COLUMN] = train_ratings[USER_COLUMN].apply(lambda x: user_ids_dict[x])
train_ratings[ITEM_COLUMN] = train_ratings[ITEM_COLUMN].apply(lambda x: movie_ids_dict[x])

num_ratings = train_ratings.shape[0]
train_ratings_np = np.concatenate(
    (np.array(train_ratings[USER_COLUMN], dtype=pd.Series).reshape(num_ratings, 1),
     np.array(train_ratings[ITEM_COLUMN], dtype=pd.Series).reshape(
         num_ratings, 1),
     np.array(train_ratings[RATING_COLUMN], dtype=pd.Series).reshape(num_ratings, 1)),
    axis=1)

num_test_ratings = test_ratings.shape[0]
test_ratings_np = np.concatenate(
    (np.array(test_ratings[USER_COLUMN], dtype=pd.Series).reshape(num_test_ratings, 1),
     np.array(test_ratings[ITEM_COLUMN], dtype=pd.Series).reshape(
         num_test_ratings, 1),
     np.array(test_ratings[RATING_COLUMN], dtype=pd.Series).reshape(num_test_ratings, 1)),
    axis=1)
test_ratings_np = test_ratings_np.astype(np.float64)
test_users, test_movies, y_test = get_separate_users_movies_ratings(test_ratings_np)

# declare save paths
date = datetime.today()
date_now = date.strftime('%Y-%m-%d_%H-%M')

SAVE_MODEL = f'{date_now}_NeuFM_compare_split'
CHECKPOINT_DIR = os.path.join(RESULT_DIR, 'checkpoints', SAVE_MODEL)
TENSORBOARD_DIR = os.path.join(RESULT_DIR, 'logs', 'tensorboard', SAVE_MODEL)
LOGS_DIR = os.path.join(RESULT_DIR, 'logs', 'neural_mf', f'{SAVE_MODEL}')
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(TENSORBOARD_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

# get model and params
params = get_params(num_users=num_users, num_items=num_movies)
model = create_model(params)

# split train to train and val data
x_train, y_train, x_val, y_val = split_data(train_ratings_np, val_split=False,
                                            train_size=params['train_size'])

# define optimizer and callbacks
optimizer = tf.keras.optimizers.Adam(lr=params["learning_rate"],
                                     beta_1=params["beta1"],
                                     beta_2=params["beta2"],
                                     epsilon=params["epsilon"])

model_filepath = os.path.join(CHECKPOINT_DIR, 'model_weights.ckpt')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_filepath,
                                                save_weights_only=True,
                                                monitor='val_loss',
                                                save_best_only=True,
                                                verbose=1,
                                                mode='auto')

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, update_freq='epoch')

model.compile(loss=params['loss'],
              metrics=[Precision(), Recall(), F1Score()],
              optimizer=optimizer)

print("Training model: ")
history = model.fit(x=x_train, y=y_train,
                    epochs=params['train_epochs'],
                    verbose=1, shuffle=True,
                    batch_size=params['batch_size'],
                    validation_data=(x_val, y_val),
                    callbacks=[checkpoint, early_stop, tensorboard])

print('Evaluating model on TEST data split: ')
model.load_weights(model_filepath)
names = model.metrics_names
loss_and_metrics = model.evaluate(x=[test_users, test_movies], y=y_test,
                                  batch_size=params['batch_size'], verbose=1)

for name, value in zip(names, loss_and_metrics):
    print('{0:s} : {1:1.4f}'.format(name, value))
