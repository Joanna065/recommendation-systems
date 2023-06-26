import io
import logging
import os
import pickle
import re
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Dict, Tuple

import pandas as pd
from funk_svd import SVD
from sklearn.metrics import mean_absolute_error

from src.datasets.funk_svd_dataset import split_dataframe

log = logging.getLogger(__name__)


def run_funk_svd_experiment(ratings: pd.DataFrame, train_params: Dict, train_split=0.8, val=False,
                            test_with_val=True, seed=25):
    log.info("Run experiment with params {}".format(train_params))
    logs_dict = defaultdict()
    test_df = None

    log.info("Splitting data")
    if val:
        if test_with_val:
            train_df, val_df, test_df = split_dataframe(ratings, train_factor=train_split,
                                                        val_split=True, rand_seed=seed)
        else:
            train_df, val_df = split_dataframe(ratings, train_factor=train_split, rand_seed=seed)

        model, model_dump_dict, trained_epochs, logs_dict = _train_with_validation(train_params,
                                                                                   logs_dict,
                                                                                   train_df, val_df)
    else:
        train_df, test_df = split_dataframe(ratings, train_factor=train_split, rand_seed=seed)
        model, model_dump_dict, trained_epochs = _train_without_validation(train_params, train_df)

    log.info(f'Trained epochs: {trained_epochs}')

    # predict on test data and calculate error
    if test_df is not None:
        pred = model.predict(test_df)
        test_mae = mean_absolute_error(test_df["rating"], pred)
        print(f'Test MAE loss: {test_mae}')
        logs_dict['test_mae'] = test_mae

    return model, model_dump_dict, logs_dict, trained_epochs


def train_funk_svd(train_df, latent_factors=100, lr=.005, regularization=0.02, epochs=20,
                   min_rate=0.5, max_rate=5.0, val_df=None, shuffle=False):
    model = SVD(learning_rate=lr, regularization=regularization, n_epochs=epochs,
                n_factors=latent_factors, min_rating=min_rate, max_rating=max_rate)

    log.info("Starting training...")
    if val_df is not None:
        model.fit(X=train_df, X_val=val_df, early_stopping=True, shuffle=shuffle)
    else:
        model.fit(X=train_df, shuffle=shuffle)

    trained_pu, trained_qi = model.pu, model.qi
    trained_bu, trained_bi = model.bu, model.bi
    funk_model_dict = {'pu': trained_pu, 'qi': trained_qi, 'bu': trained_bu, 'bi': trained_bi,
                       'user_dict': model.user_dict, 'item_dict': model.item_dict,
                       'global_mean': model.global_mean}

    return model, funk_model_dict


def _train_with_validation(train_params: Dict, logs_dict: Dict, train_df: pd.DataFrame,
                           val_df: pd.DataFrame):
    with io.StringIO() as buf, redirect_stdout(buf):
        model, model_dump_dict = train_funk_svd(train_df, **train_params, val_df=val_df)
        train_log = buf.getvalue()

    trained_epochs, max_epochs, val_loss_list, val_rmse_list, val_mae_list = \
        _parse_train_log_stdout(train_log, val=True)
    logs_dict['val_loss'] = val_loss_list
    logs_dict['val_rmse'] = val_rmse_list
    logs_dict['val_mae'] = val_mae_list
    print(f'Validation MAE loss: {val_mae_list}')

    return model, model_dump_dict, trained_epochs, logs_dict


def _train_without_validation(train_params: Dict, train_df: pd.DataFrame):
    with io.StringIO() as buf, redirect_stdout(buf):
        model, model_dump_dict = train_funk_svd(train_df, **train_params)
        train_log = buf.getvalue()

    trained_epochs, max_epochs = _parse_train_log_stdout(train_log, val=False)

    return model, model_dump_dict, trained_epochs


def _parse_train_log_stdout(train_log: str, val: bool) -> Tuple:
    epochs_list = re.findall(r'\d+/\d+', train_log)
    epochs, max_epochs = epochs_list[-1].split(sep='/')
    trained_epochs, max_epochs = int(epochs), int(max_epochs)

    if val:
        val_loss_list = re.findall(r'val_loss:\s\d.\d+', train_log)
        val_loss_list = [float(loss.strip('val_loss: ')) for loss in val_loss_list]

        val_rmse_list = re.findall(r'val_rmse:\s\d.\d+', train_log)
        val_rmse_list = [float(loss.strip('val_rmse: ')) for loss in val_rmse_list]

        val_mae_list = re.findall(r'val_mae:\s\d.\d+', train_log)
        val_mae_list = [float(loss.strip('val_mae: ')) for loss in val_mae_list]

        return trained_epochs, max_epochs, val_loss_list, val_rmse_list, val_mae_list

    return trained_epochs, max_epochs


def save_model(save_dir: str, filename: str, dump_dict):
    log.info("Saving model dump file")
    with open(os.path.join(save_dir, filename + '.pkl'), 'wb') as f:
        pickle.dump(dump_dict, f)


def save_logs(save_dir: str, filename: str, logs_dict: Dict) -> None:
    log.info("Saving logs - validation loss and metrics")
    with open(os.path.join(save_dir, filename + '.pkl'), 'wb') as f:
        pickle.dump(logs_dict, f)


def load_model(model_weights: Dict):
    model = SVD()
    model.pu = model_weights['pu']
    model.qi = model_weights['qi']
    model.bu = model_weights['bu']
    model.bi = model_weights['bi']
    model.global_mean = model_weights['global_mean']
    model.user_dict = model_weights['user_dict']
    model.item_dict = model_weights['item_dict']

    return model
