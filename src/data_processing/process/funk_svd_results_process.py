import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklego.pandas_utils import log_step


@log_step(level=logging.DEBUG)
def prepare_logs_frame(train_logs: Dict) -> pd.DataFrame:
    train_df = pd.DataFrame()
    train_df['epoch'] = np.arange(1, len(train_logs['val_loss']) + 1, 1)
    train_df['val_loss'] = train_logs['val_loss']
    train_df['val_rmse'] = train_logs['val_rmse']
    train_df['val_mae'] = train_logs['val_mae']
    train_df.reset_index(drop=True)
    return train_df


@log_step(level=logging.DEBUG)
def build_spearman_drop_frame(results_dict: Dict, drop_rate: float):
    spearman_results = [spearman for spearman, p_val in
                        results_dict[f'drop_{drop_rate}']['spearman']]
    spearman_std = np.std(np.abs(spearman_results))
    spearman_mean = np.mean(np.abs(spearman_results))

    df = pd.DataFrame()
    df['spearman_correlation'] = spearman_results
    df['drop_rate'] = [f'{drop_rate}, mean = {spearman_mean:.4f} '
                       f'± {spearman_std:.4f}'] * len(spearman_results)

    return df


@log_step(level=logging.DEBUG)
def build_true_pred_rate_frame(results_dict: Dict, drop_rate: float):
    true_rates = results_dict[f'drop_{drop_rate}']['true_rates']
    pred_rates = results_dict[f'drop_{drop_rate}']['pred_rates']

    df = pd.DataFrame()
    df['true_rate'] = true_rates
    df['pred_rate'] = pred_rates
    df['drop_rate'] = [f'{drop_rate}'] * len(true_rates)

    return df
