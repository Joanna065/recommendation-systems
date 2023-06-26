import logging

import numpy as np
import pandas as pd

from src.data_processing.dataframe_utils import prepare_summary_table, reset_index, start_pipeline

log = logging.getLogger(__name__)


def select_user_ids(ratings: pd.DataFrame, users_amount: int, user_colname: str,
                    min_rate_amount=400, min_rate_avg=2.75, max_rate_avg=4.75) -> np.ndarray:
    rating_summary = (ratings
                      .pipe(start_pipeline)
                      .pipe(prepare_summary_table, group_cols=[user_colname], aggr_col='rating',
                            col_1='rate_amount', col_2='rate_avg')
                      .pipe(reset_index))

    rates_above = rating_summary[rating_summary['rate_amount'] >= min_rate_amount]
    rate_avg_between = rates_above[rates_above['rate_avg'] >= min_rate_avg]
    rate_avg_between = rate_avg_between[rate_avg_between['rate_avg'] <= max_rate_avg]

    log.info(f'Selecting users from possible {len(rate_avg_between)} users in total')

    user_ids = np.random.choice(rate_avg_between[user_colname], size=users_amount, replace=False)

    return user_ids


def remove_user_rates(ratings: pd.DataFrame, user_ids: np.ndarray, user_colname: str,
                      drop_rate=0.2):
    selected_rates = ratings[ratings[user_colname].isin(user_ids)]
    drop_rates = selected_rates.sample(frac=drop_rate)
    new_ratings = ratings.drop(drop_rates.index)

    return new_ratings, drop_rates
