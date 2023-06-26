import datetime
import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklego.pandas_utils import log_step

logging.basicConfig(level=logging.INFO)


def date_parser(time):
    return datetime.datetime.fromtimestamp(float(time))


@log_step(level=logging.DEBUG)
def ratings_funk_process(csv_path, sep=',', with_timestamp=True) -> pd.DataFrame:
    dtype = {'u_id': np.uint32, 'i_id': np.uint32, 'rating': np.float64}

    if with_timestamp:
        names = ['u_id', 'i_id', 'rating', 'timestamp']
        df = pd.read_csv(csv_path, names=names, dtype=dtype, header=0,
                         sep=sep, parse_dates=['timestamp'],
                         date_parser=date_parser, engine='python')
        df.sort_values(by='timestamp', inplace=True)
    else:
        names = ['u_id', 'i_id', 'rating']
        df = pd.read_csv(csv_path, names=names, dtype=dtype, header=0,
                         sep=sep)
    df.reset_index(drop=True, inplace=True)
    return df


def split_dataframe(data: pd.DataFrame, train_factor=0.8, val_split=False, rand_seed=10) -> \
        Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    assert 0.6 < train_factor <= 9.0

    train_df = data.sample(frac=train_factor, random_state=rand_seed)
    if val_split:
        val_df = data.drop(train_df.index.tolist()).sample(frac=0.5, random_state=rand_seed + 1)
        test_df = data.drop(train_df.index.tolist()).drop(val_df.index.tolist())
        return train_df, val_df, test_df
    else:
        test_df = data.drop(train_df.index.tolist())

    return train_df, test_df
