import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklego.pandas_utils import log_step

from src.data_processing.dataframe_utils import (drop_unnecessary_cols, expand_column, fill_nan,
                                                 insert_nan, rename_cols, reset_index, sort_values,
                                                 unpivot_dataframe)

logging.basicConfig(level=logging.DEBUG)


@log_step
def clean_movies(dataframe: pd.DataFrame, sub_val: str, col: str, drop_val: str) -> pd.DataFrame:
    return (dataframe
            .pipe(insert_nan, sub_val)
            .pipe(remove_val_from_concat_string, col, drop_val)
            .pipe(reset_index))


def prepare_movies_genres_df(df: pd.DataFrame, fillna: object, keep_cols: List[str],
                             expand_col: str, rename_dict: Dict, drop_cols: List[str],
                             sort_cols: List[str]) -> pd.DataFrame:
    return (df
            .pipe(fill_nan, fillna)
            .pipe(expand_column, keep_cols, expand_col, split_sep="|")
            .pipe(unpivot_dataframe, keep_cols)
            .pipe(drop_unnecessary_cols, drop_cols)
            .pipe(rename_cols, rename_dict)
            .pipe(sort_values, sort_cols)
            .pipe(reset_index))


def filter_values(string: object, drop_val: str, sep="|") -> Union[str, float]:
    return sep.join(
        [val for val in str(string).split(sep) if
         val != drop_val]) if string != 'nan' else np.nan


@log_step(level=logging.DEBUG)
def remove_val_from_concat_string(df: pd.DataFrame, colname: str, value: str,
                                  sep="|") -> pd.DataFrame:
    df[colname] = df[colname].apply(lambda x: filter_values(str(x), value, sep=sep))
    return df
