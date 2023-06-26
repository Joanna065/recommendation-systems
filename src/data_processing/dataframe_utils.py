import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklego.pandas_utils import log_step

logging.basicConfig(level=logging.INFO)


@log_step
def start_pipeline(df):
    return df.copy(deep=True)


@log_step(level=logging.DEBUG)
def reset_index(df) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    return df


@log_step(level=logging.DEBUG)
def drop_unnecessary_cols(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.drop(columns, axis=1)
    return df


@log_step(level=logging.DEBUG)
def drop_unnecessary_rows(df: pd.DataFrame, indices: List[int]) -> pd.DataFrame:
    df = df.drop(indices)
    return df


@log_step(level=logging.DEBUG)
def remove_nan(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.dropna(subset=columns)
    return df


@log_step(level=logging.DEBUG)
def remove_duplicates(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.drop_duplicates(subset=columns, keep='first')
    return df


@log_step(level=logging.DEBUG)
def rename_cols(df: pd.DataFrame, colmap_dict: Dict) -> pd.DataFrame:
    df = df.rename(columns=colmap_dict)
    return df


@log_step(level=logging.DEBUG)
def rearrange_cols(df: pd.DataFrame, cols_order: List[str]) -> pd.DataFrame:
    df = df[cols_order]
    return df


@log_step(level=logging.DEBUG)
def convert_to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')
    return df


@log_step(level=logging.DEBUG)
def sort_values(df: pd.DataFrame, sort_subset: List[str], ascending=True) -> pd.DataFrame:
    df = df.sort_values(by=sort_subset, ascending=ascending)
    return df


@log_step(level=logging.DEBUG)
def col_to_onehot(df: pd.DataFrame, colname, sep="|"):
    df = pd.concat([df.drop(colname, axis=1), df[colname].T.squeeze().str.get_dummies(sep=sep)],
                   axis=1)
    return df


@log_step(level=logging.DEBUG)
def insert_nan(df: pd.DataFrame, substitute_val: Union[str, List[str]]) -> pd.DataFrame:
    return df.replace(substitute_val, np.NaN)


@log_step(level=logging.DEBUG)
def expand_column(df: pd.DataFrame, keep_cols: List[str], expand_col: str,
                  split_sep="|") -> pd.DataFrame:
    df = pd.concat([df[keep_cols], df[expand_col].str.split(split_sep, expand=True)], axis=1)
    return df


@log_step(level=logging.DEBUG)
def unpivot_dataframe(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    df = df.melt(id_vars=keep_cols)
    return df


@log_step(level=logging.DEBUG)
def fill_nan(df: pd.DataFrame, fill_val: object) -> pd.DataFrame:
    df = df.fillna(value=fill_val)
    return df


@log_step(level=logging.DEBUG)
def string_to_lowercase(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].map(lambda x: x.lower() if isinstance(x, str) else x)
    return df


@log_step(level=logging.DEBUG)
def merge_tables(df_left: pd.DataFrame, df_right: pd.DataFrame, left_on: str,
                 right_on: str) -> pd.DataFrame:
    df = pd.merge(df_left, df_right, left_on=left_on, right_on=right_on)
    return df


@log_step(level=logging.DEBUG)
def filter_greater_than_numeric(df: pd.DataFrame, numeric_col: str,
                                filter_val: int) -> pd.DataFrame:
    df = df[df[numeric_col] > filter_val]
    return df


@log_step(level=logging.DEBUG)
def filter_in_list(df: pd.DataFrame, col: str, filter_list: List) -> pd.DataFrame:
    df = df[df[col].isin(filter_list)]
    return df


@log_step(level=logging.DEBUG)
def filter_equal_value(df: pd.DataFrame, col: str, value: str) -> pd.DataFrame:
    df = df[df[col] == value]
    return df


@log_step(level=logging.DEBUG)
def prepare_summary_table(df: pd.DataFrame, group_cols: List[str], aggr_col: str,
                          col_1: str, col_2=None) -> pd.DataFrame:
    summary_df = pd.DataFrame()
    summary_df[col_1] = df.groupby(by=group_cols)[aggr_col].count()
    if col_2 is not None:
        summary_df[col_2] = df.groupby(by=group_cols)[aggr_col].mean()
    summary_df = summary_df.reset_index()
    return summary_df


@log_step(level=logging.DEBUG)
def prepare_unique_val_count_table(df: pd.DataFrame, group_col: str, aggr_col: str,
                                   series_name: str) -> pd.DataFrame:
    df = df.groupby([group_col])[aggr_col].value_counts().rename_axis(
        [group_col, aggr_col]).reset_index(name=series_name)
    return df


def show_uniq_vals(df):
    for col in df:
        print(f'{col} - unique values: {np.unique(df[col].dropna().values).shape}, NaN values: '
              f'{df[col].isna().any()}')
