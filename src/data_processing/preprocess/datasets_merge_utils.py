import logging
from typing import List

import pandas as pd
from sklego.pandas_utils import log_step

logging.basicConfig(level=logging.DEBUG)


@log_step
def prepare_movie_tag_occurrences_table(df: pd.DataFrame, group_col: str, dict_cols: List[str],
                                        reset_idx_name: str) -> pd.DataFrame:
    df = df.groupby([group_col])[dict_cols].apply(lambda g: {a: b for a, b in g.values})
    df = df.reset_index(name=reset_idx_name)
    return df


@log_step
def prepare_movie_tags_table(df: pd.DataFrame, group_cols: List[str], aggr_col: str,
                             count_col: str, list_col: str) -> pd.DataFrame:
    summary_df = pd.DataFrame()
    summary_df[list_col] = df.groupby(by=group_cols)[aggr_col].apply(list)
    summary_df[count_col] = df.groupby(by=group_cols)[aggr_col].count()
    summary_df = summary_df.reset_index()
    return summary_df
