import ast
import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklego.pandas_utils import log_step

from src.data_processing.dataframe_utils import (convert_to_numeric, drop_unnecessary_cols,
                                                 drop_unnecessary_rows, insert_nan, rearrange_cols,
                                                 remove_duplicates, remove_nan, rename_cols,
                                                 reset_index, sort_values)

logging.basicConfig(level=logging.DEBUG)


@log_step
def clean_movie_metadata(dataframe: pd.DataFrame, drop_cols: List[str], cols_to_numeric: List[str],
                         drop_nan_cols: List[str], rename_dict: Dict,
                         sort_cols: List[str], order_cols: List[str],
                         change_to_nans: List[str]) -> pd.DataFrame:
    return (dataframe
            .pipe(drop_unnecessary_cols, drop_cols)
            .pipe(fix_imdbid)
            .pipe(convert_to_separated_string, colname='genres')
            .pipe(insert_nan, change_to_nans)
            .pipe(convert_to_numeric, cols_to_numeric)
            .pipe(remove_nan, drop_nan_cols)
            .pipe(remove_duplicates, drop_nan_cols)
            .pipe(rename_cols, rename_dict)
            .pipe(sort_values, sort_cols)
            .pipe(rearrange_cols, order_cols)
            .pipe(reset_index))


@log_step
def clean_movie_keywords(dataframe: pd.DataFrame, drop_rows_idx: List[int],
                         duplicated_cols: List[str],
                         rename_dict: Dict, sort_cols: List[str]) -> pd.DataFrame:
    return (dataframe
            .pipe(drop_unnecessary_rows, drop_rows_idx)
            .pipe(remove_duplicates, duplicated_cols)
            .pipe(rename_cols, rename_dict)
            .pipe(sort_values, sort_cols)
            .pipe(reset_index))


@log_step
def clean_credits_data(dataframe: pd.DataFrame, duplicated_cols: List[str], rename_dict: Dict,
                       sort_cols: List[str], order_cols: List[str]) -> pd.DataFrame:
    return (dataframe
            .pipe(remove_duplicates, duplicated_cols)
            .pipe(rename_cols, rename_dict)
            .pipe(sort_values, sort_cols)
            .pipe(rearrange_cols, order_cols)
            .pipe(reset_index))


@log_step
def prepare_keywords_table(movie_keywords_df: pd.DataFrame) -> pd.DataFrame:
    keywords_list = [ast.literal_eval(line) for line in movie_keywords_df.keywords.values]
    keys_dict = [item for sublist in keywords_list for item in sublist]
    keywords = [(keyword['id'], keyword['name']) for keyword in keys_dict]
    keywords = pd.DataFrame(keywords, columns=['id', 'keyword']).drop_duplicates()
    return keywords


@log_step(level=logging.DEBUG)
def fix_imdbid(df: pd.DataFrame) -> pd.DataFrame:
    df.imdb_id = df.imdb_id.apply(lambda x: str(x).replace('tt', ''))
    return df


@log_step(level=logging.DEBUG)
def convert_to_separated_string(df: pd.DataFrame, colname: str, key='name',
                                sep="|") -> pd.DataFrame:
    df[colname] = df[colname].apply(
        lambda x: extract_values_to_string(ast.literal_eval(x), key, sep))
    return df


def extract_values_to_string(list_dict: List[Dict], key='name', sep="|") -> Union[str, float]:
    values = []
    for dictionary in list_dict:
        if key in dictionary:
            if dictionary[key]:
                values.append(dictionary[key])
    return np.nan if not values else sep.join(values)


def extract_crew_info(crew: List[Dict], key='name', sep="|") -> \
        Tuple[Union[str, float], Union[str, float]]:
    directors = []
    writers = []
    for crew_dict in crew:
        if 'job' in crew_dict and crew_dict['job'] == 'Director':
            if crew_dict[key]:
                directors.append(crew_dict[key])
        if 'department' in crew_dict and crew_dict['department'] == 'Writing':
            if crew_dict[key]:
                writers.append(crew_dict[key])
    directors = np.nan if not directors else sep.join(directors)
    writers = np.nan if not writers else sep.join(writers)
    return directors, writers


@log_step(level=logging.DEBUG)
def prepare_cast_crew_table(df: pd.DataFrame) -> pd.DataFrame:
    movies_info = []
    for index, row in df.iterrows():
        cast = ast.literal_eval(row['cast'])
        crew = ast.literal_eval(row['crew'])
        kaggle_id = row['kaggle_id']

        directors, writers = extract_crew_info(crew)
        actors = extract_values_to_string(cast, key='name', sep="|")
        movies_info.append((kaggle_id, directors, writers, actors))

    movies_table = pd.DataFrame(movies_info,
                                columns=['kaggle_id', 'directors', 'writers', 'actors'])
    return movies_table


@log_step(level=logging.DEBUG)
def merge_movies_metadata_keywords(cast_crew_movies: pd.DataFrame, metadata: pd.DataFrame,
                                   movie_keywords: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata[
        ['kaggle_id', 'imdb_id', 'title', 'original_title', 'release_date', 'genres', 'overview',
         'tagline']].copy(deep=True)
    df = cast_crew_movies.merge(metadata, on='kaggle_id')
    df = df.merge(movie_keywords, on='kaggle_id', how='left')
    return df


@log_step
def prepare_main_movie_table(dataframe: pd.DataFrame, metadata: pd.DataFrame,
                             keywords: pd.DataFrame, rename_dict: Dict,
                             cols_order: List[str]) -> pd.DataFrame:
    return (dataframe
            .pipe(merge_movies_metadata_keywords, metadata, keywords)
            .pipe(rename_cols, rename_dict)
            .pipe(rearrange_cols, cols_order)
            .pipe(reset_index))
