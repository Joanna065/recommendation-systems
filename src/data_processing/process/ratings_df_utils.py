import logging
from typing import List

import numpy as np
import pandas as pd
from sklego.pandas_utils import log_step

from src.data_processing.dataframe_utils import (drop_unnecessary_cols, expand_column,
                                                 filter_equal_value, merge_tables, remove_nan,
                                                 rename_cols, reset_index, sort_values,
                                                 start_pipeline, unpivot_dataframe)


def discretize_pred_rates(rates: np.ndarray):
    return np.round(rates * 2) / 2


@log_step(level=logging.DEBUG)
def prepare_user_unrated_movies_table(ratings: pd.DataFrame, user_id: int) -> pd.DataFrame:
    movies_list = ratings.movieId.unique()
    user_rated_df = ratings[ratings['userId'] == user_id]
    movies_rated_ids = user_rated_df.movieId.unique()
    movies_unrated_ids = [movie_id for movie_id in movies_list if movie_id not in movies_rated_ids]

    user_unrated_df = pd.DataFrame()
    user_unrated_df['u_id'] = [user_id] * len(movies_unrated_ids)
    user_unrated_df['i_id'] = movies_unrated_ids
    user_unrated_df.reset_index(drop=True, inplace=True)
    return user_unrated_df


@log_step(level=logging.DEBUG)
def prepare_user_rated_movies_table(ratings: pd.DataFrame, user_id: int) -> pd.DataFrame:
    user_rated_df = (ratings
                     .pipe(start_pipeline)
                     .pipe(filter_equal_value, col='userId', value=user_id)
                     .pipe(rename_cols, colmap_dict={'userId': 'u_id', 'movieId': 'i_id'})
                     .pipe(reset_index))
    return user_rated_df


@log_step(level=logging.DEBUG)
def merge_user_rating_with_movies(usr_rate_df: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies[['movieId', 'title', 'release_date', 'genres', 'rate_amount', 'rate_average',
                     'plot_keywords', 'unique_tag_list', 'unique_tag_occurrences']]
    user_rated_df = usr_rate_df.rename(columns={'u_id': 'userId', 'i_id': 'movieId'})
    df = pd.merge(user_rated_df, movies, on='movieId')
    df.reset_index(drop=True, inplace=True)
    return df


@log_step(level=logging.DEBUG)
def prepare_user_rated_genres(df: pd.DataFrame) -> pd.DataFrame:
    df = (df
          .pipe(start_pipeline)
          .pipe(drop_unnecessary_cols, columns=['rate_amount', 'rate_average',
                                                'unique_tag_list',
                                                'unique_tag_occurrences'])
          .pipe(expand_column,
                keep_cols=['userId', 'movieId', 'rating', 'timestamp', 'title',
                           'release_date'], expand_col='genres')
          .pipe(unpivot_dataframe,
                keep_cols=['userId', 'movieId', 'rating', 'timestamp',
                           'title', 'release_date'])
          .pipe(sort_values, sort_subset=['movieId'])
          .pipe(drop_unnecessary_cols, columns=['variable'])
          .pipe(remove_nan, columns=['value'])
          .pipe(rename_cols, colmap_dict={'value': 'genre'})
          .pipe(reset_index))
    return df


@log_step(level=logging.DEBUG)
def prepare_user_rated_keywords(df: pd.DataFrame) -> pd.DataFrame:
    df = (df
          .pipe(start_pipeline)
          .pipe(drop_unnecessary_cols, columns=['rate_amount', 'rate_average',
                                                'unique_tag_list', 'genres',
                                                'unique_tag_occurrences'])
          .pipe(expand_column, keep_cols=['userId', 'movieId', 'rating', 'timestamp', 'title',
                                          'release_date'], expand_col='plot_keywords')
          .pipe(unpivot_dataframe, keep_cols=['userId', 'movieId', 'rating', 'timestamp',
                                              'title', 'release_date'])
          .pipe(sort_values, sort_subset=['movieId'])
          .pipe(drop_unnecessary_cols, columns=['variable'])
          .pipe(remove_nan, columns=['value'])
          .pipe(rename_cols, colmap_dict={'value': 'plot_keyword'})
          .pipe(reset_index))
    return df


@log_step(level=logging.DEBUG)
def prepare_user_predict_rated_movies(user_unrated_df: pd.DataFrame, movies: pd.DataFrame,
                                      predicted_rates: List[float]) -> pd.DataFrame:
    df = user_unrated_df.assign(predict_rate=predicted_rates)
    df = (df
          .pipe(rename_cols, colmap_dict={'u_id': 'userId', 'i_id': 'movieId'})
          .pipe(merge_tables,
                movies[['movieId', 'title', 'release_date', 'genres', 'plot_keywords']],
                left_on='movieId', right_on='movieId')
          .pipe(sort_values, sort_subset=['predict_rate'], ascending=False)
          .pipe(reset_index))
    return df
