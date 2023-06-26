import logging
import os
from pathlib import Path

import pandas as pd

from src.data_processing.dataframe_utils import (drop_unnecessary_cols, filter_greater_than_numeric,
                                                 filter_in_list, merge_tables,
                                                 prepare_summary_table,
                                                 prepare_unique_val_count_table, remove_nan,
                                                 reset_index, start_pipeline)
from src.data_processing.preprocess.datasets_merge_utils import (
    prepare_movie_tag_occurrences_table, prepare_movie_tags_table)
from src.settings import DATA_DIR

logging.basicConfig(level=logging.INFO)

# save merged data paths
MERGED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'merged_ml25m_kaggle')
Path(MERGED_DATA_PATH).mkdir(parents=True, exist_ok=True)

MOVIES_FILENAME = 'movies_merged.csv'
RATINGS_FILENAME = 'ratings_merged.csv'
LINKS_FILENAME = 'links_merged.csv'
TAGS_FILENAME = 'tags_merged.csv'
TAGS_GENOME_FILENAME = 'tags_genome_merged.csv'
TAG_GENOME_SCORES_FILENAME = 'tags_genome_scores_merged.csv'
MOVIES_METADATA_FILENAME = 'movies_metadata_merged.csv'
KEYWORDS_GENOME_FILENAME = 'keywords_genome_merged.csv'

# load movielens data
movies = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'ml25m_cleaned', 'movies.csv'))
movies_genres = pd.read_csv(
    os.path.join(DATA_DIR, 'processed', 'ml25m_cleaned', 'movies_genres.csv'))
ratings = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'ml25m_cleaned', 'ratings.csv'))
links = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'ml25m_cleaned', 'links.csv'))
tags = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'ml25m_cleaned', 'tags.csv'))
tags_genome = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'ml25m_cleaned', 'tags_genome.csv'))
tags_genome_scores = pd.read_csv(
    os.path.join(DATA_DIR, 'processed', 'ml25m_cleaned', 'tags_genome_scores.csv'))

# load kaggle movies data
movies_kaggle = pd.read_csv(
    os.path.join(DATA_DIR, 'processed', 'kaggle_movies_cleaned', 'movies_kaggle.csv'))
movies_metadata = pd.read_csv(
    os.path.join(DATA_DIR, 'processed', 'kaggle_movies_cleaned', 'movies_metadata_kaggle.csv'))
movie_credits = pd.read_csv(
    os.path.join(DATA_DIR, 'processed', 'kaggle_movies_cleaned', 'credits_kaggle.csv'))
keywords_genome = pd.read_csv(
    os.path.join(DATA_DIR, 'processed', 'kaggle_movies_cleaned', 'keywords_genome_kaggle.csv'))

# prepare movie-ratings summary table with rate_amount
movie_rating_summary = (ratings
                        .pipe(start_pipeline)
                        .pipe(prepare_summary_table, group_cols=['movieId'], aggr_col='rating',
                              col_1='rate_amount', col_2='rate_average')
                        .pipe(reset_index))

# filter movieIds to those having rate amount above 10 and merge with imdbId links value
movie_rating_above_10 = (movie_rating_summary
                         .pipe(start_pipeline)
                         .pipe(filter_greater_than_numeric, numeric_col='rate_amount',
                               filter_val=10)
                         .pipe(merge_tables, links, left_on='movieId', right_on='movieId'))

# merge movies with rate amount above 10 with movies from kaggle
movies_merged = (movie_rating_above_10
                 .pipe(start_pipeline)
                 .pipe(merge_tables, movies_kaggle, left_on='imdbId', right_on='imdb_id')
                 .pipe(drop_unnecessary_cols, columns=['imdb_id', 'writers', 'tagline'])
                 .pipe(reset_index))

# prepare movie-tag-tag amount table
movie_tags_count = (tags
                    .pipe(start_pipeline)
                    .pipe(prepare_unique_val_count_table, group_col='movieId',
                          aggr_col='tag', series_name='tag_count')
                    .pipe(reset_index))

# prepare movie unique tags occurrences table
movie_tag_occurs = (movie_tags_count
                    .pipe(start_pipeline)
                    .pipe(prepare_movie_tag_occurrences_table, group_col='movieId',
                          dict_cols=['tag', 'tag_count'], reset_idx_name='unique_tag_occurrences')
                    .pipe(reset_index))

movie_unique_tags_summary = (movie_tags_count
                             .pipe(start_pipeline)
                             .pipe(drop_unnecessary_cols, columns=['tag_count'])
                             .pipe(prepare_movie_tags_table, group_cols=['movieId'], aggr_col='tag',
                                   count_col='unique_tag_amount', list_col='unique_tag_list')
                             .pipe(merge_tables, movie_tag_occurs, left_on='movieId',
                                   right_on='movieId')
                             .pipe(reset_index))

# prepare movie users tags table summary
movie_users_tags_summary = (tags
                            .pipe(start_pipeline)
                            .pipe(drop_unnecessary_cols, columns=['timestamp'])
                            .pipe(prepare_movie_tags_table, group_cols=['movieId'], aggr_col='tag',
                                  count_col='users_tags_amount', list_col='users_tags_list')
                            .pipe(reset_index))

# merge movie_unique_tags and movie_user_tag
movie_tags_summary = (movie_unique_tags_summary
                      .pipe(start_pipeline)
                      .pipe(merge_tables, movie_users_tags_summary, left_on='movieId',
                            right_on='movieId')
                      .pipe(reset_index))

# merge movies wih tags summary table
movies_merged_tags = (movies_merged
                      .pipe(start_pipeline)
                      .pipe(merge_tables, movie_tags_summary, left_on='movieId', right_on='movieId')
                      .pipe(remove_nan, columns=['storyline', 'title'])
                      .pipe(reset_index))

movies_merged_tags.to_csv(os.path.join(MERGED_DATA_PATH, MOVIES_FILENAME), index=False)

# select movie ids and imdb ids after filtering
selected_movie_ids = movies_merged_tags.movieId.values
selected_imdb_ids = movies_merged_tags.imdbId.values

filtered_ratings = (ratings
                    .pipe(start_pipeline)
                    .pipe(filter_in_list, col='movieId', filter_list=selected_movie_ids)
                    .pipe(reset_index))

# prepare users_rating summary table and filter users who gave more than 150 rates
user_rating_summary = (filtered_ratings
                       .pipe(start_pipeline)
                       .pipe(prepare_summary_table, group_cols=['userId'], aggr_col='rating',
                             col_1='rate_amount', col_2='rate_average')
                       .pipe(filter_greater_than_numeric, numeric_col='rate_amount', filter_val=150)
                       .pipe(reset_index))

selected_user_ids = user_rating_summary.userId.values

filtered_ratings = (filtered_ratings
                    .pipe(filter_in_list, col='userId', filter_list=selected_user_ids)
                    .pipe(reset_index))

filtered_ratings.to_csv(os.path.join(MERGED_DATA_PATH, RATINGS_FILENAME), index=False)

# filter tables with selected movie ids
filtered_links = (links
                  .pipe(start_pipeline)
                  .pipe(filter_in_list, col='movieId', filter_list=selected_movie_ids)
                  .pipe(reset_index))

filtered_links.to_csv(os.path.join(MERGED_DATA_PATH, LINKS_FILENAME), index=False)

filtered_tags = (tags
                 .pipe(start_pipeline)
                 .pipe(filter_in_list, col='movieId', filter_list=selected_movie_ids)
                 .pipe(reset_index))

filtered_tags.to_csv(os.path.join(MERGED_DATA_PATH, TAGS_FILENAME), index=False)

tags_genome.to_csv(os.path.join(MERGED_DATA_PATH, TAGS_GENOME_FILENAME), index=False)

filtered_tags_genome_scores = (tags_genome_scores
                               .pipe(start_pipeline)
                               .pipe(filter_in_list, col='movieId', filter_list=selected_movie_ids)
                               .pipe(reset_index))

filtered_tags_genome_scores.to_csv(os.path.join(MERGED_DATA_PATH, TAG_GENOME_SCORES_FILENAME),
                                   index=False)

filtered_metadata = (movies_metadata
                     .pipe(start_pipeline)
                     .pipe(filter_in_list, col='imdb_id', filter_list=selected_imdb_ids)
                     .pipe(reset_index))

filtered_metadata.to_csv(os.path.join(MERGED_DATA_PATH, MOVIES_METADATA_FILENAME), index=False)

keywords_genome.to_csv(os.path.join(MERGED_DATA_PATH, KEYWORDS_GENOME_FILENAME), index=False)
