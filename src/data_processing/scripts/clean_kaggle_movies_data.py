import logging
import os
from pathlib import Path

import pandas as pd

from src.data_processing.dataframe_utils import (drop_unnecessary_cols, rename_cols, reset_index,
                                                 sort_values, start_pipeline)
from src.data_processing.preprocess.kaggle_data_clean import (clean_credits_data,
                                                              clean_movie_keywords,
                                                              clean_movie_metadata,
                                                              convert_to_separated_string,
                                                              prepare_cast_crew_table,
                                                              prepare_keywords_table,
                                                              prepare_main_movie_table)
from src.settings import DATA_DIR

logging.basicConfig(level=logging.DEBUG)

# save cleaned data paths specified
CLEANED_KAGGLE_PATH = os.path.join(DATA_DIR, 'processed', 'kaggle_movies_cleaned')
Path(CLEANED_KAGGLE_PATH).mkdir(parents=True, exist_ok=True)
LINKS_FILENAME = 'links_kaggle.csv'
MOVIES_METADATA_FILENAME = 'movies_metadata_kaggle.csv'
CREDITS_FILENAME = 'credits_kaggle.csv'
MOVIE_KEYWORDS_FILENAME = 'movie_keywords_kaggle.csv'
PLOT_KEYWORDS_FILENAME = 'keywords_genome_kaggle.csv'
MOVIES_FILENAME = 'movies_kaggle.csv'

# read raw kaggle movies data
movies_metadata = pd.read_csv(
    os.path.join(DATA_DIR, 'raw', 'kaggle-the-movies', 'movies_metadata.csv'), low_memory=False)
movie_credits = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'kaggle-the-movies', 'credits.csv'))
movie_keywords = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'kaggle-the-movies', 'keywords.csv'))
links_kaggle = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'kaggle-the-movies', 'links.csv'))

# movies metadata preprocess
drop_cols = ['video', 'poster_path', 'belongs_to_collection', 'popularity', 'homepage',
             'vote_count', 'vote_average', 'production_companies', 'production_countries']
cols_to_numeric = ['id', 'imdb_id']
drop_nan_cols = ['id', 'imdb_id']
rename_dict = {"id": "kaggle_id"}
sort_cols = ['kaggle_id']
order_cols = ['kaggle_id', 'imdb_id', 'title', 'original_title', 'genres', 'original_language',
              'spoken_languages', 'release_date', 'status', 'runtime', 'budget', 'revenue',
              'adult', 'overview', 'tagline']
replace_to_nan = ['No overview found.', 'No movie overview available.']

movie_metadata_cleaned = (movies_metadata
                          .pipe(start_pipeline)
                          .pipe(clean_movie_metadata, drop_cols, cols_to_numeric,
                                drop_nan_cols, rename_dict, sort_cols, order_cols, replace_to_nan))

movie_metadata_cleaned.to_csv(os.path.join(CLEANED_KAGGLE_PATH, MOVIES_METADATA_FILENAME),
                              index=False)

# links preprocess
links_cleaned = (links_kaggle
                 .pipe(start_pipeline)
                 .pipe(drop_unnecessary_cols, columns=['tmdbId'])
                 .pipe(reset_index))

links_cleaned.to_csv(os.path.join(CLEANED_KAGGLE_PATH, LINKS_FILENAME), index=False)

# movie keywords preprocess
drop_rows_idx = movie_keywords.index[movie_keywords.keywords == '[]']
duplicated_cols = ['id']

movie_keywords_cleaned = (movie_keywords
                          .pipe(start_pipeline)
                          .pipe(clean_movie_keywords, drop_rows_idx, duplicated_cols, rename_dict,
                                sort_cols))

# only keywords extracting from cleaned movie_keywords
plot_keywords_genome = (movie_keywords_cleaned
                        .pipe(start_pipeline)
                        .pipe(prepare_keywords_table)
                        .pipe(rename_cols,
                              colmap_dict={"id": "keyword_id", "keywords": "plot_keywords"})
                        .pipe(sort_values, sort_subset=['keyword_id'])
                        .pipe(reset_index))

plot_keywords_genome.to_csv(os.path.join(CLEANED_KAGGLE_PATH, PLOT_KEYWORDS_FILENAME), index=False)

movie_keywords_cleaned = (movie_keywords_cleaned
                          .pipe(convert_to_separated_string, colname='keywords'))

movie_keywords_cleaned.to_csv(os.path.join(CLEANED_KAGGLE_PATH, MOVIE_KEYWORDS_FILENAME),
                              index=False)

# credits preprocess
credits_cleaned = (movie_credits
                   .pipe(start_pipeline)
                   .pipe(clean_credits_data, duplicated_cols=['id'],
                         rename_dict={"id": "kaggle_id"}, sort_cols=['kaggle_id'],
                         order_cols=['kaggle_id', 'cast', 'crew']))

credits_cleaned.to_csv(os.path.join(CLEANED_KAGGLE_PATH, CREDITS_FILENAME), index=False)

# cast and crew extracting
cast_crew_movies = (credits_cleaned
                    .pipe(start_pipeline)
                    .pipe(prepare_cast_crew_table))

# prepare main movie kaggle table
rename_dict = {"overview": "storyline", "keywords": "plot_keywords"}
cols_order = ['kaggle_id', 'imdb_id', 'title', 'original_title', 'release_date', 'genres',
              'directors', 'writers', 'actors', 'storyline', 'tagline', "plot_keywords"]

main_movie_table = (cast_crew_movies
                    .pipe(start_pipeline)
                    .pipe(prepare_main_movie_table, movie_metadata_cleaned,
                          movie_keywords_cleaned, rename_dict, cols_order))

main_movie_table.to_csv(os.path.join(CLEANED_KAGGLE_PATH, MOVIES_FILENAME), index=False)
