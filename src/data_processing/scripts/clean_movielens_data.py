import logging
import os
from pathlib import Path

import pandas as pd

from src.data_processing.dataframe_utils import (drop_unnecessary_cols, remove_nan, reset_index,
                                                 start_pipeline, string_to_lowercase)
from src.data_processing.preprocess.movielens_data_clean import (clean_movies,
                                                                 prepare_movies_genres_df)
from src.settings import DATA_DIR

logging.basicConfig(level=logging.DEBUG)

# specify save clean data paths
CLEANED_MOVIELENS_PATH = os.path.join(DATA_DIR, 'processed', 'ml25m_cleaned')
Path(CLEANED_MOVIELENS_PATH).mkdir(parents=True, exist_ok=True)

MOVIES_FILENAME = 'movies.csv'
MOVIES_GENRES_FILENAME = 'movies_genres.csv'
RATINGS_FILENAME = 'ratings.csv'
LINKS_FILENAME = 'links.csv'
TAG_FILENAME = 'tags.csv'
TAG_GENOME_FILENAME = 'tags_genome.csv'
TAG_GENOME_SCORES_FILENAME = 'tags_genome_scores.csv'

# read raw movielens 25m data
movies = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'ml-25m', 'movies.csv'))
ratings = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'ml-25m', 'ratings.csv'))
links = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'ml-25m', 'links.csv'))
tags = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'ml-25m', 'tags.csv'))
tags_scores = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'ml-25m', 'genome-scores.csv'))
tags_genome = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'ml-25m', 'genome-tags.csv'))

# movies preprocess and movies_genres table extracting
movies_cleaned = (movies
                  .pipe(start_pipeline)
                  .pipe(clean_movies, sub_val='(no genres listed)', col='genres', drop_val='IMAX'))

movies_cleaned.to_csv(os.path.join(CLEANED_MOVIELENS_PATH, MOVIES_FILENAME), index=False)

movies_genres = (movies_cleaned
                 .pipe(start_pipeline)
                 .pipe(prepare_movies_genres_df, fillna='None', keep_cols=['movieId', 'title'],
                       expand_col='genres', rename_dict={"value": "genre"}, drop_cols=['variable'],
                       sort_cols=['movieId']))

movies_genres.to_csv(os.path.join(CLEANED_MOVIELENS_PATH, MOVIES_GENRES_FILENAME), index=False)

# ratings preprocess
ratings.to_csv(os.path.join(CLEANED_MOVIELENS_PATH, RATINGS_FILENAME), index=False)

# links preprocess
links_cleaned = (links
                 .pipe(start_pipeline)
                 .pipe(drop_unnecessary_cols, columns=['tmdbId'])
                 .pipe(reset_index))

links_cleaned.to_csv(os.path.join(CLEANED_MOVIELENS_PATH, LINKS_FILENAME), index=False)

# tags preprocess
tags_cleaned = (tags
                .pipe(start_pipeline)
                .pipe(remove_nan, columns=['tag'])
                .pipe(string_to_lowercase, col='tag')
                .pipe(reset_index))

tags_cleaned.to_csv(os.path.join(CLEANED_MOVIELENS_PATH, TAG_FILENAME), index=False)

# tags genome preprocess
tags_genome.to_csv(os.path.join(CLEANED_MOVIELENS_PATH, TAG_GENOME_FILENAME), index=False)

# tags genome scores preprocess
tags_scores.to_csv(os.path.join(CLEANED_MOVIELENS_PATH, TAG_GENOME_SCORES_FILENAME), index=False)
