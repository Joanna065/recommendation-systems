import logging
import os
from pathlib import Path

import pandas as pd

from src.settings import DATA_DIR
from src.utils.scrapper_filmweb import FilmwebScraper

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

MERGED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'merged_ml25m_kaggle')
# USER = "arktos8"
# USER = "LorethRex"
USER = "Nikki97"
USER_PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed', f'user_{USER}')
Path(USER_PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

# scrap user ratings from filmweb
scraper = FilmwebScraper()
scraper.login()
rated_movies = scraper.get_user_ratings(user_name=USER)
log.info(f'User scrapped movies amount: {len(rated_movies)}')

# create user rated movies dataframe
titles, years, ratings = [], [], []
for rate_movie in rated_movies:
    if rate_movie['rate'] != '':
        titles.append(rate_movie['origin_title'])
        years.append(rate_movie['year'])
        ratings.append(int(float(rate_movie['rate'])) / 2.)

filmweb_user_rated_movies = pd.DataFrame()
filmweb_user_rated_movies['original_title'] = titles
filmweb_user_rated_movies['year'] = years
filmweb_user_rated_movies['rating'] = ratings

# load merged ml25m and kaggle data
log.info('Loading merged ml25m and kaggle data...')
ratings = pd.read_csv(os.path.join(MERGED_DATA_PATH, 'ratings_merged.csv'))
movies = pd.read_csv(os.path.join(MERGED_DATA_PATH, 'movies_merged.csv'))

movies = movies[['movieId', 'original_title', 'release_date']]
movies['year'] = movies.release_date.apply(lambda x: str(x)[0:4])

# find intersected movies between dataset and user rated movies
intersect_films = filmweb_user_rated_movies.merge(movies, how='inner',
                                                  on=['original_title', 'year'])

log.info(f'Intersection of user and dataset: {len(intersect_films)} movies')

max_user_id = max(ratings.userId.values)
filmweb_user_id = max_user_id + 1

fmb_user_ratings = pd.DataFrame()
fmb_user_ratings['userId'] = [filmweb_user_id] * len(intersect_films)
fmb_user_ratings['movieId'] = intersect_films.movieId.values
fmb_user_ratings['rating'] = intersect_films.rating.values
fmb_user_ratings.reset_index(drop=True, inplace=True)

log.info("Saving to csv...")
fmb_user_ratings.to_csv(os.path.join(USER_PROCESSED_DATA_DIR, 'user_ratings_only.csv'),
                        index=False)

concat_rates = pd.concat([ratings[['userId', 'movieId', 'rating']], fmb_user_ratings])
concat_rates.to_csv(os.path.join(USER_PROCESSED_DATA_DIR, 'concat_ratings.csv'),
                    index=False)
