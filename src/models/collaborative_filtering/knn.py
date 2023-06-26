import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

log = logging.getLogger(__name__)


class KNNModel:
    def __init__(self, movies_path: str, ratings_path: str, ratings_threshold, metric: str):
        self.knn_metric = metric
        logging.info(f"Loading movies data: {movies_path}")
        movies_df = pd.read_csv(movies_path)
        self.movies_df = movies_df[['movieId', 'title']]
        logging.info(f"Loaded {len(self.movies_df)} movies")

        logging.info(f"Loading ratings data: {ratings_path}")
        ratings_df = pd.read_csv(ratings_path)[['movieId', 'userId', 'rating']]
        self.ratings_user_df = ratings_df.query(f'rating > {ratings_threshold}')
        self.df = pd.merge(ratings_df, self.movies_df, on='movieId')
        logging.info(f"Loaded {len(self.ratings_user_df)} ratings")

    def preprocess_data(self):
        combine_movie_rating = self.df.dropna(axis=0, subset=['movieId'])
        movie_ratingCount = (
            combine_movie_rating.groupby(by=['movieId'])['rating'].count().reset_index().rename(
                columns={'rating': 'totalRatingCount'})[['movieId', 'totalRatingCount']])

        rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount,
                                                                  left_on='movieId',
                                                                  right_on='movieId', how='left')

        self.movie_features_df = rating_with_totalRatingCount.pivot_table(index='movieId',
                                                                          columns='userId',
                                                                          values='rating').fillna(0)
        print(self.movie_features_df)

        movie_features_df_matrix = csr_matrix(self.movie_features_df.values)

        self.model_knn = NearestNeighbors(metric=self.knn_metric, algorithm='brute')
        self.model_knn.fit(movie_features_df_matrix)

        self.movieId_indices = pd.Series(self.movie_features_df.index)

        logging.info("Finished processing data")

    def _get_title_from_movieId(self, movieId):
        title = self.movies_df[self.movies_df.movieId == movieId].title.values
        assert len(title) == 1, f"Movie with id {movieId} doesn't exist!"
        return title[0]

    def _get_movie_idx_from_movieId(self, movieId):
        # gettin the index of the movie that matches the title
        index = self.movieId_indices[self.movieId_indices == movieId].index
        assert len(index) == 1, f"Movie with id {movieId} doesn't exist!"
        return index[0]

    def _get_movie_idx_from_title(self, title):
        # gettin the index of the movie that matches the title
        movieId = self.movies_df[self.movies_df.title == title].movieId.values
        assert len(movieId) == 1, f"Movie {title} doesn't exist!"
        return self._get_movie_idx_from_movieId(movieId[0])

    def _get_recommendations(self, indexes, top=10):
        features = [self.movie_features_df.iloc[idx, :].values.reshape(1, -1) for idx in indexes]
        features_matrix = np.concatenate(features)

        avg_vector = np.average(features_matrix, axis=0).reshape(1, -1)

        distances, indices = self.model_knn.kneighbors(avg_vector,
                                                       n_neighbors=min(len(indexes) + top, len(
                                                           self.movie_features_df.index)))

        # populating the list with the titles of the best 10 matching movies
        recommended_movies = []
        for i, distance in zip(indices.flatten(), distances.flatten()):
            if i not in indexes and len(recommended_movies) < top:
                movieId = self.movie_features_df.index[i]
                recommended_movies.append(
                    (self._get_title_from_movieId(movieId), movieId, distance))
        return recommended_movies

    def get_users_liked_movies(self, user_id):
        liked_movies = self.ratings_user_df[self.ratings_user_df.userId == user_id].movieId
        return self.movies_df[self.movies_df['movieId'].isin(liked_movies)].title

    def get_recommendations_for_user(self, user_id, top=10):
        liked_movies = self.ratings_user_df[self.ratings_user_df.userId == user_id].movieId
        indexes = [self._get_movie_idx_from_movieId(movie_id) for movie_id in liked_movies]
        return self._get_recommendations(indexes, top)

    def get_recommendations_for_movie(self, title, top=10):
        return self.get_recommendations_for_movies([title], top)

    def get_recommendations_for_movies(self, titles, top=10):
        indexes = [self._get_movie_idx_from_title(title) for title in titles]
        return self._get_recommendations(indexes, top)

    def get_recommendations_for_movie_from_id(self, id, top=10):
        return self.get_recommendations_for_movies_from_id([id], top)

    def get_recommendations_for_movies_from_id(self, ids, top=10):
        indexes = [self._get_movie_idx_from_movieId(movie_id) for movie_id in ids]
        return self._get_recommendations(indexes, top)
