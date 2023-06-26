import ast
import logging

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

log = logging.getLogger(__name__)


class TagModel:
    def __init__(self, ratings_path: str, movies_path: str, rate_treshold=3.5):
        logging.info(f"Loading movies data: {movies_path}")
        movies_df = pd.read_csv(movies_path)
        movies_df = movies_df.set_index('movieId')
        self.movies_df = movies_df[['original_title', 'unique_tag_list', 'unique_tag_occurrences']]
        logging.info(f"Loaded {len(self.movies_df)} movies")

        logging.info(f"Loading ratings data: {ratings_path}")
        self.ratings_df = pd.read_csv(ratings_path).query(f'rating > {rate_treshold}')
        logging.info(f"Loaded {len(self.ratings_df)} positive ratings")

    def preprocess_data(self):
        movies_dict = []

        for index, row in tqdm(self.movies_df.iterrows(), desc="Processing tags",
                               total=len(self.movies_df.index)):
            movie = {
                'id': index,
                'title': row.original_title
            }
            keywords = ast.literal_eval(row.unique_tag_list)
            movie['keywords'] = ' '.join([kw.replace(' ', '') for kw in keywords])
            movies_dict.append(movie)

        self.movies_transformed = pd.DataFrame(movies_dict)

        logging.info("Calculating cosine matrix")
        count = CountVectorizer()
        count_matrix = count.fit_transform(self.movies_transformed['keywords'])
        # generating the cosine similarity matrix
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        logging.info(f"Cosine matrix size {self.cosine_sim.shape}")

        self.id_indices = pd.Series(self.movies_transformed.id)
        self.title_indices = pd.Series(self.movies_transformed.title)
        logging.info("Finished processing")

    def _get_movie_idx_from_movieId(self, movieId):
        # gettin the index of the movie that matches the title
        index = self.id_indices[self.id_indices == movieId].index
        assert len(index) == 1, f"Movie {movieId} doesn't exist!"
        return index[0]

    def _get_movie_idx_from_title(self, title):
        # gettin the index of the movie that matches the title
        index = self.title_indices[self.title_indices == title].index
        assert len(index) == 1, f"Movie {title} doesn't exist!"
        return index[0]

    def _get_recommendations_series(self, idx):
        score_series = pd.Series(self.cosine_sim[idx]).sort_values(ascending=False)
        return score_series

    def _get_recommendations(self, indexes, top=10):
        series_list = [self._get_recommendations_series(idx) for idx in indexes]
        df = pd.concat(series_list, axis=1)
        df['series'] = 0
        titles_amount = len(indexes)
        for i in range(titles_amount):
            df['series'] += df[i] / titles_amount
        series = df['series'].drop(indexes).sort_values(ascending=False)
        top_10_indexes = list(series.iloc[:top].index)
        # populating the list with the titles of the best 10 matching movies
        recommended_movies = []
        titles = list(self.movies_transformed.title)
        ids = list(self.movies_transformed.id)
        for i, score in zip(top_10_indexes, list(series.iloc[:top])):
            recommended_movies.append((titles[i], ids[i], score))
        return recommended_movies

    def get_users_liked_movies(self, user_id):
        liked_movies = self.ratings_df[self.ratings_df.userId == user_id].movieId
        return self.movies_transformed[self.movies_transformed['id'].isin(liked_movies)].title

    def get_recommendations_for_user(self, user_id, top=10):
        liked_movies = self.ratings_df[self.ratings_df.userId == user_id].movieId
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

    def get_movie_tags(self, title):
        movies = self.movies_df[self.movies_df['original_title'] == title]
        assert len(movies) == 1, f"Movie {title} doesn't exist!"
        movie = movies.iloc[0]
        occurrences = ast.literal_eval(movie.unique_tag_occurrences)

        return occurrences
