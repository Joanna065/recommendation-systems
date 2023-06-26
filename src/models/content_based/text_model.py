import logging
import multiprocessing

import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

log = logging.getLogger(__name__)


class TextModel:
    def __init__(self, ratings_path: str, movies_path: str, rate_treshold=3.5):
        logging.info(f"Loading movies data: {movies_path}")
        movies_df = pd.read_csv(movies_path)
        movies_df = movies_df.set_index('movieId')
        self.movies_df = movies_df[[
            'original_title', 'genres', 'directors', 'actors', 'storyline', 'plot_keywords']]
        logging.info(f"Loaded {len(self.movies_df)} movies")

        logging.info(f"Loading ratings data: {ratings_path}")
        self.ratings_df = pd.read_csv(ratings_path).query(f'rating > {rate_treshold}')
        logging.info(f"Loaded {len(self.ratings_df)} positive ratings")

    def preprocess_data(self):
        self.movies_dict = self._getMoviesDict()
        self.movies_transformed = pd.DataFrame(self.movies_dict)

        vocab_1 = [(act + dir).split() for act, dir in
                   zip(self.movies_transformed.actors, self.movies_transformed.directors)]
        self.w2_model_actors = self._getWord2Vec(vocab_1)
        vocab_2 = [w.split() for w in self.movies_transformed.words]
        self.w2_model_storyline = self._getWord2Vec(vocab_2)

        count_gen = CountVectorizer()
        self.count_matrix_gen = count_gen.fit_transform(self.movies_transformed.genres)

        movies_vectors = self._getVectors()
        logging.info("Calculating cosine matrix")
        # generating the cosine similarity matrix
        self.cosine_sim = cosine_similarity(movies_vectors, movies_vectors)
        logging.info(f"Cosine matrix size {self.cosine_sim.shape}")

        self.title_indices = pd.Series(self.movies_transformed.title)
        self.id_indices = pd.Series(self.movies_transformed.id)
        logging.info("Finished processing")

    def _getMoviesDict(self):
        movies_dict = []
        for index, row in tqdm(self.movies_df.iterrows(), desc="Processing tags",
                               total=len(self.movies_df.index)):
            movie = {
                'id': index,
                'title': row.original_title
            }

            metadata = {
                'directors': '',
                'actors': '',
                'plot_keywords': '',
                'genres': ''
            }

            for key in metadata.keys():
                if not pd.isna(row[key]):
                    tmp_val = row[key]
                    tmp_val = tmp_val.split('|')
                    tmp_val = [val.replace(' ', '') for val in tmp_val]
                    tmp_val = ' '.join(tmp_val)
                    metadata[key] = tmp_val

            storyline = ''
            if not pd.isna(row.storyline):
                storyline = row.storyline

            full_text = storyline

            tokens = word_tokenize(full_text.lower())

            words = [word for word in tokens if word.isalpha()]

            stop_words = set(stopwords.words('english'))
            words = ' '.join([w for w in words if w not in stop_words])

            movie['words'] = metadata['plot_keywords'] + words
            movie['directors'] = metadata['directors']
            movie['actors'] = metadata['actors']
            movie['genres'] = metadata['genres']
            movie['full_text'] = metadata['plot_keywords'] + words + metadata['directors'] + \
                metadata['actors'] + metadata['genres']

            movies_dict.append(movie)
        return movies_dict

    def _getVectors(self):
        movies_vectors = []
        genres_size = self.count_matrix_gen.shape[-1]
        for i, movie in tqdm(enumerate(self.movies_dict), desc="Processing movies"):
            try:
                temp_vectors = []
                for word in movie['words'].split():
                    temp_vectors.append(self.w2_model_storyline.wv[word])
                temp_vectors = np.stack(temp_vectors)
                avg_vector_plot = np.average(temp_vectors, axis=0).reshape((300, 1))
            except Exception:
                avg_vector_plot = np.zeros((300, 1))

            try:
                temp_vectors = []
                for word in (movie['actors'] + movie['directors']).split():
                    temp_vectors.append(self.w2_model_actors.wv[word])
                temp_vectors = np.stack(temp_vectors)
                avg_vector_act_dir = np.average(temp_vectors, axis=0).reshape((300, 1))
            except Exception:
                avg_vector_act_dir = np.zeros((300, 1))

            vec_genre = self.count_matrix_gen[i].todense().reshape((genres_size, 1))
            final_vec = np.concatenate((vec_genre, avg_vector_plot, avg_vector_act_dir)).flatten()
            movies_vectors.append(final_vec)
        movies_vectors = np.stack(movies_vectors)
        return movies_vectors

    def _getWord2Vec(self, vocab):
        cores = multiprocessing.cpu_count()
        w2v_model = Word2Vec(min_count=1,
                             window=2,
                             size=300,
                             sample=6e-5,
                             alpha=0.03,
                             min_alpha=0.0007,
                             negative=20,
                             workers=cores - 1)

        w2v_model.build_vocab(vocab, progress_per=10000)
        w2v_model.train(vocab, total_examples=w2v_model.corpus_count, epochs=10, report_delay=1)
        w2v_model.init_sims(replace=True)
        return w2v_model

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
        movies = self.movies_transformed[self.movies_transformed['title'] == title]
        assert len(movies) == 1, f"Movie {title} doesn't exist!"
        movie = movies.iloc[0]
        return movie.full_text
