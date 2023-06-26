import numpy as np
from sklearn.model_selection import train_test_split


def split_data(ratings: np.ndarray, val_split=False, train_size=0.8, stratify: np.ndarray = None):
    ratings = ratings.astype(np.float64)

    ratings_train, ratings_test = train_test_split(ratings, shuffle=True,
                                                   train_size=train_size,
                                                   stratify=stratify)

    users_train, movies_train, ratings_train = get_separate_users_movies_ratings(ratings_train)

    if val_split:
        ratings_val, ratings_test = train_test_split(ratings_test, shuffle=True, train_size=0.5)
        users_val, movies_val, ratings_val = get_separate_users_movies_ratings(ratings_val)
        users_test, movies_test, ratings_test = get_separate_users_movies_ratings(ratings_test)

        return (users_train, movies_train), ratings_train, \
               (users_val, movies_val), ratings_val, \
               (users_test, movies_test), ratings_test

    users_test, movies_test, ratings_test = get_separate_users_movies_ratings(ratings_test)

    return (users_train, movies_train), ratings_train, (users_test, movies_test), ratings_test


def get_separate_users_movies_ratings(rate_array: np.ndarray):
    assert rate_array.shape[1] == 3
    users = rate_array[:, 0].T.astype(np.int32).reshape(-1, )
    movies = rate_array[:, 1].T.astype(np.int32).reshape(-1, )
    ratings = rate_array[:, 2].T
    return users, movies, ratings
