import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs


class UserModel(tf.keras.Model):

    def __init__(self, ratings, timestamp_buckets):
        super().__init__()

        self.user_id_lookup = tf.keras.layers.StringLookup()
        self.user_id_lookup.adapt(ratings.map(lambda x: x["user_id"]))

        self.user_embedding = tf.keras.Sequential([
            self.user_id_lookup,
            tf.keras.layers.Embedding(self.user_id_lookup.vocabulary_size(), 32),
        ])
        self.timestamp_embedding = tf.keras.Sequential([
            tf.keras.layers.Discretization(timestamp_buckets.tolist()),
            tf.keras.layers.Embedding(len(timestamp_buckets) + 2, 32)
        ])
        self.normalized_timestamp = tf.keras.layers.Normalization(
            axis=None
        )
        self.normalized_timestamp.adapt(
            ratings.map(lambda x: x["timestamp"]).batch(128))

        self.user_occupation_text_vectorizer = tf.keras.layers.TextVectorization()
        self.user_occupation_text_vectorizer.adapt(
            ratings.map(lambda x: x["user_occupation_text"]))
        self.user_occupation_text_embedding = tf.keras.Sequential([
            self.user_occupation_text_vectorizer,
            tf.keras.layers.Embedding(
                self.user_occupation_text_vectorizer.vocabulary_size() + 2, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

    def call(self, inputs):
        # Take the input dictionary, pass it through each input layer,
        # and concatenate the result.
        res = tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.timestamp_embedding(inputs["timestamp"]),
            tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),
            tf.reshape(inputs["user_rating"], (-1, 1)),
            tf.reshape(tf.cast(inputs["user_occupation_label"], tf.float32), (-1, 1)),
            tf.reshape(tf.cast(inputs["raw_user_age"], tf.float32), (-1, 1)),
            # tf.reshape(tf.cast(inputs["movie_genres"], tf.float32), (-1, 1)),
            tf.reshape(tf.cast(inputs["user_gender"], tf.float32), (-1, 1)),
            self.user_occupation_text_embedding(inputs["user_occupation_text"]),
        ], axis=1)
        print("USER MODEL OUTPUT SHAPE: ", res.shape)
        return res


class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, layer_sizes, ratings, timestamp_buckets):
        """Model for encoding user queries.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        # We first use the user model for generating embeddings.
        print("CREATING USER MODEL...")
        self.embedding_model = UserModel(ratings, timestamp_buckets)
        print("USER MODEL CREATED...")

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class MovieModel(tf.keras.Model):

    def __init__(self, movies, unique_movie_titles):
        super().__init__()

        max_tokens = 10_000

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32)
        ])

        self.title_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens)

        self.title_text_embedding = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.title_vectorizer.adapt(movies)

    def call(self, titles):
        return tf.concat([
            self.title_embedding(titles),
            self.title_text_embedding(titles),
        ], axis=1)


class CandidateModel(tf.keras.Model):
    """Model for encoding movies."""

    def __init__(self, layer_sizes, movies, unique_movie_titles):
        """Model for encoding movies.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        self.embedding_model = MovieModel(movies, unique_movie_titles)

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class MovielensModel(tfrs.models.Model):

    def __init__(self, layer_sizes, movies, ratings, timestamp_buckets, unique_movie_titles):
        super().__init__()

        print("CREATING QUERY MODEL...")
        self.query_model = QueryModel(layer_sizes,  ratings, timestamp_buckets)
        print("CREATING CANDIDATE MODEL...")
        self.candidate_model = CandidateModel(layer_sizes, movies, unique_movie_titles)
        print("QUERY AND CANDIDATE MODELS CREATED...")

        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.candidate_model),
            ),
        )
        # The tasks.
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.candidate_model)
            )
        )

        # The loss weights.
        self.rating_weight = 0.5
        self.retrieval_weight = 0.5
        print("MODEL itself CREATED...")

    def compute_loss(self, features, training=False):
        # query_embeddings = self.query_model({
        #     "user_id": features["user_id"],
        #     "timestamp": features["timestamp"],
        #     "user_rating": features["user_rating"],
        #     "user_occupation_label": features["user_occupation_label"],
        #     "raw_user_age": features["raw_user_age"],
        #     #"movie_genres": features["movie_genres"],
        #     "user_gender": features["user_gender"],
        #     "user_occupation_text": features["user_occupation_text"],
        # })
        # movie_embeddings = self.candidate_model({
        #     "movie_title": features["movie_title"],
        #     #"movie_genres": features["movie_genres"],
        # })
        # return self.task(query_embeddings, movie_embeddings, compute_metrics=not training)

        ratings_popped = features.get("user_rating")

        user_embeddings, movie_emb, rating_predictions = self(features)

        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=ratings_popped,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, movie_emb)

        # And combine them using the loss weights.
        return (self.rating_weight * rating_loss
                + self.retrieval_weight * retrieval_loss)

    def call(self, features):
        query_embeddings = self.query_model({
            "user_id": features["user_id"],
            "timestamp": features["timestamp"],
            "user_rating": features["user_rating"],
            "user_occupation_label": features["user_occupation_label"],
            "raw_user_age": features["raw_user_age"],
            # "movie_genres": features["movie_genres"],
            "user_gender": features["user_gender"],
            "user_occupation_text": features["user_occupation_text"],
        })
        movie_embeddings = self.candidate_model(features["movie_title"])

        ratings_pred = self.rating_model(tf.concat([query_embeddings, movie_embeddings], axis=1))

        return query_embeddings, movie_embeddings, ratings_pred


class MODEL:
    def __init__(self):
        self.ratings = tfds.load("movielens/100k-ratings", split="train")
        self.movies = tfds.load("movielens/100k-movies", split="train")

        self.ratings = self.ratings.map(lambda x: {
            "user_id": x["user_id"],
            "timestamp": x["timestamp"],
            "user_rating": x["user_rating"],
            "user_occupation_label": x["user_occupation_label"],
            "raw_user_age": x["raw_user_age"],
            # "movie_genres": x["movie_genres"],
            "user_gender": x["user_gender"],
            "user_occupation_text": x["user_occupation_text"],
            "movie_title": x["movie_title"],
        })
        self.movies = self.movies.map(lambda x: x["movie_title"])

        self.max_timestamp = self.ratings.map(lambda x: x["timestamp"]).reduce(
            tf.cast(0, tf.int64), tf.maximum).numpy().max()
        self.min_timestamp = self.ratings.map(lambda x: x["timestamp"]).reduce(
            np.int64(1e9), tf.minimum).numpy().min()

        self.timestamp_buckets = np.linspace(
            self.min_timestamp, self.max_timestamp, num=1000)

        self.unique_movie_titles = np.unique(np.concatenate(list(self.movies.batch(1000))))

        self.unique_user_ids = np.unique(np.concatenate(list(self.ratings.map(lambda x:
                                                                              x["user_id"]).batch(128))))

        self.loaded_model = MovielensModel([32],
                                           self.movies,
                                           self.ratings,
                                           self.timestamp_buckets,
                                           self.unique_movie_titles)
        self.loaded_model.load_weights('WEIGHTS_2l')


