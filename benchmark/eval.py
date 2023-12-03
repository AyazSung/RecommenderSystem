# Importing necessary libraries
import os  # Operating system module for interacting with the operating system
import tempfile  # Module to create temporary files and directories

import matplotlib.pyplot as plt  # Matplotlib library for creating visualizations

import numpy as np  # NumPy library for numerical operations
import tensorflow as tf  # TensorFlow, a machine learning framework

# TensorFlow Datasets (tfds) for easy access to various datasets
import tensorflow_datasets as tfds

# TensorFlow Recommenders (tfrs) for building recommendation models
import tensorflow_recommenders as tfrs

ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

# Mapping the 'ratings' dataset to extract specific features of interest
ratings = ratings.map(lambda x: {
    "user_id": x["user_id"],
    "timestamp": x["timestamp"],
    "user_rating": x["user_rating"],
    "user_occupation_label": x["user_occupation_label"],
    "raw_user_age": x["raw_user_age"],
    # Commented out for now: "movie_genres": x["movie_genres"],
    "user_gender": x["user_gender"],
    "user_occupation_text": x["user_occupation_text"],
    "movie_title": x["movie_title"],
})

# Mapping the 'movies' dataset to extract only the 'movie_title' information
movies = movies.map(lambda x: x["movie_title"])

max_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(
    tf.cast(0, tf.int64), tf.maximum).numpy().max()
min_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(
    np.int64(1e9), tf.minimum).numpy().min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000)

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))

unique_user_ids = np.unique(np.concatenate(list(ratings.map(lambda x:
                                                            x["user_id"]).batch(128))))

## SPLITTING DATA INTO TRAIN AND TEST SETS

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(2048)
cached_test = test.batch(4096).cache()


# Real-world recommender systems are often composed of two stages:
#
# The retrieval stage is responsible for selecting an initial set of hundreds of candidates from all possible candidates. The main objective of this model is to efficiently weed out all candidates that the user is not interested in. Because the retrieval model may be dealing with millions of candidates, it has to be computationally efficient.
# The ranking stage takes the outputs of the retrieval model and fine-tunes them to select the best possible handful of recommendations. Its task is to narrow down the set of items the user may be interested in to a shortlist of likely candidates.
#
#
# A query model computing the query representation (normally a fixed-dimensionality embedding vector) using query features.
# A candidate model computing the candidate representation (an equally-sized vector) using the candidate features
# The outputs of the two models are then multiplied together to give a query-candidate affinity score, with higher scores expressing a better match between the candidate and the query.


# Choosing the architecture of our model is a key part of modelling.
#
# Because we are building a two-tower retrieval model, we can build each tower separately and then combine them in the final model.

# The query tower
#
# The first step is to decide on the dimensionality of the query and candidate representations:
#
#
# embedding_dimension = 32
# Higher values will correspond to models that may be more accurate, but will also be slower to fit and more prone to overfitting.
#
# The second is to define the model itself. Here, we're going to use Keras preprocessing layers to first convert user ids to integers, and then convert those to user embeddings via an Embedding layer. Note that we use the list of unique user ids we computed earlier as a vocabulary:

class UserModel(tf.keras.Model):

    def __init__(self):
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
        return res


class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, layer_sizes):
        """Model for encoding user queries.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        # We first use the user model for generating embeddings.
        print("CREATING USER MODEL...")
        self.embedding_model = UserModel()
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


# The candidate tower
# We can do the same with the candidate tower.

class MovieModel(tf.keras.Model):

    def __init__(self):
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

    def __init__(self, layer_sizes):
        """Model for encoding movies.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        self.embedding_model = MovieModel()

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


# A multi-task model
# There are two critical parts to multi-task recommenders:
#
# They optimize for two or more objectives, and so have two or more losses.
# They share variables between the tasks, allowing for transfer learning.
#     We will have two tasks: one that predicts ratings, and one that predicts movie watches.

class MovielensModel(tfrs.models.Model):

    def __init__(self, layer_sizes):
        super().__init__()

        print("CREATING QUERY MODEL...")
        self.query_model = QueryModel(layer_sizes)
        print("CREATING CANDIDATE MODEL...")
        self.candidate_model = CandidateModel(layer_sizes)

        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        #         Metrics
        # In our training data we have positive (user, movie) pairs. To figure out how good our model is, we need to compare the affinity score that the model calculates for this pair to the scores of all the other possible candidates: if the score for the positive pair is higher than for all other candidates, our model is highly accurate.
        #
        # To do this, we can use the tfrs.metrics.FactorizedTopK metric. The metric has one required argument: the dataset of candidates that are used as implicit negatives for evaluation.

        #         Loss
        # The next component is the loss used to train our model. TFRS has several loss layers and tasks to make this easy.
        #
        # In this instance, we'll make use of the Retrieval task object: a convenience wrapper that bundles together the loss function and metric computation:
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


loaded_model = MovielensModel([32])
loaded_model.load_weights('WEIGHTS_2l')

loaded_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# Evaluate the model on the test set.
print(loaded_model.evaluate(cached_test, return_dict=True))
