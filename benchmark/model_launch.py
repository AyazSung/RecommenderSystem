from model_files import *

print("CREATING MODEL...")
model = MODEL()
recommendation_model = model.loaded_model
print("Model is created...")
input_tensors = {
    'user_id': tf.constant(['138']),
    'timestamp': tf.constant([879024327]),
    'user_rating': tf.constant([4.0]),
    'user_occupation_label': tf.constant([4]),
    'raw_user_age': tf.constant([46.0]),
    'user_gender': tf.constant([True]),
    'user_occupation_text': tf.constant(['doctor']),
    'movie_title': tf.constant(["One Flew Over the Cuckoo's Nest (1975)"])
}

tr_emb_user, tr_emb_movie, tr_rating_pred = recommendation_model(input_tensors)

print("RAITING: ", tr_rating_pred.numpy()[0][0])

brute_force = tfrs.layers.factorized_top_k.BruteForce(recommendation_model.query_model)
brute_force.index_from_dataset(
    model.movies.batch(128).map(lambda title: (title, recommendation_model.candidate_model(title))))

# Get predictions for user 42.
_, titles = brute_force(input_tensors, k=10)
print(f"Top recommendations: {titles[0]}")
