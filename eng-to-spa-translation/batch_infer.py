import pickle
import re
import string

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

import base

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 15000
sequence_length = 20
batch_size = 64

model = base.build_and_compile_transformer_model()
model.load_weights("weights")


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


eng_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)

spa_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)

from_disk = pickle.load(open("eng_vec.pkl", "rb"))
eng_vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
eng_vectorization.set_weights(from_disk["weights"])

from_disk = pickle.load(open("spa_vec.pkl", "rb"))
spa_vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
spa_vectorization.set_weights(from_disk["weights"])

spa_vocab = spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20


def decode_sequences(input_sentences):
    tokenized_input_sequences = eng_vectorization(input_sentences)
    decoded_sentences = ["[start]"] * len(input_sentences)

    for i in range(max_decoded_sentence_length):
        tokenized_target_sentences = spa_vectorization(decoded_sentences)[:, :-1]
        predictions = model([tokenized_input_sequences, tokenized_target_sentences])
        sampled_token_indices = np.argmax(predictions[:, i, :], axis=1)
        sampled_tokens = [spa_index_lookup[word] for word in sampled_token_indices]
        decoded_sentences = [
            sentence + " " + sampled_token if not sampled_token == "[end]" else sentence
            for sentence, sampled_token in zip(decoded_sentences, sampled_tokens)
        ]
        if all(token == "[end]" for token in sampled_tokens):
            decoded_sentences = [sentence + " [end]" for sentence in decoded_sentences]
            break

    return decoded_sentences


import time

start = time.time()
print(
    decode_sequences(
        [
            "i have a dog",
            "i have a cat",
            "i have one cat and one dog",
            "I have five different animals",
            "I want to go on a trip to the carribeans",
        ]
    )
)
end = time.time()
print(end - start)
