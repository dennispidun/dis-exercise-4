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

print(eng_vectorization("hello"))
print(spa_vectorization("hola"))


def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = model([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence


print(decode_sequence("i have a dog"))
