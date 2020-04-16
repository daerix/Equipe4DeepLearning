import tensorflow.compat.v1 as tf
import json
import numpy as np
import keras
import pandas as pd
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dropout, Activation

dataset_url = r"./testCSV.csv"

#r"./twitter_dataset.csv"
training = np.genfromtxt(
    r"./testCSV.csv",
    delimiter='",',
    skip_header=1,
    usecols=(0, 5),
    dtype='O',
    max_rows=1000000,
)


train_x = [x[1].decode("utf-8") for x in training]
train_y_temp = np.asarray([x[0].decode("utf-8") for x in training])

train_y = []
for y in train_y_temp:
    train_y.append(int(y[1]))

train_y = np.asarray(train_y)

max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_x)

dictionary = tokenizer.word_index
with open("dictionary.json", "w") as dictionary_file:
    json.dump(dictionary, dictionary_file)


def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]


allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode="binary")
# treat the labels as categories
train_y = keras.utils.to_categorical(train_y, 2)


model = Sequential(
    [
        keras.layers.Dense(512, input_shape=(max_words,), activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='sigmoid'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation="softmax"),
    ]
)


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(
    train_x,
    train_y,
    batch_size=32,
    epochs=5,
    verbose=1,
    validation_split=0.1,
    shuffle=True,
)

