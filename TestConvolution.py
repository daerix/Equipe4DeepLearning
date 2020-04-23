import tensorflow as tf
import os
from keras import datasets
import matplotlib.pyplot as plt
import TestImport as dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train_generator, validation_generator, class_names, inputShape = dataset.importDataset()
chanDim = -1

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16, (4, 4), strides=(2, 2),
                                 padding="valid", activation="relu", input_shape=inputShape))
model.add(tf.keras.layers.Conv2D(
    32, (3, 3), padding="same", activation="relu"))
model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
model.add(tf.keras.layers.Dropout(0.35))

model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=(
    2, 2), padding="same", activation="relu"))
model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
model.add(tf.keras.layers.Dropout(0.35))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.35))
model.add(tf.keras.layers.Dense(len(class_names), activation="softmax"))

model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])


logdir = os.path.join('logs')
tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model.fit_generator(train_generator, steps_per_epoch=100, epochs=5,
                    validation_data=validation_generator,
                    validation_steps=50)
