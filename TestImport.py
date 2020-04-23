import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pathlib


def importDataset():

    train_data_dir = pathlib.Path(
        r"C:\Users\Virgile\Documents\Personal\Cours\Deep Learning\CNN\Train_Images")
    test_data_dir = pathlib.Path(
        r"C:\Users\Virgile\Documents\Personal\Cours\Deep Learning\CNN\Test_Images")


    CLASS_NAMES = np.array(
        [item.name for item in train_data_dir.glob('*') if item.name != "LICENSE.txt"])
    print(CLASS_NAMES)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)



    IMG_HEIGHT = 500
    IMG_WIDTH = 500
    BATCH_SIZE = 32

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(500, 500),
        batch_size=BATCH_SIZE,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(500, 500),
        batch_size=BATCH_SIZE,
        class_mode='binary')


    return train_generator, validation_generator, list(CLASS_NAMES), (IMG_HEIGHT, IMG_WIDTH, 3)
