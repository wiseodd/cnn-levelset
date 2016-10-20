from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from .config import *

import tensorflow as tf
import numpy as np


def smooth_l1(x):
    x = tf.abs(x)

    x = tf.select(
        tf.less(x, 1),
        tf.mul(tf.square(x), 0.5),
        tf.sub(x, 0.5)
    )

    x = tf.reshape(x, shape=[-1, 20, 4])
    x = tf.reduce_sum(x, 2)
    x = tf.reduce_mean(x, 1)

    return x


def reg_loss(y_true, y_pred):
    loss = smooth_l1(y_true - y_pred)
    return tf.reduce_mean(loss)


class Localizer(object):

    def __init__(self):
        inputs = Input(shape=(224, 224, 3))
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)

        # Classification head; Output: 20-way sigmoid
        cls_head = Dense(20, activation='sigmoid', name='cls')(x)

        # Regression head; Output: 20 classes x 4 regression points
        reg_head = Dense(80, activation='linear', name='reg')(x)

        self.model = Model(input=base_model.input, output=[cls_head, reg_head])
        self.model.compile(optimizer='adam',
                           loss={'cls': 'binary_crossentropy', 'reg': reg_loss},
                           loss_weights={'cls': 1., 'reg': 1.},
                           metrics={'cls': 'accuracy'})

    def train(self, data_generator, nb_epoch=10):
        self.model.fit_generator(generator=data_generator,
                                 samples_per_epoch=5120,
                                 nb_epoch=nb_epoch,
                                 callbacks=[ModelCheckpoint(MODEL_PATH)])

    def predict(self, X):
        return self.model.predict(X)
