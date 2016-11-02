from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.regularizers import l2
from .config import *

import keras.backend as K
import tensorflow as tf
import numpy as np


def smooth_l1(x):
    x = tf.abs(x)

    x = tf.select(
        tf.less(x, 1),
        tf.mul(tf.square(x), 0.5),
        tf.sub(x, 0.5)
    )

    x = tf.reshape(x, shape=[-1, 4])
    x = tf.reduce_sum(x, 1)

    return x


def reg_loss(y_true, y_pred):
    return smooth_l1(y_true - y_pred)


def scheduler(epoch):
    if 0 <= epoch < 80:
        return 1e-3

    if 80 <= epoch < 120:
        return 1e-4

    if 120 <= epoch < 150:
        return 1e-5

    return 1e-6


class Localizer(object):

    custom_objs = {'reg_loss': reg_loss}

    def __init__(self, load=False):
        if load:
            self.load_model()
        else:
            # VGG16 last conv features
            inputs = Input(shape=(7, 7, 512))

            # Cls head
            x = Convolution2D(128, 1, 1)(inputs)
            x = Flatten()(x)
            x = Dense(256, activation='relu', W_regularizer=l2(l=0.01))(x)
            x = Dropout(p=0.5)(x)
            cls_head = Dense(20, activation='softmax', name='cls')(x)

            # Reg head
            x = Dense(256, activation='relu', W_regularizer=l2(l=0.01))(x)
            reg_head = Dense(4, activation='linear', name='reg')(x)

            # Joint model
            self.model = Model(input=inputs, output=[cls_head, reg_head])

    def train(self, X, y, val_split=0.1, optimizer='adam', nb_epoch=10):
        self.model.compile(optimizer='adam',
                           loss={'cls': 'categorical_crossentropy', 'reg': reg_loss},
                           loss_weights={'cls': 1., 'reg': 1.},
                           metrics={'cls': 'accuracy'})

        callbacks = [ModelCheckpoint(MODEL_PATH),
                     TensorBoard(),
                     LearningRateScheduler(scheduler)]

        self.model.fit(X, y, batch_size=64, nb_epoch=80,
                       validation_split=val_split,
                       callbacks=callbacks)

    def predict(self, X):
        return self.model.predict(X)

    def load_model(self):
        self.model = load_model(MODEL_PATH, custom_objects=self.custom_objs)
