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
    if 0 <= epoch < 100:
        return 1e-3

    if 100 <= epoch < 140:
        return 1e-4

    if 140 <= epoch < 180:
        return 1e-5

    return 1e-6


class Localizer(object):

    custom_objs = {'reg_loss': reg_loss}

    def __init__(self, model_path=None):
        if model_path is not None:
            self.model = self.load_model(model_path)
        else:
            # VGG16 last conv features
            inputs = Input(shape=(7, 7, 512))
            x = Convolution2D(128, 1, 1)(inputs)
            x = Flatten()(x)

            # Cls head
            h_cls = Dense(256, activation='relu', W_regularizer=l2(l=0.01))(x)
            h_cls = Dropout(p=0.5)(h_cls)
            cls_head = Dense(20, activation='softmax', name='cls')(h_cls)

            # Reg head
            h_reg = Dense(256, activation='relu', W_regularizer=l2(l=0.01))(x)
            reg_head = Dense(80, activation='linear', name='reg')(h_reg)

            # Joint model
            self.model = Model(input=inputs, output=[cls_head, reg_head])

    def train(self, X, y, optimizer='adam', nb_epoch=200):
        self.model.compile(optimizer='adam',
                           loss={'cls': 'categorical_crossentropy', 'reg': reg_loss},
                           loss_weights={'cls': 1., 'reg': 1.},
                           metrics={'cls': 'accuracy'})

        callbacks = [ModelCheckpoint(MODEL_PATH),
                     LearningRateScheduler(scheduler)]

        self.model.fit(X, y, batch_size=32, nb_epoch=nb_epoch, callbacks=callbacks)

    def predict(self, X):
        return self.model.predict(X)

    def load_model(self, model_path):
        return load_model(model_path, custom_objects=self.custom_objs)
