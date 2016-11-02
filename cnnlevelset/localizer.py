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
    if 0 <= epoch < 5:
        return 1e-3

    if 5 <= epoch < 10:
        return 1e-4

    if 10 <= epoch < 15:
        return 1e-5

    return 1e-6


class Localizer(object):

    custom_objs = {'reg_loss': reg_loss}

    def __init__(self, load=False):
        if load:
            self.model = load_model(MODEL_PATH, custom_objects=self.custom_objs)
        else:
            inputs = Input(shape=(224, 224, 3))
            base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

            for layer in base_model.layers:
                layer.trainable = False

            conv_feature = Flatten()(base_model.output)

            # Classification head
            x = Dense(512, activation='relu', W_regularizer=l2(l=0.01))(conv_feature)
            cls_head = Dense(20, activation='softmax', name='cls')(x)

            # Regression head
            x = Dense(512, activation='relu', W_regularizer=l2(l=0.01))(conv_feature)
            reg_head = Dense(4, activation='linear', name='reg')(x)

            self.model = Model(input=base_model.input, output=[cls_head, reg_head])

    def train(self, data_generator, optimizer='adam', nb_epoch=10):
        self.model.compile(optimizer=optimizer,
                           loss={'cls': 'categorical_crossentropy', 'reg': reg_loss},
                           loss_weights={'cls': 1., 'reg': 1.},
                           metrics={'cls': 'accuracy'})

        callbacks = [ModelCheckpoint(MODEL_PATH),
                     TensorBoard(),
                     LearningRateScheduler(scheduler)]

        self.model.fit_generator(generator=data_generator,
                                 samples_per_epoch=4800,
                                 nb_epoch=nb_epoch,
                                 callbacks=callbacks)

    def predict(self, X):
        return self.model.predict(X)

    def load_model(self):
        self.model = load_model(MODEL_PATH, custom_objects=self.custom_objs)
