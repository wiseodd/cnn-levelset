from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.regularizers import l2
from keras.optimizers import SGD
from cnnlevelset.pascalvoc_util import PascalVOC
from cnnlevelset.config import *
from cnnlevelset.generator import pascal_datagen, pascal_datagen_singleobj

import keras.backend as K
import tensorflow as tf
import numpy as np


tf.python.control_flow_ops = tf


def split_labels(y):
    y_cls = y[:, :, 0]
    y_reg = y[:, :, 1:]
    idxes = np.argmax(y_cls, axis=1)
    y_reg = y_reg[range(y.shape[0]), idxes]
    return [y_cls, y_reg]


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
    if 0 <= epoch < 10:
        return 1e-2

    if 10 <= epoch < 20:
        return 1e-3

    if 20 <= epoch < 30:
        return 1e-4

    if 30 <= epoch < 40:
        return 1e-5

    return 1e-7


pascal = PascalVOC('/Users/wiseodd/Projects/VOCdevkit/VOC2012/')

inputs = Input(shape=(7, 7, 512))

x = Convolution2D(128, 1, 1)(inputs)
x = Flatten()(x)
x = Dense(256, activation='relu', W_regularizer=l2(l=0.01))(x)
x = Dropout(p=0.5)(x)
cls_head = Dense(20, activation='softmax', name='cls')(x)

x = Dense(256, activation='relu', W_regularizer=l2(l=0.01))(x)
reg_head = Dense(4, activation='linear', name='reg')(x)

model = Model(input=inputs, output=[cls_head, reg_head])
model.compile(optimizer='adam',
              loss={'cls': 'categorical_crossentropy', 'reg': reg_loss},
              loss_weights={'cls': 1., 'reg': 1.},
              metrics={'cls': 'accuracy'})

X_train, y_train = pascal.load_train_data()
y_train = split_labels(y_train)

model.fit(X_train, y_train,
          batch_size=64, nb_epoch=80, validation_split=0.1)

model.save('data/models/model_vgg_singleobj.h5')