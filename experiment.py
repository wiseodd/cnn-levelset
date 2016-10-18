from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from cnnlevelset.pascalvoc_util import PascalVOC

import sys
import tensorflow as tf


m = 64
nb_epoch = 10
model_path = 'data/model.h5'
pascal = PascalVOC(voc_dir='/Users/wiseodd/Projects/VOCdevkit/VOC2012/')


def data_generator(mb_size, flatten=True):
    while True:
        X, y = pascal.next_minibatch(size=mb_size)

        y_cls = y[:, :, 0]
        y_reg = y[:, :, 1:]

        if flatten:
            X = X.reshape(mb_size, -1)
            y_reg = y_reg.reshape(mb_size, -1)

        print(X[0], y_cls[0])

        yield X, y_cls


def k_binary_logloss(y_true, y_pred):
    return tf.contrib.losses.log_loss(y_pred, y_true)


if len(sys.argv) > 1 and sys.argv[1] == 'test':
    model = load_model(model_path)
    X_test, y_test = pascal.get_test_set(10, random=False)

    cls_pred, reg_pred = model.predict(X_test[6].reshape(1, -1))
    cls_pred = list(enumerate(cls_pred[0]))

    for idx, prob in sorted(cls_pred, key=lambda x: x[1], reverse=True)[:5]:
        print('{}: {:.4f}'.format(pascal.idx2label[idx], prob))

    sys.exit(0)

inputs = Input(shape=(112, 112, 3))

x = ZeroPadding2D((3, 3))(inputs)
x = Convolution2D(8, 7, 7, subsample=(2, 2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = ZeroPadding2D((1, 1))(x)
x = Convolution2D(16, 3, 3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

x = ZeroPadding2D((1, 1))(x)
x = Convolution2D(32, 3, 3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

x = ZeroPadding2D((1, 1))(x)
x = Convolution2D(64, 3, 3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(1024, W_regularizer=l2(0.01))(x)
x = Activation('relu')(x)
logit = Dense(20)(x)
prob = Activation('sigmoid', name='cls')(logit)
# reg_head = Dense(80, activation='linear', name='reg')(x)

model = Model(input=inputs, output=prob)
model.compile(optimizer='adam',
              loss={'cls': 'binary_crossentropy'},
              metrics={'cls': 'accuracy'})

datagen = data_generator(m, flatten=False)
model.fit_generator(datagen, 1280, nb_epoch, callbacks=[ModelCheckpoint(model_path)])
