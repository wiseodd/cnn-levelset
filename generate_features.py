from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.regularizers import l2

import keras.backend as K
import tensorflow as tf
import numpy as np
import cnnlevelset.generator as gen


inputs = Input(shape=(224, 224, 3))
model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

generator = gen.pascal_datagen_singleobj(64, include_label=False, random=False)
features = model.predict_generator(generator, gen.pascal.train_set.size)

np.save('cnn_features_train.npy', features)
