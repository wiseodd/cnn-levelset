from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
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


model = VGG16(weights='imagenet', include_top=False)

generator = gen.pascal_datagen_singleobj(64, include_label=False, random=False)
features = model.predict_generator(generator, gen.pascal.train_set.size)

np.save('vgg_features_train.npy', features)
