import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Activation, Flatten
from keras.models import Model


class Localizer(object):

    def __init__(self):
        base_model = ResNet50(include_top=False, weight='imagenet')

        for layer in base_model:
            layer.trainable = False

        # Classification head; Output: 20-way sigmoid
        x = base_model.output
        x = Flatten()(x)
        cls_head = Dense(20, activation='sigmoid', name='cls')(x)

        # Regression head; Output: 20 classes x 4 regression points
        x = base_model.output
        x = Flatten()(x)
        reg_head = Dense(80, activation='linear', name='reg')(x)

        self.model = Model(input=base_model, output=[cls_head, reg_head])
        model.compile(optimizer='adam',
                      loss={'cls': 'binary_crossentropy', 'reg': 'mse'},
                      loss_weights={'cls': 1., 'reg': 1.})

    def train(self, X, y_cls, y_reg):
        self.model.fit(x={'cls': X, 'reg': X},
                       y={'cls': y_cls, 'reg': y_reg},
                       batch_size=32, nb_epoch=10)

    def predict(self, X):
        return self.model.predict(X)
