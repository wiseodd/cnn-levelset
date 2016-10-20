from keras.models import load_model
from skimage.io import imshow
from cnnlevelset.pascalvoc_util import PascalVOC
from cnnlevelset.localizer import Localizer
from cnnlevelset.generator import pascal_datagen
from cnnlevelset import config as cfg

import sys
import tensorflow as tf
import numpy as np


m = 32
nb_epoch = 10
pascal = PascalVOC(voc_dir=cfg.PASCAL_PATH)

if len(sys.argv) > 1 and sys.argv[1] == 'test':
    model = load_model(cfg.MODEL_PATH)
    X_test, y_test = pascal.get_test_set(10, random=False)

    img = X_test[9:10]

    cls_pred, bbox_pred = model.predict(X_test[9:10])

    idxs = np.where(cls_pred[0] > 0.5)[0]
    labels = [pascal.idx2label[i] for i in idxs]
    bboxes = bbox_pred[0].reshape(20, 4)[idxs]

    img = img.reshape(224, 224, 3)
    print(img.shape, bboxes.shape)

    imshow(pascal.draw_bbox(img, bboxes[0]))

    sys.exit(0)

localizer = Localizer()
localizer.train(pascal_datagen(m), nb_epoch)
