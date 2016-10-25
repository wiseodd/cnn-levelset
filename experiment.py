from keras.models import load_model
from skimage.io import imshow
from cnnlevelset.pascalvoc_util import PascalVOC
from cnnlevelset.localizer import Localizer
from cnnlevelset.generator import pascal_datagen
from cnnlevelset import config as cfg

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


m = 64
nb_epoch = 10
pascal = PascalVOC(voc_dir=cfg.PASCAL_PATH)

if len(sys.argv) > 1 and sys.argv[1] == 'test':
    localizer = Localizer(load=True)

    model = localizer.model
    X_test, y_test = pascal.get_test_set(10, random=True)

    cls_preds, bbox_preds = model.predict(X_test)
    print(cls_preds.shape, bbox_preds)

    for img, y, cls_pred, bbox_pred in zip(X_test, y_test, cls_preds, bbox_preds):
        idxs = np.where(cls_pred > 0.5)[0]

        if idxs.size == 0:
            continue

        labels = [pascal.idx2label[i] for i in idxs]
        bboxes = bbox_pred.reshape(20, 4)[idxs]

        print(labels)

        img = img.reshape(224, 224, 3)
        imshow(pascal.draw_bbox(img, bboxes[0]))
        plt.show()

    sys.exit(0)

localizer = Localizer()
localizer.train(pascal_datagen(m), nb_epoch)
