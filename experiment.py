from skimage.io import imshow
from cnnlevelset.pascalvoc_util import PascalVOC
from cnnlevelset.localizer import Localizer
from cnnlevelset.segmenter import *

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.python.control_flow_ops = tf

pascal = PascalVOC('/Users/wiseodd/Projects/VOCdevkit/VOC2012/')

with open('data/test_segmentation.txt') as f:
    names = f.read().split('\n')

# X_img_test, X_test, y_test = pascal.get_test_data(1, random=True)
X_img_test, X_test, y_test = pascal.get_data_by_name(names)

localizer = Localizer(model_path='data/models/model_vgg_singleobj.h5')
cls_preds, bbox_preds = localizer.predict(X_test)

for img, y, cls_pred, bbox_pred in zip(X_img_test, y_test, cls_preds, bbox_preds):
    label = pascal.idx2label[np.argmax(cls_pred)]

    print(label)

    img = img.reshape(224, 224, 3)
    plt.imshow(pascal.draw_bbox(img, bbox_pred))
    plt.show()

    phi = phi_from_bbox(img[:, :, 0], bbox_pred)
    levelset_segment(img, phi=phi, sigma=5, v=1, alpha=100000, n_iter=80, print_after=None)

    input()
