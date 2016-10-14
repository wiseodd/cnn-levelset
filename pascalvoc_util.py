import numpy as np
import pandas as pd
import xmltodict
from skimage import io
from skimage import transform
from skimage import draw
from collections import defaultdict


class PascalVOC(object):
    """
    Pascal VOC dataset utility.

    Arguments
    ---------
        voc_idr: string
            Indicating path of the Pascal VOC devkit.
        y_onehot: bool, default True
            Whether the class should be onehot encoded value or not
    """

    img_idx = 0
    lbl_idx = 1
    labels = [
        'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
    ]
    label2idx = {lbl: idx for idx, lbl in enumerate(labels)}
    idx2label = {idx: lbl for idx, lbl in enumerate(labels)}

    def __init__(self, voc_dir, y_onehot=True):
        self.voc_dir = voc_dir.rstrip('/')
        self.imageset_dir = voc_dir + '/ImageSets/Main'
        self.img_dir = voc_dir + '/JPEGImages'
        self.bbox_dir = voc_dir + '/Annotations'
        self.train_set, self.test_set = self._load()
        self.onehot = y_onehot

    def next_minibatch(self, size):
        mb = self.train_set.sample(size)

        X = [
            io.imread(self._img_path(img))
            for img
            in mb[self.img_idx]
        ]

        y = [np.column_stack(self.get_class_bbox(img)) for img in mb[self.img_idx]]

        return np.array(X), np.array(y)

    def draw_bbox(self, img, bbox, color=[255, 0, 0], line_width=3):
        xmin, ymin, xmax, ymax = bbox
        img_bbox = np.copy(img)

        img_bbox[ymin-line_width:ymin, xmin-line_width:xmax+line_width] = color
        img_bbox[ymax:ymax+line_width, xmin-line_width:xmax+line_width] = color
        img_bbox[ymin-line_width:ymax+line_width, xmin-line_width:xmin] = color
        img_bbox[ymin-line_width:ymax+line_width, xmax:xmax+line_width] = color

        return img_bbox

    def get_class_bbox(self, img_name):
        with open(self._label_path(img_name), 'r') as f:
            xml = xmltodict.parse(f.read())

        objs = xml['annotation']['object']

        if type(objs) is not list:
            objs = [objs]

        clses = np.zeros_like(self.labels, dtype=np.float)
        bboxes = np.zeros(shape=[len(self.labels), 4], dtype=np.float)
        bbox_cls = defaultdict(list)

        for obj in objs:
            idx = self.label2idx[obj['name']]
            clses[idx] = 1

            bndbox = obj['bndbox']
            bbox_cls[idx].append(bbox)

        def bbox_area(bbox):
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin
            return w * h

        for k, v in bbox_cls.items():
            max_bbox = sorted(bboxes, key=bbox_area, reverse=True)[0]
            bboxes[k] = max_bbox

        return clses, bboxes

    def resize_img(self, img, to_size=(224, 224), bbox=None):
        new_bbox = bbox

        if bbox is not None:
            x, y, w, h = bbox
            img_w, img_h = img.shape[:2]
            new_w, new_h = to_size
            rw, rh = new_w / img_w, new_h / img_h
            new_bbox = (x * rw, y * rh, w * rw, h * rh)

        new_img = transform.resize(img, output_shape=to_size, preserve_range=True)

        return new_img, new_bbox

    def _load(self):
        train_set = self._read_dataset(self.imageset_dir + '/train.txt')
        test_set = self._read_dataset(self.imageset_dir + '/val.txt')
        return train_set, test_set

    def _read_dataset(self, filename):
        return pd.read_csv(filename, header=None, delim_whitespace=True)

    def _img_path(self, img):
        return '{}/{}.jpg'.format(self.img_dir, img)

    def _label_path(self, img):
        return '{}/{}.xml'.format(self.bbox_dir, img)