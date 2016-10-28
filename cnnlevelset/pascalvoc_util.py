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
    img_size = (224, 224)

    def __init__(self, voc_dir):
        self.voc_dir = voc_dir.rstrip('/')
        self.imageset_dir = voc_dir + '/ImageSets/Main'
        self.img_dir = voc_dir + '/JPEGImages'
        self.bbox_dir = voc_dir + '/Annotations'
        self.train_set, self.test_set = self._load()

    def next_minibatch(self, size):
        mb = self.train_set.sample(size)
        return self.load_data(mb)

    def get_test_set(self, size, random=True):
        if random:
            imgs = self.test_set.sample(size)
        else:
            imgs = self.test_set.head(size)

        return self.load_data(imgs)

    def load_data(self, img_names):
        X = [transform.resize(io.imread(self._img_path(img)), self.img_size)
             for img
             in img_names[self.img_idx]]

        y = [np.column_stack(self.get_class_bbox(img))
             for img
             in img_names[self.img_idx]]

        return np.array(X), np.array(y)

    def draw_bbox(self, img, bbox, color=[1, 0, 0], line_width=3):
        xmin, ymin, xmax, ymax = bbox
        h, w = img.shape[:2]

        xmin = int(round(xmin * w))
        xmax = int(round(xmax * w))
        ymin = int(round(ymin * h))
        ymax = int(round(ymax * h))

        img_bbox = np.copy(img)

        img_bbox[ymin-line_width:ymin, xmin-line_width:xmax+line_width] = color
        img_bbox[ymax:ymax+line_width, xmin-line_width:xmax+line_width] = color
        img_bbox[ymin-line_width:ymax+line_width, xmin-line_width:xmin] = color
        img_bbox[ymin-line_width:ymax+line_width, xmax:xmax+line_width] = color

        return img_bbox

    def get_class_bbox(self, img_name):
        with open(self._label_path(img_name), 'r') as f:
            xml = xmltodict.parse(f.read())

        img_size = xml['annotation']['size']
        img_w, img_h = float(img_size['width']), float(img_size['height'])

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
            bbox = (bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax'])
            bbox = self._normalize_bbox(bbox, (img_w, img_h))
            bbox = np.array(bbox, dtype=np.float)
            bbox_cls[idx].append(bbox)

        for k, v in bbox_cls.items():
            sample_idx = np.random.randint(0, len(v))
            bboxes[k] = v[sample_idx]

        return clses, bboxes

    def _load(self):
        # train_set = self._read_dataset(self.imageset_dir + '/train_singleobj.txt')
        train_set = self._read_dataset('data/train_singleobj.txt')
        test_set = self._read_dataset(self.imageset_dir + '/val.txt')

        return train_set, test_set

    def _read_dataset(self, filename):
        return pd.read_csv(filename, header=None, delim_whitespace=True)

    def _img_path(self, img):
        return '{}/{}.jpg'.format(self.img_dir, img)

    def _label_path(self, img):
        return '{}/{}.xml'.format(self.bbox_dir, img)

    def _normalize_bbox(self, bbox, img_dim):
        w, h = img_dim
        xmin, ymin, xmax, ymax = bbox

        def normalize(x, s):
            return float(x) / s

        xmin, ymin = normalize(xmin, w), normalize(ymin, h)
        xmax, ymax = normalize(xmax, w), normalize(ymax, h)

        return [xmin, ymin, xmax, ymax]