import pandas as pd
import xmltodict
from skimage import io
from skimage import transform


class PascalVOC(object):

    img_idx = 0
    lbl_idx = 1
    labels = [
        'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
        'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor'
    ]
    label2idx = {lbl: idx for idx, lbl in enumerate(labels)}
    idx2label = {idx: lbl for idx, lbl in enumerate(labels)}

    def __init__(self, voc_dir='', y_onehot=True):
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

        y = [self._get_class(img) for img in mb[self.img_idx]]
        bbox = [self._get_bbox(img) for img in mb[self.img_idx]]

        return np.array(X), np.array(y, dtype=np.uint8), np.array(bbox)

    def _load(self):
        train_set = self._read_set(self.imageset_dir + '/train.txt')
        test_set = self._read_set(self.imageset_dir + '/val.txt')
        return train_set, test_set

    def _read_set(self, filename):
        return pd.read_csv(filename, header=None, delim_whitespace=True)

    def _img_path(self, img):
        return '{}/{}.jpg'.format(self.img_dir, img)

    def _label_path(self, img):
        return '{}/{}.xml'.format(self.bbox_dir, img)

    def _get_class(self, img_name):
        with open(self._label_path(img_name), 'r') as f:
            xml = xmltodict.parse(f.read())

        obj = xml['annotation']['object']

        if type(obj) is list:
            obj = obj[0]

        label = self.label2idx[obj['name']]

        if self.onehot:
            y = np.zeros_like(self.labels, dtype=np.uint8)
            y[label] = 1
        else:
            y = label

        return y

    def _get_bbox(self, img_name):
        with open(self._label_path(img_name), 'r') as f:
            xml = xmltodict.parse(f.read())

        obj = xml['annotation']['object']

        if type(obj) is list:
            obj = obj[0]

        bbox = np.array(list(obj['bndbox'].values()), dtype=np.uint8)

        return np.array(bbox)

    def _resize_img(self, img, to_size=(224, 224), bbox=None):
        new_bbox = bbox

        if bbox is not None:
            x, y, w, h = bbox
            img_w, img_h = img.shape[:2]
            new_w, new_h = to_size
            rw, rh = new_w / img_w, new_h / img_h
            new_bbox = (x * rw, y * rh, w * rw, h * rh)

        new_img = transform.resize(img, output_shape=to_size, preserve_range=True)

        return new_img, new_bbox
