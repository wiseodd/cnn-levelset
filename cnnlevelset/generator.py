from .pascalvoc_util import PascalVOC


pascal = PascalVOC(voc_dir='/home/lab_sd/Projects/VOCdevkit/VOC2012')


def pascal_datagen(mb_size, include_regression=True, flatten=False):
    while True:
        X, y = pascal.next_minibatch(size=mb_size)

        y_cls = y[:, :, 0]
        y_reg = y[:, :, 1:]
        y_reg = y_reg.reshape(mb_size, -1)

        if flatten:
            X = X.reshape(mb_size, -1)

        if include_regression:
            yield X, [y_cls, y_reg]
        else:
            yield X, y_cls
