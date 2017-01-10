import cnnlevelset.segmenter as s
import matplotlib.pyplot as plt
from skimage import io


if __name__ == '__main__':
    img = io.imread('data/img_74_ori.jpg')
    print(img.shape)

    # res = s.levelset_segment(img, sigma=5, v=1, alpha=100000, n_iter=80)

    res = s.levelset_segment_theano(img, sigma=5, v=1, alpha=100000, n_iter=80)

    io.imshow(res)
    plt.show()
