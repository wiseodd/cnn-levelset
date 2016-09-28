import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import io, color


def imshow(img):
    plt.imshow(img, cmap='Greys_r')
    plt.show()


def grad_norm(grad):
    """ Calculate gradient magnitude (2-norm) """
    return np.sqrt(np.sum(np.square(grad), axis=0))


dt = 1.
alpha = 1.


if __name__ == '__main__':
    # Load image and convert to grayscale
    img = io.imread('data/img/twoObj.bmp')
    img = color.rgb2gray(img)

    assert len(img.shape) == 2

    # Preprocessing: mean substraction and denoising (by applying Gaussian blur)
    img = img - np.mean(img)
    img_smooth = scipy.ndimage.filters.gaussian_filter(img, 0)

    # Derive image edge features f from its gradient
    dimg = np.gradient(img_smooth)
    dimg_norm = grad_norm(dimg)

    f = np.exp(-alpha * dimg_norm)
    # f = 1. / (1. + dimg_norm**2)

    assert img.shape == f.shape

    # Initialize surface function phi
    # 5px from image border with value 5 and -5 elsewhere
    # The 0 level curve would be a square curve
    phi = 5 * np.ones_like(img)
    phi[5:-5, 5:-5] = -5.

    imshow(img_smooth)
    imshow(f)

    for i in range(100):
        # Calculate grad phi's norm
        dphi = np.gradient(phi)
        dphi_norm = grad_norm(dphi)

        # Solve level set equation's PDE
        phi = phi + dt * (f * dphi_norm)

        if i % 10 == 0:
            plt.imshow(img, cmap='Greys_r')
            plt.hold(True)
            plt.contour(phi, 0, colors='r')
            plt.draw()
            plt.hold(False)
            plt.show()
