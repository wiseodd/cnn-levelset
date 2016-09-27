import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt


def imshow(img):
    plt.imshow(img, cmap='Greys_r')
    plt.show()


dt = 1.
alpha = 1.


if __name__ == '__main__':
    img = plt.imread('twoObj.bmp')
    img = img - np.mean(img)
    img_smooth = scipy.ndimage.filters.median_filter(img, 5)

    dimg_y, dimg_x = np.gradient(img_smooth)
    dimg_norm = np.sqrt(dimg_x**2 + dimg_y**2)

    f = np.exp(-alpha * dimg_norm)

    phi = 5 * np.ones_like(img)
    phi[5:-5, 5:-5] = -5.

    imshow(img_smooth)
    imshow(f)

    for i in range(100):
        dphi_y, dphi_x = np.gradient(phi)
        dphi_norm = np.sqrt(dphi_x**2 + dphi_y**2)

        phi = phi + dt * (f * dphi_norm)

        if i % 10 == 0:
            plt.imshow(img, cmap='Greys_r')
            plt.hold(True)
            plt.contour(phi, 0, colors='r')
            plt.draw()
            plt.hold(False)
            plt.show()
