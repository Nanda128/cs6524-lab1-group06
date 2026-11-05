import numpy as np
from scipy import ndimage

RED_WEIGHT = .299
GREEN_WEIGHT = .587
BLUE_WEIGHT = .114
BLUR_SIGMA = 2
KERNEL_RADIUS_IN_SIGMAS = 3


def to_grayscale(rgb):
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    return RED_WEIGHT * red + GREEN_WEIGHT * green + BLUE_WEIGHT * blue


def build_gaussian_kernel(sigma):
    radius = int(np.ceil(KERNEL_RADIUS_IN_SIGMAS * sigma))
    offsets = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(offsets, offsets)

    distance_squared = x**2 + y**2
    kernel = np.exp(-distance_squared / (2 * sigma**2))

    return kernel / kernel.sum()


def gaussian_blur(image, sigma):
    kernel = build_gaussian_kernel(sigma)
    return ndimage.convolve(image, kernel, mode='nearest')


def preprocess(rgb, blur_sigma=BLUR_SIGMA):
    grayscale = to_grayscale(rgb)
    blurred = gaussian_blur(grayscale, blur_sigma)

    return [
        ('Original', rgb),
        ('Grayscale', grayscale),
        (f'Gaussian blur (sigma={blur_sigma})', blurred),
    ]
