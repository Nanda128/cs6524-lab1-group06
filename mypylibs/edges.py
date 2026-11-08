import numpy as np
from scipy import ndimage

BLUR_SIGMA = 2
MASK_SIGMA = 2
KERNEL_RADIUS_IN_SIGMAS = 3
RING_WIDTH = 10
THRESHOLD = 0.15

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
SOBEL_Y = SOBEL_X.T


def build_gaussian_kernel(sigma):
    radius = int(np.ceil(KERNEL_RADIUS_IN_SIGMAS * sigma))
    offsets = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(offsets, offsets)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(image, sigma):
    return ndimage.convolve(image, build_gaussian_kernel(sigma), mode='nearest')


def blur_colour_image(rgb, sigma):
    layers = [gaussian_blur(rgb[:, :, layer], sigma) for layer in range(3)]
    return np.dstack(layers).clip(0, 1)


def background_colour(rgb):
    w = RING_WIDTH
    ring = np.concatenate([rgb[:w].reshape(-1, 3), rgb[-w:].reshape(-1, 3),
                           rgb[w:-w, :w].reshape(-1, 3), rgb[w:-w, -w:].reshape(-1, 3)])
    return np.median(ring, axis=0)


def colour_distance(rgb, colour):
    return np.sqrt(((rgb - colour)**2).sum(axis=2) / 3)


def keep_largest_object(mask):
    labels, blobs = ndimage.label(mask, structure=np.ones((3, 3)))
    if blobs == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # label 0 is the background
    return ndimage.binary_fill_holes(labels == sizes.argmax())


def sobel_magnitude(mask):
    image = mask.astype(float)
    return np.hypot(ndimage.convolve(image, SOBEL_X, mode='constant'),
                    ndimage.convolve(image, SOBEL_Y, mode='constant'))


def preprocess(rgb):
    blurred = blur_colour_image(rgb, BLUR_SIGMA)
    distance = colour_distance(blurred, background_colour(rgb))
    mask = distance > THRESHOLD
    obj = keep_largest_object(mask)
    smooth = gaussian_blur(obj.astype(float), MASK_SIGMA) > 0.5
    strength = sobel_magnitude(smooth)
    border = (strength > 0) & smooth

    stages = [
        ('Original', rgb),
        (f'Gaussian blur, sigma {BLUR_SIGMA}', blurred),
        ('Distance from background', distance),
        (f'Threshold at {THRESHOLD}', mask),
        ('Largest object, holes filled', obj),
        (f'Mask smoothed, sigma {MASK_SIGMA}', smooth),
        ('Sobel magnitude', strength),
        ('Border', border),
    ]
    return stages, border
