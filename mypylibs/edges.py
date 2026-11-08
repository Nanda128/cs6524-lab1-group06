import numpy as np
from scipy import ndimage

BLUR_SIGMA = 6
MASK_SMOOTHING_SIGMA = 2
KERNEL_RADIUS_IN_SIGMAS = 3
EDGE_RING_WIDTH = 10

THRESHOLD_CANDIDATES = np.linspace(0.04, 0.40, 37)
MIN_OBJECT_AREA = 0.02
MAX_OBJECT_AREA = 0.45
FALLBACK_THRESHOLD = 0.15

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
SOBEL_Y = SOBEL_X.T


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


def blur_colour_image(rgb, sigma):
    layers = [gaussian_blur(rgb[:, :, layer], sigma) for layer in range(3)]
    return np.dstack(layers)


def estimate_background_colour(rgb):
    w = EDGE_RING_WIDTH
    top = rgb[:w].reshape(-1, 3)
    bottom = rgb[-w:].reshape(-1, 3)
    left = rgb[w:-w, :w].reshape(-1, 3)
    right = rgb[w:-w, -w:].reshape(-1, 3)
    ring = np.concatenate([top, bottom, left, right])
    return np.median(ring, axis=0)


def colour_distance(rgb, colour):
    difference = rgb - colour
    return np.sqrt((difference**2).sum(axis=2)) / np.sqrt(3)


def choose_threshold(distance):
    areas = np.array([keep_centre_object(distance > level).mean()
                      for level in THRESHOLD_CANDIDATES])

    change = np.abs(areas[2:] - areas[:-2]) / np.maximum(areas[1:-1], 1e-6)
    sensible = (areas[1:-1] > MIN_OBJECT_AREA) & (areas[1:-1] < MAX_OBJECT_AREA)
    if not sensible.any():
        return FALLBACK_THRESHOLD

    change[~sensible] = np.inf
    return THRESHOLD_CANDIDATES[1:-1][np.argmin(change)]


def keep_centre_object(mask):
    eight_neighbours = np.ones((3, 3))  # pixels touching diagonally count as connected
    labels, blob_count = ndimage.label(mask, structure=eight_neighbours)
    if blob_count == 0:
        return mask

    _, nearest = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    centre_row = mask.shape[0] // 2
    centre_col = mask.shape[1] // 2
    chosen = labels[nearest[0][centre_row, centre_col], nearest[1][centre_row, centre_col]]

    return ndimage.binary_fill_holes(labels == chosen)


def smooth_mask(mask, sigma):
    return gaussian_blur(mask.astype(float), sigma) > 0.5


def sobel_magnitude(mask):
    image = mask.astype(float)
    gradient_x = ndimage.convolve(image, SOBEL_X, mode='constant', cval=0)
    gradient_y = ndimage.convolve(image, SOBEL_Y, mode='constant', cval=0)
    return np.hypot(gradient_x, gradient_y)


def find_border(mask):
    strength = sobel_magnitude(mask)
    return (strength > 0) & mask


def preprocess(rgb):
    blurred = blur_colour_image(rgb, BLUR_SIGMA)
    background = estimate_background_colour(rgb)
    distance = colour_distance(blurred, background)
    level = choose_threshold(distance)
    mask = distance > level
    cleaned = keep_centre_object(mask)
    smoothed = smooth_mask(cleaned, MASK_SMOOTHING_SIGMA)
    strength = sobel_magnitude(smoothed)
    border = find_border(smoothed)

    stages = [
        ('Original', rgb),
        (f'Gaussian blur (sigma={BLUR_SIGMA})', blurred),
        ('Distance from background colour', distance),
        (f'Threshold (auto-picked: {level:.2f})', mask),
        ('Centre object, holes filled', cleaned),
        (f'Smoothed mask (sigma={MASK_SMOOTHING_SIGMA})', smoothed),
        ('Sobel gradient magnitude', strength),
        ('Border (1 pixel thick)', border),
    ]
    return stages, border
