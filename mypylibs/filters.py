"""Hand-built spatial filters (no OpenCV / scikit-image)."""
import numpy as np
from scipy import ndimage


def to_grayscale(img):
    """RGB(A) uint8 array -> float grayscale in [0, 1] (Rec. 601 luma)."""
    img = np.asarray(img, dtype=float)
    if img.ndim == 2:
        return img / 255.0
    weights = np.array([0.299, 0.587, 0.114])
    return img[..., :3] @ weights / 255.0


def gaussian_kernel(sigma, size=None):
    """Normalised 2D Gaussian kernel. Default size covers +-3 sigma."""
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    if size % 2 == 0:
        size += 1
    r = size // 2
    ax = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def convolve(img, kernel):
    """2D convolution with edge replication."""
    return ndimage.convolve(img, kernel, mode='nearest')


def gaussian_blur(img, sigma=2.0, size=None):
    return convolve(img, gaussian_kernel(sigma, size))


SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)
SOBEL_Y = SOBEL_X.T


def sobel(img):
    """Return (gradient magnitude, gx, gy)."""
    gx = convolve(img, SOBEL_X)
    gy = convolve(img, SOBEL_Y)
    return np.hypot(gx, gy), gx, gy


def otsu_threshold(img, bins=256):
    """Otsu's threshold for an image in [0, 1]."""
    hist, edges = np.histogram(img, bins=bins, range=(0, 1))
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1 - w0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        between = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    return centers[np.nanargmax(between)]


def threshold(img, t=None, invert=None):
    """Binary mask. t=None -> Otsu.

    invert=True selects dark objects, False bright ones. invert=None picks
    automatically: the background is assumed to be what touches the image
    frame, so if most frame pixels are 'on' the mask is flipped.
    """
    if t is None:
        t = otsu_threshold(img)
    mask = img > t
    if invert is None:
        frame = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        invert = frame.mean() > 0.5
    return ~mask if invert else mask


def clean_mask(mask):
    """Keep the largest connected blob and fill its holes."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    largest = labels == (np.argmax(sizes) + 1)
    return ndimage.binary_fill_holes(largest)


def border(mask, eps=1e-6):
    """Thin (1 px) border of a binary mask via Sobel on the mask.

    Sobel responds on both sides of the edge (2 px wide); intersecting
    with the mask keeps only the inner pixel row.
    """
    mag, _, _ = sobel(mask.astype(float))
    return (mag > eps) & mask
