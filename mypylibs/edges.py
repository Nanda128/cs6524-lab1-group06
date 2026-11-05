"""Canny-style edge detection built from linear filters + scipy.ndimage.

Pipeline: per-channel Gaussian blur -> colour gradient (Di Zenzo) ->
non-maximum suppression -> hysteresis thresholding -> gap closing ->
silhouette mask -> 1 px border.
"""
import numpy as np
from scipy import ndimage

from . import filters


def color_gradient(rgb, sigma):
    """Gradient magnitude and direction using all colour channels.

    Di Zenzo: build the 2x2 structure tensor summed over channels and take
    its dominant eigenvalue/orientation. Catches edges that are invisible
    in grayscale (e.g. red vs. green of equal luma).
    """
    rgb = np.asarray(rgb, dtype=float)[..., :3] / 255.0
    gxx = gyy = gxy = 0.0
    for c in range(3):
        ch = filters.gaussian_blur(rgb[..., c], sigma=sigma)
        _, gx, gy = filters.sobel(ch)
        gxx = gxx + gx * gx
        gyy = gyy + gy * gy
        gxy = gxy + gx * gy
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)  # gradient direction
    lam = 0.5 * (gxx + gyy + np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2))
    return np.sqrt(lam), theta


def non_max_suppression(mag, theta):
    """Keep only pixels that are local maxima along the gradient direction."""
    h, w = mag.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx = np.cos(theta)
    dy = np.sin(theta)
    # sample magnitude one pixel ahead and behind along the gradient
    ahead = ndimage.map_coordinates(mag, [yy + dy, xx + dx], order=1, mode='nearest')
    behind = ndimage.map_coordinates(mag, [yy - dy, xx - dx], order=1, mode='nearest')
    return np.where((mag >= ahead) & (mag >= behind), mag, 0.0)


def hysteresis(nms, low, high):
    """Keep weak edges (> low) only if connected to a strong one (> high)."""
    strong = nms > high
    weak = nms > low
    labels, n = ndimage.label(weak, structure=np.ones((3, 3)))
    if n == 0:
        return strong
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    return keep[labels]


def canny(rgb, sigma=2.0, low=0.1, high=0.3):
    """low/high are fractions of the maximum gradient after NMS."""
    mag, theta = color_gradient(rgb, sigma)
    nms = non_max_suppression(mag, theta)
    m = nms.max()
    if m == 0:
        return np.zeros_like(nms, dtype=bool), mag
    return hysteresis(nms, low * m, high * m), mag


def silhouette_from_edges(edges, close_radius=3):
    """Close gaps in an edge map and return a filled silhouette mask.

    Edges are dilated until gaps seal, interior is filled, the largest
    region kept, then eroded back so the outline doesn't grow outward.
    """
    st = ndimage.generate_binary_structure(2, 2)
    closed = ndimage.binary_dilation(edges, structure=st, iterations=close_radius)
    # Pad so shapes touching the frame can still be filled
    padded = np.pad(closed, 1, constant_values=False)
    filled = ndimage.binary_fill_holes(padded)[1:-1, 1:-1]
    filled = ndimage.binary_erosion(filled, structure=st, iterations=close_radius)
    return filters.clean_mask(filled)
