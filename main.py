import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from os import path as opath
from mypylibs import filters, edges as cedges

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'

DEFAULTS = {
    'method': 'canny',    # 'canny' (colour gradient + NMS + hysteresis) or 'threshold'
    'low': 0.1,           # canny: weak-edge level, fraction of max gradient
    'high': 0.3,          # canny: strong-edge level, fraction of max gradient
    'close': 3,           # canny: gap-closing radius (pixels)
    'sigma': 2.0,         # Gaussian std-dev (pixels)
    'threshold': None,    # None -> Otsu
    'invert': None,       # None -> auto; True: dark object, False: bright object
    'clean': True,        # keep largest blob + fill holes
}


def load_params(case: int):
    params = dict(DEFAULTS)
    cfg = opath.join(ROOT_IN, f'{case:02}.yaml')
    if opath.exists(cfg):
        with open(cfg) as f:
            params.update(yaml.safe_load(f) or {})
    return params


def preprocess_canny(img, params):
    stages = {}
    rgb = np.asarray(img.convert('RGB'))
    stages['Original'] = rgb
    mag, _ = cedges.color_gradient(rgb, params['sigma'])
    stages[f"Colour gradient magnitude (sigma={params['sigma']})"] = mag
    raw, _ = cedges.canny(rgb, params['sigma'], params['low'], params['high'])
    stages[f"NMS + hysteresis ({params['low']}/{params['high']})"] = raw
    mask = cedges.silhouette_from_edges(raw, params['close'])
    stages[f"Gaps closed (r={params['close']}), filled, largest blob"] = mask
    edges = filters.border(mask)
    stages['Thin border mask'] = edges
    return stages, edges


def preprocess(img, params):
    if params['method'] == 'canny':
        return preprocess_canny(img, params)
    return preprocess_threshold(img, params)


def preprocess_threshold(img, params):
    """Grayscale -> blur -> threshold -> Sobel border."""
    stages = {}
    stages['Original'] = np.asarray(img.convert('RGB'))
    gray = filters.to_grayscale(stages['Original'])
    stages['Grayscale'] = gray
    blur = filters.gaussian_blur(gray, sigma=params['sigma'])
    stages[f"Gaussian blur (sigma={params['sigma']})"] = blur
    mask = filters.threshold(blur, params['threshold'], params['invert'])
    t = params['threshold'] if params['threshold'] is not None else 'Otsu'
    stages[f'Threshold ({t})'] = mask
    if params['clean']:
        mask = filters.clean_mask(mask)
        stages['Largest blob, holes filled'] = mask
    mag, _, _ = filters.sobel(mask.astype(float))
    stages['Sobel gradient magnitude'] = mag
    edges = filters.border(mask)
    stages['Thin border mask'] = edges
    return stages, edges


def save_transforms(stages, case):
    n = len(stages)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, im) in zip(axes, stages.items()):
        ax.imshow(im, cmap=None if im.ndim == 3 else 'gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(opath.join(ROOT_OUT, f'{case:02}_transforms.{EXT}'), dpi=120)
    plt.close(fig)


def main(case: int):
    img = Image.open(opath.join(ROOT_IN, f'{case:02}.{EXT}'))
    params = load_params(case)

    stages, edges = preprocess(img, params)
    save_transforms(stages, case)

    # TODO: trajectory -> XX_trajectory.png
    # TODO: Fourier reconstruction -> XX_final.png
    # (edge_y, edge_x) = np.nonzero(edges)


if __name__ == '__main__':

    case = 0
    while True:
        try:
            main(case)
        except FileNotFoundError:
            break
        case += 1
