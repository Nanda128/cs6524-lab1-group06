from itertools import count
from os import makedirs, path as opath

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mypylibs import edges

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'
MAX_PIXEL_VALUE = 255
PANEL_INCHES = 4
PANELS_PER_ROW = 4


def save_transforms(stages, output_path):
    rows = int(np.ceil(len(stages) / PANELS_PER_ROW))
    figure, axes = plt.subplots(rows, PANELS_PER_ROW,
                                figsize=(PANELS_PER_ROW * PANEL_INCHES, rows * PANEL_INCHES))
    axes = axes.ravel()

    for ax, (title, image) in zip(axes, stages):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)

    for ax in axes:
        ax.axis('off')

    figure.savefig(output_path, bbox_inches='tight')
    plt.close(figure)


def main(case: int):
    image = Image.open(opath.join(ROOT_IN, f'{case:02}.{EXT}')).convert('RGB')
    rgb = np.asarray(image, float) / MAX_PIXEL_VALUE

    stages, border = edges.preprocess(rgb)
    save_transforms(stages, opath.join(ROOT_OUT, f'{case:02}_transforms.{EXT}'))


if __name__ == '__main__':
    makedirs(ROOT_OUT, exist_ok=True)

    for case in count():
        if not opath.exists(opath.join(ROOT_IN, f'{case:02}.{EXT}')):
            break
        main(case)
