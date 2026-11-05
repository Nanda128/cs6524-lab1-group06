
import yaml
import numpy as np
import matplotlib.pyplot as plt

from itertools import count
from PIL import Image
from os import path as opath
from preprocessing import edges

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'
MAX_PIXEL_VALUE = 255
PANEL_INCHES = 4


def load_params(case: int) -> dict:
    params_path = opath.join(ROOT_IN, f'{case:02}.yaml')
    if not opath.exists(params_path):
        return {}
    with open(params_path) as f:
        return yaml.safe_load(f) or {}

def save_transforms(stages, output_path):
    figure, axes = plt.subplots(1, len(stages), figsize=(len(stages) * PANEL_INCHES, PANEL_INCHES))
    for ax, (title, image) in zip(axes, stages):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    figure.savefig(output_path, bbox_inches='tight')
    plt.close(figure)


def main(case: int):
    image = Image.open(opath.join(ROOT_IN, f'{case:02}.{EXT}')).convert('RGB')
    rgb = np.asarray(image, float) / MAX_PIXEL_VALUE

    stages = edges.preprocess(rgb, **load_params(case))
    save_transforms(stages, opath.join(ROOT_OUT, f'{case:02}_transforms.{EXT}'))


if __name__ == '__main__':

    for case in count():
        try:
            main(case)
        except FileNotFoundError:
            break
