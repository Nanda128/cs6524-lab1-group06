from itertools import count
from os import makedirs, path as opath

import numpy as np
from PIL import Image

from mypylibs import edges, figures, trajectory

ROOT_IN = 'in'
ROOT_OUT = 'out'


def main(case: int):
    image = Image.open(opath.join(ROOT_IN, f'{case:02}.png')).convert('RGB')
    rgb = np.asarray(image, float) / 255

    stages, border = edges.preprocess(rgb)
    x, y = trajectory.calculate(border)
    z = x + 1j * y

    figures.save_transforms(stages, opath.join(ROOT_OUT, f'{case:02}_transforms.png'))
    figures.save_trajectory(z, opath.join(ROOT_OUT, f'{case:02}_trajectory.png'))
    figures.save_reconstructions(z, opath.join(ROOT_OUT, f'{case:02}_final.png'))
    figures.save_animation(z, opath.join(ROOT_OUT, f'{case:02}.mp4'))


if __name__ == '__main__':
    makedirs(ROOT_OUT, exist_ok=True)

    for case in count():
        if not opath.exists(opath.join(ROOT_IN, f'{case:02}.png')):
            break
        main(case)
