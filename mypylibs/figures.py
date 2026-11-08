import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from mypylibs.fourier import epicycles, reconstruct

PANEL_INCHES = 3
VIDEO_PERCENT = 20
VIDEO_FRAMES = 250
VIDEO_FPS = 25
VIDEO_DPI = 100
MIN_DRAWN_RADIUS = 1  # smaller circles are invisible at video resolution and slow every frame
UNIT_CIRCLE = np.exp(2j * np.pi * np.linspace(0, 1, 100))

try:
    import imageio_ffmpeg
    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:  # the base image lacks it, and the required figures must still be produced
    imageio_ffmpeg = None


def grid(panels, columns):
    rows = int(np.ceil(panels / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * PANEL_INCHES, rows * PANEL_INCHES))
    return figure, axes.ravel()


def save(figure, axes, path):
    for ax in axes:
        ax.axis('off')
    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)


def frame_curve(ax, z):
    margin = 0.1 * max(np.ptp(z.real), np.ptp(z.imag))  # low-harmonic curves overshoot the outline
    ax.set(xlim=(z.real.min() - margin, z.real.max() + margin),
           ylim=(z.imag.max() + margin, z.imag.min() - margin), aspect='equal')


def save_transforms(stages, path):
    figure, axes = grid(len(stages), 4)
    for ax, (title, image) in zip(axes, stages):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
    save(figure, axes, path)


def save_trajectory(z, path):
    figure, axes = grid(10, 5)
    closed = np.append(z, z[0])
    for ax, fraction in zip(axes, np.linspace(0, 1, 10)):
        end = 1 + int(fraction * len(z))
        ax.plot(closed[:end].real, closed[:end].imag)
        ax.plot(closed[end - 1].real, closed[end - 1].imag, 'r.')
        frame_curve(ax, z)
        ax.set_title(f't = {fraction:.2f} T')
    save(figure, axes, path)


def save_reconstructions(z, path):
    figure, axes = grid(20, 5)
    for ax, percent in zip(axes, range(5, 101, 5)):
        curve = reconstruct(z, percent)
        curve = np.append(curve, curve[0])
        ax.plot(curve.real, curve.imag)
        frame_curve(ax, z)
        ax.set_title(f'{percent}% of harmonics')
    save(figure, axes, path)


def circle_outlines(centres, radii):
    big = radii > MIN_DRAWN_RADIUS
    rings = centres[big, None] + radii[big, None] * UNIT_CIRCLE
    return np.column_stack((rings, np.full(len(rings), np.nan))).ravel()  # NaN breaks the line between circles


def save_animation(z, path):
    if imageio_ffmpeg is None:
        print(f'imageio-ffmpeg not installed, skipping {path}')
        return

    figure, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    frame_curve(ax, z)
    ax.set_title(f'{VIDEO_PERCENT}% of harmonics')

    amplitudes, speeds = epicycles(z, VIDEO_PERCENT)
    radii = np.where(speeds == 0, 0, np.abs(amplitudes))  # the constant term only sets the position
    curve = reconstruct(z, VIDEO_PERCENT)
    curve = np.append(curve, curve[0])
    circles, = ax.plot([], [], lw=0.5, color='gray')
    arms, = ax.plot([], [], lw=0.8, color='black')
    trace, = ax.plot([], [], lw=2, color='tab:blue')

    writer = FFMpegWriter(fps=VIDEO_FPS, extra_args=['-pix_fmt', 'yuv420p'])
    with writer.saving(figure, path, VIDEO_DPI):
        for frame in range(VIDEO_FRAMES + 1):
            end = frame * len(z) // VIDEO_FRAMES
            arrows = amplitudes * np.exp(2j * np.pi * speeds * end / len(z))
            tips = np.cumsum(arrows)
            outlines = circle_outlines(tips - arrows, radii)
            circles.set_data(outlines.real, outlines.imag)
            arms.set_data(tips.real, tips.imag)
            trace.set_data(curve[:end + 1].real, curve[:end + 1].imag)
            writer.grab_frame()
    plt.close(figure)
