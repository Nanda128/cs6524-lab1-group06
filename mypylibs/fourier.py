import numpy as np


def dominant(spectrum, percent):
    keep = len(spectrum) * percent // 100
    return np.argsort(np.abs(spectrum))[::-1][:keep]


def reconstruct(z, percent):
    spectrum = np.fft.fft(z)
    chosen = dominant(spectrum, percent)
    kept = np.zeros_like(spectrum)
    kept[chosen] = spectrum[chosen]
    return np.fft.ifft(kept)


def epicycles(z, percent):
    spectrum = np.fft.fft(z) / len(z)
    chosen = dominant(spectrum, percent)
    return spectrum[chosen], np.fft.fftfreq(len(z), 1 / len(z))[chosen]


if __name__ == '__main__':
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    circle = 5 + 3j + 10 * np.exp(1j * t)
    assert np.allclose(reconstruct(circle, 2), circle)  # offset + one harmonic

    noisy = circle + np.random.default_rng(0).normal(size=100)
    assert np.allclose(reconstruct(noisy, 100), noisy)

    # the tip of the stacked rotating vectors must trace the reconstruction
    amplitudes, speeds = epicycles(noisy, 20)
    tips = [(amplitudes * np.exp(2j * np.pi * speeds * n / 100)).sum() for n in range(100)]
    assert np.allclose(tips, reconstruct(noisy, 20))
