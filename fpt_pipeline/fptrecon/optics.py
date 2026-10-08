"""Fourier-optics helpers: pupils, Fourier resampling, band limits, Zernike fits."""
from math import factorial

import numpy as np
from scipy.fft import next_fast_len


def kgrid(M, dk):
    yy, xx = np.mgrid[-(M // 2):M - M // 2, -(M // 2):M - M // 2] * dk
    return yy, xx


def pupil_fn(k, kc, z=0.0, astig=(0.0, 0.0), lam=None, edge=0.05):
    """Disc pupil (tanh edge) with defocus z and astigmatism (a0: kx^2-ky^2, a1: 2kxky), all in um.
    k: (..., 2) = (row, col) spatial frequency in um^-1. W = pi*lam*z*|k|^2."""
    ky, kx = k[..., 0], k[..., 1]
    r2 = ky ** 2 + kx ** 2
    W = np.pi * lam * z * r2 + np.pi * lam * (astig[0] * (kx ** 2 - ky ** 2) + astig[1] * 2 * kx * ky)
    amp = 0.5 * (1 - np.tanh((np.sqrt(r2) - kc) / edge))
    return amp * np.exp(1j * W)


def analytic_W(M, dk, lam, z, astig):
    yy, xx = kgrid(M, dk)
    return np.pi * lam * z * (yy ** 2 + xx ** 2) + np.pi * lam * (astig[0] * (xx ** 2 - yy ** 2) + astig[1] * 2 * xx * yy)


def fcrop_intensity(I, M):
    """Band-limit / downsample real images (..., N, N) -> (..., M, M) by Fourier cropping (values preserved)."""
    N = I.shape[-1]
    F = np.fft.fftshift(np.fft.fft2(I, norm="ortho"), axes=(-2, -1))
    c, h = N // 2, M // 2
    Fc = F[..., c - h:c - h + M, c - h:c - h + M]
    return (np.fft.ifft2(np.fft.ifftshift(Fc, axes=(-2, -1)), norm="ortho").real * (M / N)).astype(np.float32)


def fresample(img, Nout):
    """Fourier resample a real image (N, N) -> (Nout, Nout), preserving values."""
    N = img.shape[-1]
    F = np.fft.fftshift(np.fft.fft2(img, norm="ortho"))
    if Nout <= N:
        c, h = N // 2, Nout // 2
        Fo = F[c - h:c - h + Nout, c - h:c - h + Nout]
    else:
        Fo = np.zeros((Nout, Nout), complex)
        c, h = Nout // 2, N // 2
        Fo[c - h:c - h + N, c - h:c - h + N] = F
    return (np.fft.ifft2(np.fft.ifftshift(Fo), norm="ortho") * (Nout / N)).real


def band_limit(img, dx, kmin=0.3, kmax=3.5, taper_lo=0.1, taper_hi=0.3):
    """Keep the spatial-frequency band [kmin, kmax] (um^-1) with raised-cosine edges."""
    N = img.shape[-1]
    k = np.fft.fftfreq(N, d=dx)
    KY, KX = np.meshgrid(k, k, indexing="ij")
    KR = np.hypot(KY, KX)
    fh = np.clip((kmax + taper_hi - KR) / (2 * taper_hi), 0, 1)
    fh = 0.5 - 0.5 * np.cos(np.pi * fh)
    if kmin > 0:
        fl = np.clip((KR - (kmin - taper_lo)) / (2 * taper_lo), 0, 1)
        fl = 0.5 - 0.5 * np.cos(np.pi * fl)
    else:
        fl = 1.0
    return np.fft.ifft2(np.fft.fft2(img) * fh * fl).real


def flip_k(A):
    """A(k) -> A(-k) on an even, FFT-centred grid (centre index M//2)."""
    return np.roll(A[::-1, ::-1], 1, axis=(0, 1))


def fast_even(n):
    n = int(np.ceil(n))
    while True:
        m = next_fast_len(n)
        if m % 2 == 0:
            return m
        n = m + 1


# ---- Zernike (Noll-normalised, unit disc) -------------------------------------------------
ZLIST = [(0, 0, "piston"), (1, 1, "tilt x"), (1, -1, "tilt y"), (2, 0, "defocus"),
         (2, 2, "astig 0deg"), (2, -2, "astig 45deg"), (3, 1, "coma x"), (3, -1, "coma y"),
         (3, 3, "trefoil x"), (3, -3, "trefoil y"), (4, 0, "spherical"),
         (4, 2, "2nd astig 0deg"), (4, -2, "2nd astig 45deg")]


def zernike_nm(n, m, rho, th):
    R = np.zeros_like(rho)
    for s in range((n - abs(m)) // 2 + 1):
        R += ((-1) ** s * factorial(n - s) /
              (factorial(s) * factorial((n + abs(m)) // 2 - s) * factorial((n - abs(m)) // 2 - s)) * rho ** (n - 2 * s))
    norm = np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)
    return norm * R * (np.cos(m * th) if m >= 0 else np.sin(-m * th))


def zernike_fit(W, dk, kc):
    """Least-squares Zernike fit of a pupil phase map W (M x M, centred) over |k| <= kc.
    Returns coefficients (rad RMS, order of ZLIST) and the fitted map."""
    M = W.shape[-1]
    yy, xx = kgrid(M, dk)
    rho = np.hypot(yy, xx) / kc
    th = np.arctan2(yy, xx)
    m = rho <= 1.0
    B = np.stack([zernike_nm(n, mm, rho[m], th[m]) for n, mm, _ in ZLIST], 1)
    coef, *_ = np.linalg.lstsq(B, W[m], rcond=None)
    fit = np.full(W.shape, np.nan)
    fit[m] = B @ coef
    return coef, fit
