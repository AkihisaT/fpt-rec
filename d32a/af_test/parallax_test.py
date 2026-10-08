# Which quadratic phase moves the image? Weak mixed object, coherent FPM image |F^-1[P(q) O^(q - k_n)]|^2 (30 keV, kc 2 um^-1).
# (i) pupil defocus z (exp(i pi lam z |q|^2) on the pupil) vs (ii) the same quadratic phase on the object spectrum.
# Shift = least-squares slope of the cross-spectrum phase (0.05-1 um^-1) relative to the in-focus image.
import json, os
import numpy as np
N = 256; dx = 0.0319; lam = 1.239842 / 30 * 1e-3; kc = 2.0; z = 1000.0
q = np.fft.fftfreq(N, dx); QY, QX = np.meshgrid(q, q, indexing="ij"); Q2 = QX ** 2 + QY ** 2
rng = np.random.default_rng(1)
f = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) * np.exp(-Q2 / 2))); f /= f.std()
O = np.exp(-0.02 * f + 1j * 0.1 * f); yy, xx = np.mgrid[0:N, 0:N] * dx
def img(Oh, kn, Wp):
    o = np.fft.ifft2(Oh) * np.exp(2j * np.pi * (kn[0] * yy + kn[1] * xx))
    return np.abs(np.fft.ifft2(np.fft.fft2(o) * (np.sqrt(Q2) < kc) * np.exp(1j * Wp))) ** 2
def shift(a, b):
    C = np.fft.fft2(a - a.mean()) * np.conj(np.fft.fft2(b - b.mean())); m = (np.sqrt(Q2) < 1.0) & (np.sqrt(Q2) > 0.05)
    w = np.sqrt(np.abs(C[m])); G = np.stack([QY[m], QX[m]], 1) * (-2 * np.pi)
    return np.linalg.lstsq(G * w[:, None], np.angle(C[m]) * w, rcond=None)[0] / dx
Oh = np.fft.fft2(O); out = []
for kn in ([0.0, 1.2], [1.0, 0.6], [-0.8, 0.0]):
    kn = np.array(kn); I0 = img(Oh, kn, 0 * Q2)
    sp = shift(img(Oh, kn, np.pi * lam * z * Q2), I0); so = shift(img(Oh * np.exp(1j * np.pi * lam * z * Q2), kn, 0 * Q2), I0)
    out.append(dict(k_um_inv=kn.tolist(), lam_z_k_px=(lam * z * kn / dx).tolist(), pupil_shift_px=sp.tolist(), object_shift_px=so.tolist()))
    print(out[-1])
json.dump(dict(z_um=z, results=out), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "parallax_test.json"), "w"), indent=1)
