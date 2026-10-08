"""Estimate the FOV-effect curvature kappa (O_eff = O * exp(i pi kappa |x|^2), solver frame) from the vignetting of
the direct (no-sample) images: model D_n = |F^-1{ P(k) F{exp(i 2 pi k_n x) Q(x)} }|^2 with a hard NA disc P (radius
s_p * kc), compared with the measured median direct images (ring 1 and 2) as bright-region masks (IoU) and as
normalised profiles.  k_n from the ring-1 calibration.  Grid: 512 px of 63.8 nm (2x binning)."""
import json, sys, numpy as np
from scipy import ndimage as ndi
z = np.load("d036/data/prep_036.npz"); D = z["D"]
ill = json.load(open("d036/data/illumination_036.json")); kc = ill["kc"]
kn = np.array([o["k"] for o in ill["positions"]])
lam, p_nom = 0.041328e-3, 0.75e6                     # um
kap_nom = 1 / (lam * p_nom)
N, dx = 512, 0.0638
Db = D[:, ::2, ::2] + D[:, 1::2, ::2] + D[:, ::2, 1::2] + D[:, 1::2, 1::2]
ref = np.median([np.median(Db[p]) for p in range(9)])
y = (np.arange(N) - N / 2) * dx; Y, X = np.meshgrid(y, y, indexing="ij")
f = np.fft.fftfreq(N, dx); KY, KX = np.meshgrid(f, f, indexing="ij")
use = list(range(25))
meas = [ndi.uniform_filter(Db[p], 9) / ref > 0.5 for p in use]
def model_masks(kap, sp, sgn):
    out = []
    for p in use:
        E = np.exp(2j * np.pi * (kn[p, 0] * Y + kn[p, 1] * X) + 1j * np.pi * sgn * kap * (X ** 2 + Y ** 2))
        I = np.abs(np.fft.ifft2(np.fft.fft2(E) * (np.hypot(KY, KX) <= sp * kc))) ** 2
        out.append(ndi.uniform_filter(I, 9) > 0.5)
    return out
def iou(ms):
    inter = sum((a & b).sum() for a, b in zip(ms, meas)); uni = sum((a | b).sum() for a, b in zip(ms, meas))
    return inter / uni
res = []
for sgn in (+1, -1):
    for c in np.arange(0.0, 3.01, 0.25):
        res.append((sgn, c, 1.0, iou(model_masks(c * kap_nom, 1.0, sgn))))
best = max(res, key=lambda t: t[3])
fine = []
for c in np.arange(max(best[1] - 0.3, 0), best[1] + 0.31, 0.05):
    for sp in (0.97, 1.0, 1.03):
        fine.append((best[0], round(float(c), 3), sp, iou(model_masks(c * kap_nom, sp, best[0]))))
bf = max(fine, key=lambda t: t[3])
print("coarse (sign, c, pupil scale, IoU):", [(s, round(float(c), 2), round(v, 3)) for s, c, _, v in res])
print("best fine:", bf)
json.dump(dict(kappa_nom=kap_nom, coarse=[[int(s), float(c), float(v)] for s, c, _, v in res], fine=[[int(s), c, sp, float(v)] for s, c, sp, v in fine],
               best=dict(sign=int(bf[0]), c=bf[1], pupil_scale=bf[2], iou=float(bf[3]))), open("d036/data/kappa_direct_fit.json", "w"), indent=1)

# ring-2 only (edge images carry the curvature information)
use = list(range(9, 25)); meas = [ndi.uniform_filter(Db[p], 9) / ref > 0.5 for p in use]
cov_meas = np.mean([m.mean() for m in meas])
r2 = []
for c in np.arange(0.5, 4.01, 0.25):
    for sp in (1.0, 1.05, 1.1):
        ms = model_masks(c * kap_nom, sp, -1)
        r2.append((round(float(c), 2), sp, round(float(iou(ms)), 3), round(float(np.mean([m.mean() for m in ms])), 3)))
print("measured ring-2 bright coverage (>0.5 of ring-1 level):", round(float(cov_meas), 3))
print("ring-2 (c, pupil scale, IoU, coverage):", [t for t in r2 if t[2] > 0.3 or t[0] in (1.0, 2.0, 3.0)])
