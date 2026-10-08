# -*- coding: utf-8 -*-
"""Directional (elliptical) bias of the reconstructed star spectrum, before vs after the posaffine correction.
For each reconstruction (unfiltered phase, star annulus r = 15-200 px of the 512 crop, or the same physical radii in the
1000 crop) the power along -22 deg and along 68 deg (the axes of the former spurious astigmatism) is compared in
frequency bins 0.5-3.5 um^-1; the score is the rms of log(P(-22)/P(68)) (0 = isotropic).
usage: python d32a/aniso_compare_32a.py  -> d32a/aniso_compare_32a.json"""
import json, os
import numpy as np
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
PX = 0.0319
def score(img, cy, cx):
    N = img.shape[0]; yy, xx = np.mgrid[0:N, 0:N]; r = np.hypot(yy - cy, xx - cx)
    m = np.clip((r - 15) / 8, 0, 1) * np.clip((200 - r) / 15, 0, 1)
    q1 = np.fft.fftfreq(N, PX); QX, QY = np.meshgrid(q1, q1); Q = np.hypot(QX, QY); PH = np.degrees(np.arctan2(QY, QX))
    F = np.abs(np.fft.fft2((img - np.sum(img * m) / m.sum()) * m)) ** 2
    rat = []
    for lo in np.arange(0.5, 3.5, 0.25):
        a = (np.abs(((PH + 22 + 90) % 180) - 90) < 15) & (Q >= lo) & (Q < lo + 0.25)
        b = (np.abs(((PH - 68 + 90) % 180) - 90) < 15) & (Q >= lo) & (Q < lo + 0.25)
        rat.append(np.median(F[a]) / np.median(F[b]))
    rat = np.array(rat); return float(np.sqrt(np.mean(np.log(rat) ** 2))), rat.tolist()
out = {}
for ds in ("32abf", "32abfaff", "32abfaff2", "32adf", "32adfaff", "32adfaff2"):
    for g in (512, 1000):
        d = W + f"d32a/out_{ds}_{g}/"
        if not os.path.exists(d + "recs.npz"): continue
        z = np.load(d + "recs.npz"); S = json.load(open(d + "compare_summary.json")); cy, cx = S["star_centre_crop_px"]
        res = {}
        for k in z.files:
            if not k.endswith("|phi_unf"): continue
            sc, rat = score(z[k].astype(float), cy, cx); res[k.split("|")[0]] = dict(score=sc, ratio=rat)
        out[f"{ds}_{g}"] = res
json.dump(out, open(W + "d32a/aniso_compare_32a.json", "w"), indent=1)
for key, res in out.items():
    print(key, {k.replace(", plane-wave model", " pw").replace(", FOV model", " FOV"): round(v["score"], 2) for k, v in res.items()})
