"""Rotationally symmetric quadratic fits of the free pupils of the posaffine check (aff, aff2) over annuli, and the radial
profile (piston, tilt, astigmatism removed). Input d32a/af_test/pupil_check_{aff,aff2}.npz (pupil_basis_test.py).
usage: python d32a/af_test/pupil_annulus_quadratic.py -> d32a/af_test/pupil_annulus_quadratic.json"""
import json, sys, numpy as np
sys.path.insert(0, "fpt_pipeline"); from fptrecon.optics import kgrid
lam, kc = 4.1328e-5, 2.0; out = {}
for tag in ("aff", "aff2"):
    z = np.load(f"d32a/af_test/pupil_check_{tag}.npz"); W, Pamp, dk = z["W"], z["Pamp"], float(z["dk"])
    yy, xx = kgrid(W.shape[0], dk); r = np.hypot(yy, xx); fits = []
    for lo, hi in ((0.25, 1.9), (0.5, 1.9), (0.86, 1.75), (0.3, 1.2)):
        m = np.isfinite(W) & (Pamp > 0.5) & (r >= lo) & (r <= hi)
        X = np.stack([np.ones(m.sum()), yy[m], xx[m], (xx ** 2 - yy ** 2)[m], (2 * xx * yy)[m], (yy ** 2 + xx ** 2)[m]], 1)
        c = np.linalg.lstsq(X, W[m], rcond=None)[0]
        fits.append(dict(r_min=lo, r_max=hi, z_um=float(c[5] / (np.pi * lam)), astig_um=[float(c[3] / (np.pi * lam)), float(c[4] / (np.pi * lam))]))
    m = np.isfinite(W) & (Pamp > 0.5) & (r < 0.95 * kc)
    X = np.stack([np.ones(m.sum()), yy[m], xx[m], (xx ** 2 - yy ** 2)[m], (2 * xx * yy)[m], (yy ** 2 + xx ** 2)[m]], 1)
    c = np.linalg.lstsq(X, W[m], rcond=None)[0]; Wd = np.full_like(W, np.nan); Wd[m] = W[m] - X[:, :5] @ c[:5]
    e = np.arange(0, 0.96 * kc, 0.05); rc = 0.5 * (e[1:] + e[:-1])
    pr = np.array([np.nanmean(Wd[(r >= a) & (r < b) & m]) for a, b in zip(e[:-1], e[1:])])
    out[tag] = dict(annulus_quadratic=fits, central_step_rad=float(pr[(rc > 0.2) & (rc < 0.3)].max() - pr[0]), profile_r=rc.tolist(), profile_rad=pr.tolist(),
                    note="rotationally symmetric quadratic + piston + tilt + astigmatism fitted over r_min<=|k|<=r_max (um^-1); central_step = max phase at 0.2-0.3 um^-1 minus the first bin (0-0.05)")
    print(tag, [round(f["z_um"]) for f in fits], round(out[tag]["central_step_rad"], 2))
json.dump(out, open("d32a/af_test/pupil_annulus_quadratic.json", "w"), indent=1)
