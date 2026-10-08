# -*- coding: utf-8 -*-
"""036 partial coherence, step 1b: split the effective pupil profile F(r) (direct_fit.json) into a hard pupil disc of
radius r_p (units of kc) and a radially symmetric illumination angular distribution S with unit integral.
For a radially symmetric S, (S * disc)(r) = sum_j s_j * frac(r, rho_j, r_p), where s_j is the fraction of S in the
annulus rho_j and frac is the fraction of the circle of radius rho_j (centred at distance r from the pupil centre) that
lies inside the disc.  s_j >= 0 is solved by NNLS with a second-difference smoothness penalty on log-spaced annuli,
for each r_p on a grid; the fitted quantity is F(r) / A over the edge and tail (r = 0.9-1.85), weighted in log.
Coherent alternative with the same F: delta-like S and a soft pupil |P(r)|^2 = F(r)/A (not distinguishable here).
usage: python d036/pc/deconv_S.py -> d036/pc/S_fit.json"""
import json, numpy as np
from scipy.optimize import nnls
W = "./"
fj = json.load(open(W + "d036/pc/direct_fit.json"))
r = np.array(fj["r_over_kc"]); lF = np.array(fj["logF"]); den = np.array(fj["weight_per_bin"])
m = (den > 200) & (r > 0.60) & (r < 1.85)
rr, yy = r[m], np.exp(lF[m])
rho = np.concatenate([[0.0], np.geomspace(0.004, 1.2, 60)])          # annulus radii (kc)

def frac(r_, rho_, rp):
    if rho_ == 0: return (r_ <= rp).astype(float)
    c = (r_ ** 2 + rho_ ** 2 - rp ** 2) / (2 * r_ * rho_)
    return np.where(c <= -1, 1.0, np.where(c >= 1, 0.0, np.arccos(np.clip(c, -1, 1)) / np.pi))

def solve(rp, lam=3e-2):
    M = np.stack([frac(rr, q, rp) for q in rho], 1)
    # relative (log-like) weighting: divide rows by the data value
    wrow = 1.0 / np.maximum(yy, 1e-3)
    A_ = M * wrow[:, None]; b_ = yy * wrow
    Dm = np.diff(np.eye(len(rho)), 2, axis=0)
    Aa = np.vstack([A_, np.sqrt(lam) * Dm * 10]); ba = np.concatenate([b_, np.zeros(len(Dm))])
    s, _ = nnls(Aa, ba, maxiter=5000)
    fit = M @ s
    return s, fit, float(np.sqrt(np.mean(np.log(np.maximum(fit, 1e-6) / yy) ** 2)))

scan = []
for rp in np.arange(0.96, 1.141, 0.01):
    s, fit, e = solve(rp); scan.append((round(float(rp), 3), e, float(s.sum())))
rp = min(scan, key=lambda t: t[1])[0]
s, fit, e = solve(rp)
A = s.sum(); S = s / A                                                  # unit integral; F plateau level = A
enc = np.cumsum(S)
out = dict(model="hard pupil disc r_p + radially symmetric S (NNLS on annuli); lengths in kc", r_p=rp, level_A=float(A),
           rms_log=e, rp_scan=scan, rho=rho.tolist(), S_annulus_fraction=S.tolist(),
           encircled={f"{R:g}": float(np.interp(R, rho, enc)) for R in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)},
           r_fit=rr.tolist(), F_data=yy.tolist(), F_model=fit.tolist())
json.dump(out, open(W + "d036/pc/S_fit.json", "w"), indent=1)
print("r_p scan (r_p, rms_log):", [(a, round(b, 4)) for a, b, _ in scan])
print("r_p", rp, "A", round(A, 4), "rms_log", round(e, 4), "encircled:", {k: round(v, 3) for k, v in out["encircled"].items()})
