# -*- coding: utf-8 -*-
"""036 partial coherence, step 1: effective pupil profile F(r) = (S * |P|^2)(r) from the direct (no-sample) images.

Geometric model of a direct image (solver frame, FOV curvature kappa = c * kappa_nom, sign -1 as in fit_kappa_direct.py):
    D_n(x) = g_n * E(x) * F( |k_n - kappa x - k0| / kc )
  g_n  illumination level of position n;  E(x) common illumination envelope (smooth);  F radial profile of the pupil
  intensity transmission convolved with the illumination angular distribution S (both unknown; only S*|P|^2 is seen).
Fitted in log space by alternating weighted least squares (F: 0.005 kc bins; E: Gaussian-smoothed, sigma 12 binned px;
g_n per image) for rings 1 and 2 (ring-3 direct frames define the subtracted background, so they carry no level).
A correct kappa makes all 25 images collapse onto one curve F(r): the weighted rms of the log residual is scanned over
c and over a radial scale of the ring-2 k vectors (stage->k extrapolation).  Grid: 4x4 binning (256 px of 127.6 nm).
usage: python d036/pc/fit_direct_pc.py  -> d036/pc/direct_fit.json, direct_fit.npz"""
import json, numpy as np
from scipy import ndimage as ndi
W = "./"
z = np.load(W + "d036/data/prep_036.npz"); D = z["D"].astype(np.float64)
cal = json.load(open(W + "d036/work_df/calibration.json"))
kn = np.array(cal["k_solver"]); ring = np.array(cal["ring"]); k0 = np.array(cal["k0"]); kc = 2.5
lam = 0.041328e-3; kap_nom = 1 / (lam * 0.75e6)
B = 4; N = 1024 // B; dx = 0.0319 * B
Db = D.reshape(44, N, B, N, B).mean((2, 4))
ref = np.median([np.percentile(ndi.gaussian_filter(Db[i], 2), 99) for i in np.where(ring == 1)[0]])
Db /= ref
y = (np.arange(N) - N / 2 + 0.5) * dx; Y, X = np.meshgrid(y, y, indexing="ij")
use = np.where(ring <= 2)[0]
Dm = Db[use]
wt = np.clip(Dm, 0, None) / (np.clip(Dm, 0, None) + 3e-3)          # down-weight near-zero (noise, background)
wt[Dm <= 1e-4] = 0.0
L = np.log(np.clip(Dm, 1e-4, None))
edges = np.arange(0.40, 2.00001, 0.005); rc = 0.5 * (edges[1:] + edges[:-1])

def radius(c, s2, use_k0=True):
    kk = kn[use].copy(); kk[ring[use] == 2] *= s2
    off = k0 if use_k0 else np.zeros(2)
    ky = kk[:, 0, None, None] - c * kap_nom * Y[None] - off[0]
    kx = kk[:, 1, None, None] - c * kap_nom * X[None] - off[1]
    return np.hypot(ky, kx) / kc

def fit(r, n_iter=12):
    b = np.clip(np.digitize(r, edges) - 1, 0, len(rc) - 1)
    lg = np.zeros(len(use)); lE = np.zeros((N, N)); lF = np.zeros(len(rc))
    for _ in range(n_iter):
        res = L - lg[:, None, None] - lE[None]
        num = np.bincount(b.ravel(), (wt * res).ravel(), len(rc)); den = np.bincount(b.ravel(), wt.ravel(), len(rc))
        lF = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        lFi = np.interp(np.arange(len(rc)), np.where(den > 0)[0], lF[den > 0])
        lF_px = lFi[b]
        res = L - lg[:, None, None] - lF_px
        lE = ndi.gaussian_filter((wt * res).sum(0), 12) / np.maximum(ndi.gaussian_filter(wt.sum(0), 12), 1e-9)
        lE -= lE.mean()
        res = L - lE[None] - lF_px
        lg = (wt * res).sum((1, 2)) / wt.sum((1, 2))
        lg -= np.median(lg[ring[use] == 1])
    resid = L - lg[:, None, None] - lE[None] - lF_px
    rms = float(np.sqrt((wt * resid ** 2).sum() / wt.sum()))
    per_ring = {int(R): float(np.sqrt((wt * resid ** 2)[ring[use] == R].sum() / wt[ring[use] == R].sum())) for R in (1, 2)}
    return rms, per_ring, dict(lF=lFi, den=den, lE=lE, lg=lg, resid=resid)

scan = []
for c in np.arange(0.0, 2.51, 0.25):
    rms, pr, _ = fit(radius(c, 1.0)); scan.append(dict(c=float(c), s2=1.0, k0=True, rms=rms, per_ring=pr))
cb = min(scan, key=lambda d: d["rms"])["c"]
fine = []
for c in np.arange(max(cb - 0.3, 0), cb + 0.301, 0.05):
    for s2 in (0.97, 0.985, 1.0, 1.015, 1.03):
        rms, pr, _ = fit(radius(c, s2)); fine.append(dict(c=round(float(c), 3), s2=s2, k0=True, rms=rms, per_ring=pr))
best = min(fine, key=lambda d: d["rms"])
rms0, pr0, _ = fit(radius(best["c"], best["s2"], use_k0=False))
rms, pr, F = fit(radius(best["c"], best["s2"]), n_iter=25)
print("coarse c scan (c, rms):", [(d["c"], round(d["rms"], 4)) for d in scan])
print("best:", best, "| without k0:", round(rms0, 4))
json.dump(dict(kappa_nom=kap_nom, ref_level=float(ref), coarse=scan, fine=fine, best=best, rms_without_k0=rms0,
               r_over_kc=rc.tolist(), logF=F["lF"].tolist(), weight_per_bin=F["den"].tolist(), log_g=F["lg"].tolist(), images=use.tolist()),
          open(W + "d036/pc/direct_fit.json", "w"), indent=1)
np.savez_compressed(W + "d036/pc/direct_fit.npz", logE=F["lE"].astype(np.float32), resid=F["resid"].astype(np.float32),
                    r=radius(best["c"], best["s2"]).astype(np.float32), Dm=Dm.astype(np.float32), wt=wt.astype(np.float32))
