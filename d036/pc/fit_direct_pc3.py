# -*- coding: utf-8 -*-
"""036 partial coherence, step 1e: collapse of the direct images with a COMMON illumination level (g_n = 1 for all
positions, as assumed by the dark-field preprocessing) and a smooth common envelope E (Gaussian sigma 40 binned px =
5.1 um).  D_n(x) = E(x) F(|k_n + dk_n - kappa x - k0| / kc).  dk_n: per-image offset (grid +-0.08 kc, ring 2 only;
ring 1 is calibrated).  Scans c; for each c reports the weighted log rms and the resulting F(r).
usage: python d036/pc/fit_direct_pc3.py -> d036/pc/direct_fit_common.json"""
import json, numpy as np
from scipy import ndimage as ndi
W = "./"
z = np.load(W + "d036/data/prep_036.npz"); D = z["D"].astype(np.float64)
cal = json.load(open(W + "d036/work_df/calibration.json"))
kn = np.array(cal["k_solver"]); ring = np.array(cal["ring"]); k0 = np.array(cal["k0"]); kc = 2.5
kap_nom = 1 / (0.041328e-3 * 0.75e6)
B = 4; N = 1024 // B; dx = 0.0319 * B
Db = D.reshape(44, N, B, N, B).mean((2, 4))
ref = np.median([np.percentile(ndi.gaussian_filter(Db[i], 2), 99) for i in np.where(ring == 1)[0]]); Db /= ref
y = (np.arange(N) - N / 2 + 0.5) * dx; Y, X = np.meshgrid(y, y, indexing="ij")
use = np.where(ring <= 2)[0]; Dm = Db[use]; R2 = ring[use] == 2
wt = np.clip(Dm, 0, None) / (np.clip(Dm, 0, None) + 3e-3); wt[Dm <= 1e-4] = 0
L = np.log(np.clip(Dm, 1e-4, None))
edges = np.arange(0.40, 2.00001, 0.005); rc = 0.5 * (edges[1:] + edges[:-1])
Ys, Xs = Y[::2, ::2], X[::2, ::2]
def fit(c, with_dk, n_round=4):
    kap = c * kap_nom; dk = np.zeros((len(use), 2)); lE = np.zeros((N, N)); lFi = np.zeros(len(rc))
    rmap = lambda j, off, YY, XX: np.hypot(kn[use[j], 0] + off[0] - k0[0] - kap * YY, kn[use[j], 1] + off[1] - k0[1] - kap * XX) / kc
    for rnd in range(n_round):
        R = np.stack([rmap(j, dk[j], Y, X) for j in range(len(use))]); b = np.clip(np.digitize(R, edges) - 1, 0, len(rc) - 1)
        for _ in range(4):
            res = L - lE[None]
            num = np.bincount(b.ravel(), (wt * res).ravel(), len(rc)); den = np.bincount(b.ravel(), wt.ravel(), len(rc))
            ok = den > 0; lFi = np.interp(np.arange(len(rc)), np.where(ok)[0], num[ok] / den[ok])
            res = L - lFi[b]
            lE = ndi.gaussian_filter((wt * res).sum(0), 40) / np.maximum(ndi.gaussian_filter(wt.sum(0), 40), 1e-9)
            lE -= np.average(lE, weights=wt[~R2].sum(0) + 1e-9)          # gauge: F carries the ring-1 level
        if not with_dk: break
        offs = np.arange(-0.08, 0.0801, 0.01) * kc
        for j in np.where(R2)[0]:
            Lj, wj, Ej = L[j, ::2, ::2], wt[j, ::2, ::2], lE[::2, ::2]
            e = [(float((wj * (Lj - Ej - np.interp(rmap(j, (oy, ox), Ys, Xs), rc, lFi)) ** 2).sum()), oy, ox) for oy in offs for ox in offs]
            dk[j] = min(e)[1:]
    R = np.stack([rmap(j, dk[j], Y, X) for j in range(len(use))])
    resid = L - lE[None] - np.interp(R, rc, lFi)
    rms = float(np.sqrt((wt * resid ** 2).sum() / wt.sum()))
    pr = {int(k): float(np.sqrt((wt * resid ** 2)[ring[use] == k].sum() / wt[ring[use] == k].sum())) for k in (1, 2)}
    return rms, pr, dict(lF=lFi, den=den, lE=lE, dk=dk, resid=resid, R=R)
out = dict(scan=[])
for c in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
    rms, pr, _ = fit(c, False); out["scan"].append(dict(c=c, rms=rms, per_ring=pr)); print("c", c, "rms", round(rms, 4), {k: round(v, 4) for k, v in pr.items()}, flush=True)
res_dk = {}
for c in (1.0, min(out["scan"], key=lambda d: d["rms"])["c"]):
    rms, pr, F = fit(c, True); res_dk[c] = F
    out[f"dk_c{c:g}"] = dict(rms=rms, per_ring=pr, r_over_kc=rc.tolist(), logF=F["lF"].tolist(), weight_per_bin=F["den"].tolist(),
                             dk_over_kc=(F["dk"] / kc).tolist())
    print("with ring-2 dk, c", c, "rms", round(rms, 4), {k: round(v, 4) for k, v in pr.items()}, "median |dk|", round(float(np.median(np.hypot(*F['dk'][R2].T)) / kc), 3), flush=True)
json.dump(out, open(W + "d036/pc/direct_fit_common.json", "w"), indent=1)
np.savez_compressed(W + "d036/pc/direct_fit_common_c1.npz", logE=res_dk[1.0]["lE"].astype(np.float32), resid=res_dk[1.0]["resid"].astype(np.float32), R=res_dk[1.0]["R"].astype(np.float32))
