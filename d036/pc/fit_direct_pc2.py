# -*- coding: utf-8 -*-
"""036 partial coherence, step 1c: refined collapse of the direct images at a fixed FOV strength c (default 1, as in the
reconstructions).  Per image n: illumination offset dk_n (grid search, +-0.06 kc) and a planar log-intensity tilt
(log g_n + a_n y + b_n x, linear LS) on top of the common envelope E(x) and the common profile F(r); alternating rounds.
Then S and r_p are re-derived with the NNLS decomposition of deconv_S.py.
usage: python d036/pc/fit_direct_pc2.py [c] -> d036/pc/direct_fit_c<c>.json, S_fit_c<c>.json"""
import json, sys, numpy as np
from scipy import ndimage as ndi
from scipy.optimize import nnls
W = "./"
c = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
z = np.load(W + "d036/data/prep_036.npz"); D = z["D"].astype(np.float64)
cal = json.load(open(W + "d036/work_df/calibration.json"))
kn = np.array(cal["k_solver"]); ring = np.array(cal["ring"]); kc = 2.5
kap = c / (0.041328e-3 * 0.75e6)
B = 4; N = 1024 // B; dx = 0.0319 * B
Db = D.reshape(44, N, B, N, B).mean((2, 4))
ref = np.median([np.percentile(ndi.gaussian_filter(Db[i], 2), 99) for i in np.where(ring == 1)[0]]); Db /= ref
y = (np.arange(N) - N / 2 + 0.5) * dx; Y, X = np.meshgrid(y, y, indexing="ij")
use = np.where(ring <= 2)[0]; Dm = Db[use]
wt = np.clip(Dm, 0, None) / (np.clip(Dm, 0, None) + 3e-3); wt[Dm <= 1e-4] = 0
L = np.log(np.clip(Dm, 1e-4, None))
edges = np.arange(0.40, 2.00001, 0.005); rc = 0.5 * (edges[1:] + edges[:-1])
Ys, Xs = Y[::2, ::2], X[::2, ::2]                                  # coarse grid for the per-image search
dk = np.zeros((len(use), 2)); tilt = np.zeros((len(use), 3)); lE = np.zeros((N, N)); lFi = np.zeros(len(rc))
def rmap(j, off, YY, XX):
    k = kn[use[j]] + off
    return np.hypot(k[0] - kap * YY, k[1] - kap * XX) / kc
def Fint(rv): return np.interp(rv, rc, lFi)
basis = lambda YY, XX: np.stack([np.ones_like(YY), YY / 16, XX / 16])
Bf, Bs = basis(Y, X), basis(Ys, Xs)
hist = []
for rnd in range(6):
    # (1) common F from all images
    R = np.stack([rmap(j, dk[j], Y, X) for j in range(len(use))])
    b = np.clip(np.digitize(R, edges) - 1, 0, len(rc) - 1)
    res = L - lE[None] - np.einsum("jk,kyx->jyx", tilt, Bf)
    num = np.bincount(b.ravel(), (wt * res).ravel(), len(rc)); den = np.bincount(b.ravel(), wt.ravel(), len(rc))
    ok = den > 0; lFi = np.interp(np.arange(len(rc)), np.where(ok)[0], (num[ok] / den[ok]))
    # (2) common envelope
    res = L - np.einsum("jk,kyx->jyx", tilt, Bf) - lFi[b]
    lE = ndi.gaussian_filter((wt * res).sum(0), 24) / np.maximum(ndi.gaussian_filter(wt.sum(0), 24), 1e-9); lE -= lE.mean()
    # (3) per image: dk grid search with LS tilt
    offs = np.arange(-0.06, 0.0601, 0.01) * kc
    for j in range(len(use)):
        Lj, wj, Ej = L[j, ::2, ::2], wt[j, ::2, ::2], lE[::2, ::2]
        best = None
        for oy in offs:
            for ox in offs:
                rj = rmap(j, np.array([oy, ox]), Ys, Xs); t = Lj - Ej - Fint(rj)
                A_ = (Bs * np.sqrt(wj)).reshape(3, -1).T; coef = np.linalg.lstsq(A_, (t * np.sqrt(wj)).ravel(), rcond=None)[0]
                e = float((wj * (t - np.einsum("k,kyx->yx", coef, Bs)) ** 2).sum())
                if best is None or e < best[0]: best = (e, oy, ox, coef)
        dk[j] = best[1:3]; tilt[j] = best[3]
    tilt[:, 0] -= np.median(tilt[ring[use] == 1, 0])
    R = np.stack([rmap(j, dk[j], Y, X) for j in range(len(use))])
    resid = L - lE[None] - np.einsum("jk,kyx->jyx", tilt, Bf) - Fint(R)
    rms = float(np.sqrt((wt * resid ** 2).sum() / wt.sum())); hist.append(rms)
    print(f"round {rnd}: rms_log {rms:.4f}, |dk| median {np.median(np.hypot(*dk.T)) / kc:.3f} kc", flush=True)
# NNLS decomposition (as deconv_S.py)
m = (den > 200) & (rc > 0.60) & (rc < 1.85); rr, yy = rc[m], np.exp(lFi[m])
rho = np.concatenate([[0.0], np.geomspace(0.004, 1.2, 60)])
def frac(r_, q, rp):
    if q == 0: return (r_ <= rp).astype(float)
    cc = (r_ ** 2 + q ** 2 - rp ** 2) / (2 * r_ * q)
    return np.where(cc <= -1, 1.0, np.where(cc >= 1, 0.0, np.arccos(np.clip(cc, -1, 1)) / np.pi))
def solve(rp, lam=3e-2):
    M = np.stack([frac(rr, q, rp) for q in rho], 1); wr = 1 / np.maximum(yy, 1e-3)
    Dd = np.diff(np.eye(len(rho)), 2, axis=0)
    s, _ = nnls(np.vstack([M * wr[:, None], np.sqrt(lam) * Dd * 10]), np.concatenate([yy * wr, np.zeros(len(Dd))]), maxiter=5000)
    fit = M @ s; return s, fit, float(np.sqrt(np.mean(np.log(np.maximum(fit, 1e-6) / yy) ** 2)))
scan = [(round(float(rp), 3),) + solve(rp)[2:] for rp in np.arange(0.96, 1.141, 0.005)]
rp = min(scan, key=lambda t: t[1])[0]; s, fit, e = solve(rp); A = s.sum(); S = s / A; enc = np.cumsum(S)
encd = {f"{R:g}": float(np.interp(R, rho, enc)) for R in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)}
print("r_p", rp, "A", round(A, 3), "rms_log", round(e, 4), "encircled", {k: round(v, 3) for k, v in encd.items()})
tag = f"c{c:g}"
json.dump(dict(c=c, rms_hist=hist, dk_over_kc=(dk / kc).tolist(), tilt=tilt.tolist(), images=use.tolist(), r_over_kc=rc.tolist(),
               logF=lFi.tolist(), weight_per_bin=den.tolist()), open(W + f"d036/pc/direct_fit_{tag}.json", "w"), indent=1)
json.dump(dict(c=c, r_p=rp, level_A=float(A), rms_log=e, rp_scan=scan, rho=rho.tolist(), S_annulus_fraction=S.tolist(), encircled=encd,
               r_fit=rr.tolist(), F_data=yy.tolist(), F_model=fit.tolist()), open(W + f"d036/pc/S_fit_{tag}.json", "w"), indent=1)
np.savez_compressed(W + f"d036/pc/direct_fit_{tag}.npz", logE=lE.astype(np.float32), resid=resid.astype(np.float32), R=R.astype(np.float32))
