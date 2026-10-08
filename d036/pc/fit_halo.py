# -*- coding: utf-8 -*-
"""036 partial coherence, step 1d: halo of the illumination angular distribution from the tail of F(r) (r >= 1.15 kc,
sampled by the ring-2 direct images; the edge region 1.0-1.1 kc is not sampled at c = 1, so the core width is left to the
sample images).  S(rho) = (1 - eta) core + eta H(rho),  H = 2-D generalised Cauchy (1 + rho^2/ell^2)^-beta (unit integral);
F(r) = A [ (1 - eta) disc(r) + eta (H * disc)(r) ],  pupil radius r_p fixed.  Fitted to both collapses (c = 1.5
envelope-only and c = 1 refined) for r in [1.15, 1.8] (log), and eta from the plateau level F(0.8) = A[(1-eta) + eta (H*disc)(0.8)].
usage: python d036/pc/fit_halo.py -> d036/pc/halo_fit.json"""
import json, numpy as np
from scipy.optimize import least_squares
W = "./"
def frac(r_, q, rp):
    if q == 0: return (r_ <= rp).astype(float)
    cc = (r_ ** 2 + q ** 2 - rp ** 2) / (2 * r_ * q)
    return np.where(cc <= -1, 1.0, np.where(cc >= 1, 0.0, np.arccos(np.clip(cc, -1, 1)) / np.pi))
rho = np.linspace(0.0, 3.0, 1501)[1:]; drho = rho[1] - rho[0]
def Hconv(r_, ell, beta, rp):
    h = (1 + (rho / ell) ** 2) ** (-beta) * 2 * np.pi * rho * drho; h /= h.sum()
    return np.stack([frac(r_, q, rp) for q in rho], 1) @ h
out = {}
for name, f in (("c1.5_envelope", "d036/pc/direct_fit.json"), ("c1_refined", "d036/pc/direct_fit_c1.json")):
    fj = json.load(open(W + f)); r = np.array(fj["r_over_kc"]); lF = np.array(fj["logF"]); den = np.array(fj["weight_per_bin"])
    for rp in (1.02, 1.05, 1.08):
        m = (den > 200) & (r >= 1.15) & (r <= 1.8)
        def res(p):
            lAe, ell, beta = p
            return lAe + np.log(np.maximum(Hconv(r[m], ell, beta, rp), 1e-12)) - lF[m]
        best = None
        for ell0 in (0.03, 0.1, 0.3):
            for b0 in (1.3, 2.0, 3.0):
                s = least_squares(res, [np.log(0.1), ell0, b0], bounds=([-10, 0.003, 1.01], [2, 2.0, 8.0]))
                if best is None or s.cost < best.cost: best = s
        lAe, ell, beta = best.x
        pl = (den > 200) & (r > 0.7) & (r < 0.9); Fp = float(np.exp(np.median(lF[pl])))
        Hp = float(Hconv(np.array([0.8]), ell, beta, rp)[0])
        # Fp = A(1-eta) + A eta Hp,  A eta = exp(lAe)  ->  A = Fp - Ae (Hp - 1)
        Ae = float(np.exp(lAe)); A = Fp - Ae * (Hp - 1); eta = Ae / A
        hh = (1 + (rho / ell) ** 2) ** (-beta) * 2 * np.pi * rho * drho; hh /= hh.sum(); ench = np.cumsum(hh)
        out[f"{name}_rp{rp}"] = dict(r_p=rp, eta=eta, ell=ell, beta=beta, A=A, rms_log=float(np.sqrt(np.mean(res(best.x) ** 2))),
                                    halo_encircled={f"{R:g}": float(np.interp(R, rho, ench)) for R in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)})
for k, v in out.items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else {a: round(b, 3) for a, b in vv.items()}) for kk, vv in v.items()})
json.dump(out, open(W + "d036/pc/halo_fit.json", "w"), indent=1)
