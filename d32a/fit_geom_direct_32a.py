"""32a illumination geometry from the vignetting of the direct (sample-out) frames -> d32a/geom_direct_v2.json.

Parallel-beam illumination + objective-FZP scan: in the field of view, the direct beam passes the objective pupil only
inside a disc |x - x_c,n| < R_field whose centre moves with the lens position d_n = (FZP X * 2.5 um, FZP Y * 0.2 um).
The transmission of each registered direct frame (Gaussian-smoothed, divided by the ring-0 median) is fitted, for the
rings 2-5 (where the disc edge is inside the field), with a soft disc 0.5 (1 - tanh((|x - x_c,n| - R) / soft)),
x_c,n = A d_n + c0 (A: full 2x2 matrix, (row, col) <- (X, Y); um from the image centre).
Stage 1: grid search over rotation / scale / mirror + Nelder-Mead (x_c = s Rot(theta) d + c0, soft 0.3 um fixed);
stage 2: full 2x2 A, c0, R and the edge width (Nelder-Mead).
In the pipeline forward model (O_eff = O exp(i pi kappa |x|^2)) x_c = -k_n / kappa and R = kc / kappa, so
k_n / kc = -x_c,n / R independently of the unmeasured sample-objective distance p; kappa_eff = kc / R.
This is the script form of the fit made interactively for the 32a report (2026-09-27); run it from the package root:
  python d32a/fit_geom_direct_32a.py            (needs d32a/data_reg from prep_32a_reg.py)
  python d32a/fit_geom_direct_32a.py --check    (compare with an existing d32a/geom_direct_v2.json, write *_refit.json)"""
import json, os, sys
import numpy as np, tifffile
from scipy import ndimage as ndi
from scipy.optimize import minimize
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
REG = W + "d32a/data_reg/"; PX = 0.0319; DS = 4
P = np.loadtxt(REG + "positions_um.csv", delimiter=",")                      # idx, X um, Y um, det X, det Y, exposure s
X, Y02, expo = P[:, 1], P[:, 2], P[:, 5]
ring = np.repeat(np.arange(8), [15, 20, 25, 30, 35, 40, 45, 50])
use = np.where((ring >= 2) & (ring <= 5))[0]
D = np.stack([tifffile.imread(REG + f"direct/a{i + 1:03d}.tif") for i in range(len(P))]) / expo[:, None, None]
N = D.shape[-1]
ref = ndi.gaussian_filter(np.median(D[ring == 0], 0), 8); band = ref > 0.3 * ref.max()
T = np.array([ndi.gaussian_filter(D[i], 8) / np.maximum(ref, 1e-3) for i in use])
Tm = np.clip(T[:, ::DS, ::DS], 0, 1.5); bd = band[::DS, ::DS]; Tc = np.clip(Tm, 0, 1)
yy, xx = np.mgrid[0:N:DS, 0:N:DS]; Yf = (yy - N / 2) * PX; Xf = (xx - N / 2) * PX
def disc(cr, cc, R, soft):
    d = np.hypot(Yf[None] - cr[:, None, None], Xf[None] - cc[:, None, None])
    return 0.5 * (1 - np.tanh((d - R) / max(abs(soft), 0.05)))
def cost_rot(p, fl):
    th, s, c0y, c0x, R = p; ys = Y02[use] * (-1 if fl else 1); a = np.deg2rad(th)
    cr = s * (X[use] * np.sin(a) + ys * np.cos(a)) + c0y; cc = s * (X[use] * np.cos(a) - ys * np.sin(a)) + c0x
    return float((((disc(cr, cc, R, 0.3) - Tc) ** 2) * bd[None]).sum() / bd.sum() / len(use))
def cost_aff(p):
    a11, a12, a21, a22, c0y, c0x, R, soft = p
    cr = a11 * X[use] + a12 * Y02[use] + c0y; cc = a21 * X[use] + a22 * Y02[use] + c0x
    return float((((disc(cr, cc, R, soft) - Tc) ** 2) * bd[None]).sum() / bd.sum() / len(use))
best = None
for fl in (0, 1):
    for th in range(0, 360, 15):
        for R in (35, 50, 65, 80):
            for s in (0.8, 1.0, 1.3):
                c = cost_rot([th, s, 0, 0, R], fl)
                if best is None or c < best[0]: best = (c, [th, s, 0, 0, R], fl)
fl = best[2]
r1 = minimize(lambda q: cost_rot(q, fl), best[1], method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-3, fatol=1e-7))
th, s, c0y, c0x, Rf = r1.x; a = np.deg2rad(th); sg = -1 if fl else 1
p0 = [s * np.sin(a), sg * s * np.cos(a), s * np.cos(a), -sg * s * np.sin(a), c0y, c0x, Rf, 0.3]
ra = minimize(cost_aff, p0, method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-4, fatol=1e-8, adaptive=True))
A = np.array([[ra.x[0], ra.x[1]], [ra.x[2], ra.x[3]]])
geo = dict(A=A.tolist(), c0_um=ra.x[4:6].tolist(), R_field_um=float(ra.x[6]), soft_um=float(abs(ra.x[7])), cost=float(ra.fun),
           NA_edge_lens_shift_um=float(ra.x[6] / np.sqrt(abs(np.linalg.det(A)))),
           note="direct-image vignetting fit on registered direct frames (rings 2-5): transmitted disc centre x_c = A (X_um, Y_um) + c0 "
                "(um, (row,col) from the image centre), radius R_field; model |k_n + kappa x| < kc => k_n/kc = -x_c / R_field; kappa_eff = kc / R_field",
           stage1=dict(grid_best_cost=best[0], mirror=int(fl), theta_deg=float(th), scale=float(s), cost=float(r1.fun)))
xc = np.c_[X, Y02] @ A.T + ra.x[4:6]
geo["xc_um"] = xc.tolist(); geo["k_over_kc"] = (np.hypot(*xc.T) / ra.x[6]).tolist()
out = W + "d32a/geom_direct_v2.json"
if "--check" in sys.argv and os.path.exists(out):
    g0 = json.load(open(out)); out = W + "d32a/geom_direct_v2_refit.json"
    print("A", np.round(A, 4).tolist(), "vs", np.round(np.array(g0["A"]), 4).tolist(), "| R", round(geo["R_field_um"], 2), "vs", round(g0["R_field_um"], 2),
          "| max |x_c diff| um", round(float(np.abs(xc - np.array(g0["xc_um"])).max()), 3))
json.dump(geo, open(out, "w"), indent=1)
print("cost", round(ra.fun, 5), "A", np.round(A, 4).tolist(), "c0", np.round(ra.x[4:6], 3).tolist(), "R", round(ra.x[6], 2), "soft", round(abs(ra.x[7]), 3),
      "median |k|/kc per ring", [round(float(np.median(np.array(geo["k_over_kc"])[ring == r])), 3) for r in range(8)], "->", out)
