"""Reproducibility / resolution of 036 from independent half data sets (A = repeats 0,2,4; B = 1,3,5).

FRC between the phase maps reconstructed from A and from B (unfiltered, solver frame, mean removed, Tukey 0.2 window)
for BLIS-FPM (FOV model, full field 1024 camera px, and the central 512 px) and EPRY (512 crop), each with bright-field
images only (ring 1, 9 images) and with bright + dark field (rings 1 + 3, 28 images).
Resolution = first frequency (> 1 um^-1) where the FRC falls below the 1/7 threshold (and the half-bit curve).
Output: d036/frc_halves.json, d036/fig_frc_halves.png"""
import json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, W + "dip_pipeline"); sys.path.insert(0, W + "epry_pipeline")
from dipfpm import evaluate as E, train as T
from dipfpm.physics import fourier_resample

dx = 0.0319; H = 1024
def blis_phi(f, crop=None):
    z = np.load(f); p = fourier_resample(torch.tensor(z["phi"], dtype=torch.float64), H).numpy()
    if crop: p = p[H // 2 - crop // 2:H // 2 + crop // 2, H // 2 - crop // 2:H // 2 + crop // 2]
    return p
def epry_phi(d, tag="EPRY_FOV_c1"):
    return T.load_state(os.path.join(d, f"{tag}_state.npz"))["best"]["phi"]
def limit(q, f, thr, qmin=1.0):
    """first crossing of f below thr (array or scalar) for q > qmin, linearly interpolated"""
    thr = np.broadcast_to(np.asarray(thr, float), f.shape); d = f - thr
    for i in range(1, len(q)):
        if q[i] > qmin and d[i] < 0 <= d[i - 1]:
            return float(q[i - 1] + (q[i] - q[i - 1]) * d[i - 1] / (d[i - 1] - d[i]))
    return float("nan") if d[q > qmin].min() < 0 else float(q[-1])
NB = 60
def halfbit(q, N, dx_):
    """half-bit threshold (van Heel & Schatz 2005); n = Fourier pixels per ring of width 0.5/dx/NB"""
    n = np.maximum(2 * np.pi * (q * N * dx_) * (0.5 / dx_ / NB * N * dx_), 1)
    return (0.2071 + 1.9102 / np.sqrt(n)) / (1.2071 + 0.9102 / np.sqrt(n))

cases = {
    "BLIS-FPM, bright field (ring 1), full field": lambda: (blis_phi(W + "d036/blis_half_r1/s-1_halfA.npz"), blis_phi(W + "d036/blis_half_r1/s-1_halfB.npz")),
    "BLIS-FPM, bright + dark field (rings 1, 3), full field": lambda: (blis_phi(W + "d036/blis_half_r13/s-1_halfA.npz"), blis_phi(W + "d036/blis_half_r13/s-1_halfB.npz")),
    "BLIS-FPM, bright field (ring 1), centre 512": lambda: (blis_phi(W + "d036/blis_half_r1/s-1_halfA.npz", 512), blis_phi(W + "d036/blis_half_r1/s-1_halfB.npz", 512)),
    "BLIS-FPM, bright + dark field (rings 1, 3), centre 512": lambda: (blis_phi(W + "d036/blis_half_r13/s-1_halfA.npz", 512), blis_phi(W + "d036/blis_half_r13/s-1_halfB.npz", 512)),
    "EPRY, bright field (ring 1), centre 512": lambda: (epry_phi(W + "epry_pipeline/epry_output_036bf_halfA"), epry_phi(W + "epry_pipeline/epry_output_036bf_halfB")),
    "EPRY, bright + dark field (rings 1, 3), centre 512": lambda: (epry_phi(W + "epry_pipeline/epry_output_036df_halfA"), epry_phi(W + "epry_pipeline/epry_output_036df_halfB")),
}
out = {}
for lab, fn in cases.items():
    try:
        a, b = fn()
    except FileNotFoundError as e:
        print("missing", lab, e); continue
    q, f = E.frc(a, b, dx, nbins=NB)
    hb = halfbit(q, a.shape[0], dx)
    out[lab] = dict(q=q.tolist(), frc=f.tolist(), halfbit=hb.tolist(), N=int(a.shape[0]),
                    limit_1_7=limit(q, f, 1 / 7), limit_halfbit=limit(q, f, hb))
    print(f"{lab:58s} 1/7: {out[lab]['limit_1_7']:.2f} um^-1   half-bit: {out[lab]['limit_halfbit']:.2f} um^-1")
json.dump(out, open(W + "d036/frc_halves.json", "w"), indent=1)
