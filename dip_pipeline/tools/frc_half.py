#!/usr/bin/env python
"""Half-data FRC (repeat 1 vs repeat 2) of two independent reconstructions.

  python tools/frc_half.py --dip dip_output_halfA/DIP_planewave_state.npz dip_output_halfB_seed1/DIP_planewave_state.npz
  python tools/frc_half.py --prev prev_crop_A_c1.npz prev_crop_B_c1.npz --crop 512

Use different random seeds for the two DIP runs: with the same seed the untrained network produces the same
high-frequency structure in both halves and the FRC is overestimated.
Thresholds: half-bit (van Heel & Schatz 2005) and 1/7. Physical phase (twin -1 of the solver frame)."""
import argparse, json, os, sys
import numpy as np, torch
from scipy.signal.windows import tukey
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dipfpm import train as T
from dipfpm.physics import fourier_resample


def frc_rings(a, b, dx, ring_px=2.0, apod=0.25):
    N = a.shape[-1]; w = np.outer(tukey(N, apod), tukey(N, apod))
    F1 = np.fft.fftshift(np.fft.fft2((a - a.mean()) * w)); F2 = np.fft.fftshift(np.fft.fft2((b - b.mean()) * w))
    yy, xx = np.mgrid[-(N // 2):N - N // 2, -(N // 2):N - N // 2]; rr = np.hypot(yy, xx)
    edges = np.arange(1, N // 2, ring_px); q, f, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (rr >= lo) & (rr < hi)
        f.append(np.real((F1[s] * np.conj(F2[s])).sum()) / np.sqrt((np.abs(F1[s]) ** 2).sum() * (np.abs(F2[s]) ** 2).sum()))
        q.append(0.5 * (lo + hi) / (N * dx)); n.append(s.sum())
    q, f, n = map(np.array, (q, f, n))
    return q, f, (0.2071 + 1.9102 / np.sqrt(n)) / (1.2071 + 0.9102 / np.sqrt(n))


def crossing(q, f, thr, qmin=0.5):
    thr = np.broadcast_to(thr, f.shape); below = np.where((q >= qmin) & (f < thr))[0]
    if not len(below): return float("nan")
    i = below[0]
    if i == 0: return float(q[0])
    d0, d1 = f[i - 1] - thr[i - 1], f[i] - thr[i]
    return float(q[i - 1] + (q[i] - q[i - 1]) * d0 / (d0 - d1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dip", nargs=2, help="two <tag>_state.npz files")
    g.add_argument("--prev", nargs=2, help="two fptrecon crop results (run_prev_crop.py)")
    ap.add_argument("--which", default="best", choices=["best", "final"]); ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--pixel-um", type=float, default=0.0319); ap.add_argument("--border", type=int, default=32)
    ap.add_argument("--twin", type=int, default=-1); ap.add_argument("--json")
    a = ap.parse_args()
    if a.dip:
        ims = [a.twin * T.load_state(p)[a.which]["phi"] for p in a.dip]
    else:
        ims = [a.twin * fourier_resample(torch.tensor(np.load(p)["phi"].astype(np.float64)), a.crop).numpy() for p in a.prev]
    b = a.border; q, f, hb = frc_rings(ims[0][b:-b, b:-b], ims[1][b:-b, b:-b], a.pixel_um)
    qh, q7 = crossing(q, f, hb), crossing(q, f, 1 / 7)
    print(f"half-bit: {qh:.2f} um^-1 (half-pitch {500 / qh:.0f} nm) | 1/7: {q7:.2f} um^-1 (half-pitch {500 / q7:.0f} nm)")
    if a.json:
        json.dump(dict(q=q.tolist(), frc=f.tolist(), halfbit=hb.tolist(), q_halfbit=qh, q_1_7=q7), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
