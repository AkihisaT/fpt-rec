#!/usr/bin/env python
"""Geometry / preprocessing evidence figure for 32a (d32a/fig_geometry_32a.png)."""
import json, os, sys
import numpy as np, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon.preprocess import read_itex
RAW = os.environ.get("FPT32A_RAW", "").rstrip("/") + "/"
if not os.environ.get("FPT32A_RAW"): raise SystemExit("生データのフォルダを環境変数 FPT32A_RAW で指定してください（paths_local.example.sh を参照）")
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False})
P = np.loadtxt(RAW + "平行照明計算fzp250nm.csv", delimiter=","); X, Y = P[:, 1] * 2.5, P[:, 2] * 0.2
ring = np.repeat(np.arange(8), [15, 20, 25, 30, 35, 40, 45, 50])
g = json.load(open(W + "d32a/geom_direct_v2.json")); A = np.array(g["A"]); c0 = np.array(g["c0_um"]); RF = g["R_field_um"]
C = np.load(W + "d32a/star_centres.npy")
resX = (P[:, 3] / -527.75 - P[:, 1]) * 2.5 / 0.0319
fig = plt.figure(figsize=(7.4, 6.8)); gs = fig.add_gridspec(3, 4, hspace=0.62, wspace=0.62)
# a: scan
ax = fig.add_subplot(gs[0, 0]); cols = {0: "#08519c", 1: "#08519c", 2: "#08519c", 3: "0.6", 4: "0.6", 5: "0.6", 6: "#e6550d", 7: "#e6550d"}
for r in range(8):
    s = ring == r; ax.scatter(X[s], Y[s], s=5, color=cols[r], lw=0)
th = np.linspace(0, 2 * np.pi, 200); Rn = g["NA_edge_lens_shift_um"]
ax.plot(Rn * np.cos(th), Rn * np.sin(th), "k--", lw=0.7); ax.set_aspect("equal")
ax.set_xlabel("FZP X (µm) = col.2 × 2.5"); ax.set_ylabel("FZP Y (µm) = col.3 × 0.2")
ax.set_title("a  scan positions", loc="left"); ax.text(0, 0, f"blue: BF set\norange: DF set\n-- NA edge {Rn:.0f} µm", ha="center", va="center", fontsize=5.8)
# b: vignetting example with fitted circle
ax = fig.add_subplot(gs[0, 1]); i = 67
d = read_itex(RAW + f"2_direct/a{i + 1:03d}.img").astype(np.float32) - 100
ax.imshow(ndi.gaussian_filter(d, 2), cmap="gray", vmin=0, vmax=np.percentile(d, 99.5))
xc = A @ np.array([X[i], Y[i]]) + c0
ph = np.linspace(0, 2 * np.pi, 400); yy = 500 + (xc[0] + RF * np.sin(ph)) / 0.0319; xx = 500 + (xc[1] + RF * np.cos(ph)) / 0.0319
ok = (yy > 0) & (yy < 1000) & (xx > 0) & (xx < 1000); ax.plot(xx[ok], yy[ok], "c--", lw=0.8)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("b  pupil-edge fit (ring 3)", loc="left")
# c: lens-detector mismatch shift
ax = fig.add_subplot(gs[0, 2:]); s = ring <= 2
ax.scatter(resX[s], C[s, 1] - np.median(C[s, 1]), s=6, color="#08519c", lw=0)
ax.plot([0, 39.1], [0, -39.1], "k-", lw=0.7); ax.set_xlabel("detector − FZP position mismatch, X (px; 1.25 µm = 39.1 px)")
ax.set_ylabel("star centre col. (px)"); ax.set_title("c  image shift follows the lens–detector mismatch (line: slope −1)", loc="left")
# d: spectra with kc circles
Rb = np.load(W + "fpt_pipeline/fpt_output_32a_bf_reg2/work/preprocessed.npy", mmap_mode="r")
cal = json.load(open(W + "fpt_pipeline/fpt_output_32a_bf_reg2/work/calibration.json")); kn = np.array(cal["k_solver"])
N = 1000; w = np.outer(np.hanning(N), np.hanning(N)); dk = 1 / (N * 0.0319)
for j, i in enumerate([20, 40]):
    ax = fig.add_subplot(gs[1, j]); r = np.asarray(Rb[i]); F = np.abs(np.fft.fftshift(np.fft.fft2((r - r.mean()) * w))) ** 2
    Fl = np.log10(ndi.gaussian_filter(F, 1.5) + 1e-9); e = N / 2 * dk
    ax.imshow(Fl, cmap="magma", extent=[-e, e, e, -e], vmin=Fl.max() - 4.3, vmax=Fl.max() - 0.3)
    for kcv, ls, col in [(2.0, "--", "c"), (2.5, ":", "w")]:
        for sg in (1, -1): ax.plot(sg * kn[i, 1] + kcv * np.cos(th), sg * kn[i, 0] + kcv * np.sin(th), ls, color=col, lw=0.8)
    ax.set_xlim(-5, 5); ax.set_ylim(5, -5); ax.set_xlabel("q_x (µm⁻¹)")
    if j == 0: ax.set_ylabel("q_y (µm⁻¹)")
    ax.set_title(f"{'de'[j]}  |I(q)|², ring {ring[i]}" + ("\ncyan kc 2.0, white kc 2.5" if j == 0 else "\n"), loc="left")
# f: registration corrections
rg = json.load(open(W + "d32a/register_32a.json")); cb = np.array(rg["bright"]["correction_px"])
rm = json.load(open(W + "fpt_pipeline/fpt_output_32a_bf_reg2/work/register_to_model.json")); sm = np.array(rm["shifts_px"])
ax = fig.add_subplot(gs[1, 2:]); ii = np.arange(60)
ax.plot(ii, np.hypot(*cb.T), "o-", ms=2.5, lw=0.7, color="#08519c", label="pairwise network (random + rotation part)")
ax.plot(ii, np.hypot(*sm[:60].T), "s-", ms=2.5, lw=0.7, color="#e6550d", label="remaining after it (model-based)")
ax.set_xlabel("bright-field image index (rings 0 | 1 | 2)"); ax.set_ylabel("|correction| (px)")
for b in (14.5, 34.5): ax.axvline(b, color="0.7", lw=0.6)
ax.legend(frameon=False, loc="upper left"); ax.set_title("f  residual stage errors removed before reconstruction", loc="left")
# g, h: 0th-order fringes (ratio images ring 0 vs ring 2)
for j, i in enumerate([0, 35]):
    ax = fig.add_subplot(gs[2, j]); s_ = read_itex(RAW + f"1_sample/a{i + 1:03d}.img").astype(np.float32) - 100
    d_ = read_itex(RAW + f"2_direct/a{i + 1:03d}.img").astype(np.float32) - 100; r = s_ / np.maximum(d_, 1); r /= np.median(r)
    ax.imshow(r, cmap="gray", vmin=0.6, vmax=1.4); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{'gh'[j]}  S/D, ring {ring[i]} (|d| {np.hypot(X[i], Y[i]):.0f} µm)", loc="left")
ax = fig.add_subplot(gs[2, 2:]); ax.axis("off")
ax.text(0, 0.95, "g: zone-plate-like arcs centred on the point whose\nsample-out ray passes the FZP axis (FZP 0th-order light\n"
        "interfering with the image). Strong in rings 0–1, weak in ring 2,\nabsent from ring 3. Not in the forward model of any method.",
        va="top", fontsize=6.8)
fig.savefig(W + "d32a/fig_geometry_32a.png", dpi=220, bbox_inches="tight")
print("written")
