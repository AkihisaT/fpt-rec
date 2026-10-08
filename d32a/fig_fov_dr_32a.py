#!/usr/bin/env python
"""32a: FOV-strength scan and dr comparison (BLIS-FPM, bright field rings 0-2, 7 held-out images) -> d32a/fig_fov_dr_32a.png, d32a/fov_dr_summary.json"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False})
scan = {0.0: "dr_cmp/reg2_dr250_c0", 0.5: "fov_scan/c0.5", 1.0: "dr_cmp/reg2_dr250_c1", 1.25: "fov_scan/c1.25", 1.5: "fov_scan/c1.5", 2.0: "fov_scan/c2.0"}
S = {c: json.load(open(W + "d32a/" + f + ".json")) for c, f in scan.items()}
cs = np.array(sorted(S)); fit = np.array([S[c]["misfit"] for c in cs]); ho = np.array([S[c]["heldout_mean"] for c in cs])
g = json.load(open(W + "d32a/geom_direct_v2.json")); lam = 1.239842 / 30 * 1e-3; c_eff = (2.0 / g["R_field_um"]) * lam * 0.75e6
dr = json.load(open(W + "d32a/dr_cmp/dr_compare_summary.json"))
fig = plt.figure(figsize=(7.2, 4.6)); gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45)
ax = fig.add_subplot(gs[0, 0])
ax.plot(cs, fit, "o-", color="#1b9e77", ms=3.5, lw=1, label="fitted (53 images)"); ax.plot(cs, ho, "s--", color="#1b9e77", ms=3.5, lw=1, mfc="w", label="held out (7 images)")
ax.axvline(c_eff, color="0.5", lw=0.7, ls=":"); ax.text(c_eff + 0.05, 0.67, f"c from\nvignetting\n= {c_eff:.2f}", fontsize=6, color="0.35", va="center")
ax.set_xlabel("FOV factor c  (κ = c / (λ p), p = 0.75 m)"); ax.set_ylabel("band misfit (lower = better)"); ax.text(0.45, 0.455, "fitted (53 images)", fontsize=6.3, color="#1b9e77", va="center"); ax.text(0.3, 1.0, "held out (7 images)", fontsize=6.3, color="#1b9e77", va="center"); ax.set_xlim(-0.1, 2.1); ax.set_ylim(0.42, 1.05)
ax.set_title("a  BLIS-FPM vs FOV strength", loc="left")
ax = fig.add_subplot(gs[0, 1]); xs = np.arange(4); lab = ["kc 2.0\nplane", "kc 2.0\nFOV", "kc 2.5\nplane", "kc 2.5\nFOV"]
keys = ["reg2_dr250_c0", "reg2_dr250_c1", "reg2_dr200_c0", "reg2_dr200_c1"]
ax.bar(xs - 0.18, [dr[k]["fitted"] for k in keys], 0.36, color=["#1b9e77", "#1b9e77", "#bbbbbb", "#bbbbbb"], alpha=0.5, label="fitted")
ax.bar(xs + 0.18, [dr[k]["heldout"] for k in keys], 0.36, color=["#1b9e77", "#1b9e77", "#777777", "#777777"], label="held out")
ax.set_xticks(xs); ax.set_xticklabels(lab); ax.set_ylabel("band misfit"); ax.legend(frameon=False, loc="upper left", ncol=2)
ax.set_ylim(0, 1.25); ax.set_title("b  Δr 250 nm (kc 2.0) vs 200 nm (kc 2.5)", loc="left")
for j, (t, kc) in enumerate([("reg2_dr250_c1", 2.0), ("reg2_dr200_c1", 2.5)]):
    z = np.load(W + f"d32a/dr_cmp/{t}.npz"); Wm = z["W"]; M = int(z["M"]); dk = float(z["dk"])
    kk = (np.arange(M) - M // 2) * dk; KY, KX = np.meshgrid(kk, kk, indexing="ij"); Wm = np.where(np.hypot(KY, KX) <= kc, Wm, np.nan); Wm = Wm - np.nanmean(Wm)
    ax = fig.add_subplot(gs[1, j]); im = ax.imshow(Wm, cmap="RdBu_r", vmin=-4, vmax=4, extent=[kk[0], kk[-1], kk[-1], kk[0]])
    ax.set_xlim(-2.7, 2.7); ax.set_ylim(2.7, -2.7); ax.set_xlabel("k_x (µm⁻¹)"); ax.set_ylabel("k_y (µm⁻¹)") if j == 0 else None
    ax.set_title(f"{'cd'[j]}  pupil phase (FOV), kc {kc:g} µm⁻¹", loc="left")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("rad")
ax = fig.add_subplot(gs[:, 2]); ax.axis("off")
txt = ["held-out = 7 images left out of the fit", "(indices 0, 9, 18, 27, 36, 45, 54),", "predicted with the fitted object and pupil.", "",
       f"kc 2.0: fitted {dr['reg2_dr250_c1']['fitted']:.3f} / held out {dr['reg2_dr250_c1']['heldout']:.3f} (FOV)",
       f"kc 2.5: fitted {dr['reg2_dr200_c1']['fitted']:.3f} / held out {dr['reg2_dr200_c1']['heldout']:.3f} (FOV)", "",
       "kc 2.5 puts pupil area beyond the", "data support (c: noisy rim)."]
ax.text(0, 0.98, "\n".join(txt), va="top", fontsize=6.8)
fig.savefig(W + "d32a/fig_fov_dr_32a.png", dpi=220, bbox_inches="tight")
json.dump(dict(fov_scan={str(c): dict(fitted=float(f), heldout=float(h)) for c, f, h in zip(cs, fit, ho)}, c_eff_vignetting=c_eff, dr=dr),
          open(W + "d32a/fov_dr_summary.json", "w"), indent=1)
print(dict(zip(cs.tolist(), np.round(ho, 3).tolist())), round(c_eff, 3))
