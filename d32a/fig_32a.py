#!/usr/bin/env python
"""Figures for the 32a comparison (all reconstructions shown WITHOUT band filtering).
usage: python d32a/fig_32a.py <out_dir of compare_32a.py> [title prefix]
Writes into <out_dir>: fig_images_planewave.png, fig_images_FOV_c1.png, fig_zoom.png, fig_spokes.png, fig_misfit.png"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
out = sys.argv[1].rstrip("/") + "/"; ttl = sys.argv[2] if len(sys.argv) > 2 else ""
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif"})
z = np.load(out + "recs.npz"); S = json.load(open(out + "compare_summary.json")); SP = json.load(open(out + "spokes.json"))
dx, dk, kc = float(z["dx"]), float(z["dk"]), float(z["kc"]); cy, cx = z["star"]
METH = [("DIP (optimal it.)", "DIP opt.", "#7570b3"), ("DIP (30 it.)", "DIP 30 it.", "#a6761d"), ("EPRY", "EPRY", "#1f78b4"), ("BLIS-FPM", "BLIS-FPM", "#1b9e77")]
MODS = [("planewave", "plane-wave model"), ("FOV_c1", "FOV model (c = 1)")]
get = lambda lb, k: z[f"{lb}|{k}"] if f"{lb}|{k}" in z.files else None
def scalebar(ax, n_px, um, col="w"):
    L = um / dx; x0 = n_px * 0.06; y0 = n_px * 0.93
    ax.plot([x0, x0 + L], [y0, y0], color=col, lw=2.5, solid_capstyle="butt")
    ax.text(x0 + L / 2, y0 - n_px * 0.03, f"{um:g} µm", color=col, ha="center", va="bottom", fontsize=7)
def scale(imgs, lo=0.5, hi=99.5):
    v = np.array([np.percentile(i, [lo, hi]) for i in imgs if i is not None]); return tuple(np.median(v, 0))
Z = 160                                    # zoom half-size (px) around the star centre -> 10 um box
for mtag, mlab in MODS:
    rows = [(m, s, c) for m, s, c in METH if get(f"{m}, {mlab.split(' (')[0]}", "phi_unf") is not None]
    if not rows: continue
    lbl = lambda m: f"{m}, {mlab.split(' (')[0]}"
    P = {m: get(lbl(m), "phi_unf") for m, _, _ in rows}; T = {m: get(lbl(m), "T_unf") for m, _, _ in rows}; Wp = {m: get(lbl(m), "W_phys") - np.nanmean(get(lbl(m), "W_phys")) for m, _, _ in rows}     # piston removed
    P = {m: p - np.median(p) for m, p in P.items()}
    N = next(iter(P.values())).shape[0]
    y0, y1 = int(max(cy - Z, 0)), int(min(cy + Z, N)); x0, x1 = int(max(cx - Z, 0)), int(min(cx + Z, N))
    vP = scale(list(P.values())); vT = scale(list(T.values())); vZ = scale([p[y0:y1, x0:x1] for p in P.values()])
    wmax = np.nanmax([np.nanpercentile(np.abs(w), 99) for w in Wp.values()])
    fig, axs = plt.subplots(len(rows), 4, figsize=(7.2, 1.85 * len(rows) + 0.35), gridspec_kw=dict(wspace=0.04, hspace=0.12))
    axs = np.atleast_2d(axs)
    for i, (m, short, col) in enumerate(rows):
        r = S["results"][lbl(m)]; mu = r["misfit_unfiltered"]["heldout"]["all"]; q3 = r.get("spoke_snr3_um_inv")
        for j, (img, cm, vv) in enumerate([(T[m], "gray", vT), (P[m], "gray", vP), (P[m][y0:y1, x0:x1], "gray", vZ)]):
            ax = axs[i, j]; ax.imshow(img, cmap=cm, vmin=vv[0], vmax=vv[1], interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(col); sp.set_linewidth(1.5)
            scalebar(ax, img.shape[0], 5 if j < 2 and N > 600 else (2 if j == 2 else 5))
            if j == 1: ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#e7298a", lw=0.8))
        ax = axs[i, 3]; kk = (np.arange(Wp[m].shape[0]) - Wp[m].shape[0] // 2) * dk
        ax.imshow(Wp[m], cmap="RdBu_r", vmin=-wmax, vmax=wmax, extent=[kk[0], kk[-1], kk[-1], kk[0]], interpolation="nearest")
        ax.set_xlim(-1.1 * kc, 1.1 * kc); ax.set_ylim(1.1 * kc, -1.1 * kc); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        axs[i, 0].set_ylabel(short, color=col, fontsize=8)
        hb, hd = r["misfit_unfiltered"]["heldout"].get("bright"), r["misfit_unfiltered"]["heldout"].get("dark")
        mtxt = f"held-out misfit {mu:.3f}" + (f"\n(bright {hb:.3f}, dark {hd:.3f})" if hd is not None and hb is not None else "")
        axs[i, 3].text(1.02, 0.5, mtxt + (f"\nspoke SNR>3: {q3:.2f} µm⁻¹" if q3 == q3 else ""),
                       transform=axs[i, 3].transAxes, fontsize=6.5, va="center", ha="left")
    heads = [f"transmission\ngrey [{vT[0]:.3f}, {vT[1]:.3f}]", f"phase (rad)\ngrey [{vP[0]:.2f}, {vP[1]:.2f}]",
             "phase, star centre\n(pink box, 10 µm)", f"pupil phase\n±{wmax:.1f} rad, |k| ≤ {kc:g} µm⁻¹"]
    for j, h in enumerate(heads): axs[0, j].set_title(h, fontsize=7, loc="left")
    fig.text(0.125, axs[0, 0].get_position().y1 + 0.42 / fig.get_figheight(), f"{ttl}{mlab} — no band filter on any image", fontsize=8, ha="left", va="bottom")
    fig.savefig(out + f"fig_images_{mtag}.png", dpi=220, bbox_inches="tight"); plt.close(fig)
# zoom panel: all methods x both models, common scale per model
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9), gridspec_kw=dict(wspace=0.04, hspace=0.1))
for i, (mtag, mlab) in enumerate(MODS):
    mm = mlab.split(" (")[0]; imgs = {}
    for m, short, col in METH:
        p = get(f"{m}, {mm}", "phi_unf")
        if p is None: continue
        N = p.shape[0]; y0, y1 = int(max(cy - 80, 0)), int(min(cy + 80, N)); x0, x1 = int(max(cx - 80, 0)), int(min(cx + 80, N))
        imgs[m] = p[y0:y1, x0:x1] - np.median(p[y0:y1, x0:x1])
    vv = scale(list(imgs.values()))
    for j, (m, short, col) in enumerate(METH):
        ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
        if m not in imgs: ax.axis("off"); continue
        ax.imshow(imgs[m], cmap="gray", vmin=vv[0], vmax=vv[1], interpolation="nearest")
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(col); sp.set_linewidth(1.5)
        scalebar(ax, imgs[m].shape[0], 1)
        if i == 0: ax.set_title(short, color=col, fontsize=8)
        q3 = S["results"][f"{m}, {mm}"].get("spoke_snr3_um_inv")
        ax.text(0.97, 0.04, f"{q3:.2f} µm⁻¹" if q3 == q3 else "", transform=ax.transAxes, color="w", fontsize=6.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="k", ec="none", alpha=0.55))
    axs[i, 0].set_ylabel(mlab, fontsize=8)
fig.text(0.125, axs[0, 0].get_position().y1 + 0.3 / fig.get_figheight(), f"{ttl}star centre (5.1 µm box), unfiltered phase; number = spoke SNR>3 limit", fontsize=8, ha="left", va="bottom")
fig.savefig(out + "fig_zoom.png", dpi=250, bbox_inches="tight"); plt.close(fig)
# spokes
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, gridspec_kw=dict(wspace=0.06))
for ax, (mtag, mlab) in zip(axs, MODS):
    mm = mlab.split(" (")[0]
    for m, short, col in METH:
        k = f"{m}, {mm}"
        if k not in SP: continue
        q = np.array(SP[k]["q"]); rt = np.array(SP[k]["mod"]) / np.array(SP[k]["offharm"]); o = np.argsort(q)
        ax.semilogy(q[o], rt[o], color=col, lw=1.0, label=f"{short} ({SP[k]['q_snr3']:.2f})")
    ax.axhline(3, color="0.4", lw=0.7, ls="--"); ax.set_xlim(0.8, 8); ax.set_xlabel("spoke frequency 36/(2πr) (µm⁻¹)")
    ax.set_title(mlab, loc="left"); ax.legend(frameon=False, loc="upper right")
axs[0].set_ylabel("36-fold modulation / off-harmonic level")
fig.savefig(out + "fig_spokes.png", dpi=220, bbox_inches="tight"); plt.close(fig)
# misfit bars
fig, ax = plt.subplots(figsize=(7.2, 2.4))
xs, labs = [], []
for i, (mtag, mlab) in enumerate(MODS):
    mm = mlab.split(" (")[0]
    for j, (m, short, col) in enumerate(METH):
        k = f"{m}, {mm}"
        if k not in S["results"]: continue
        r = S["results"][k]["misfit_unfiltered"]; x = i * 5 + j
        ax.bar(x - 0.2, r["train"]["all"], 0.4, color=col, alpha=0.45); ax.bar(x + 0.2, r["heldout"]["all"], 0.4, color=col)
        xs.append(x); labs.append(f"{short}\n{'plane' if mtag == 'planewave' else 'FOV'}")
ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=6); ax.set_ylabel("band misfit (lower = better)")
ax.set_title(f"{ttl}data misfit of the unfiltered objects: light = fitted images, dark = held-out images", loc="left")
fig.savefig(out + "fig_misfit.png", dpi=220, bbox_inches="tight"); plt.close(fig)
print("figures written to", out)
