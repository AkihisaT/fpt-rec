#!/usr/bin/env python
"""32a reconstructions before / after the posaffine focus loop (residual object defocus moved into the pupil), FOV model.
Unfiltered phase (no band filter), as reconstructed (no refocusing afterwards).
 fig_focusloop_zoom_32a_<grid>.png : star centre (5.1 um box); rows = BF before, BF after, BF+DF before, BF+DF after;
                                     columns = DIP opt., DIP 30 it., EPRY, BLIS-FPM; one grey scale per data set (BF or BF+DF)
                                     shared by before and after and by the 4 methods.
 fig_focusloop_field_32a_<grid>.png: whole field (512 or 1000 px) of the phase, same layout and grey-scale rule.
Numbers in the panels: spoke SNR>3 limit (um^-1) and object refocus z (mm, compare_32a.py pupil_drift_check).
Inputs d32a/out_32a{bf,df}aff{,2}_<grid>/ (recs.npz, compare_summary.json).
usage: python d32a/fig_focusloop_images_32a.py [512|1000]"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
G = sys.argv[1] if len(sys.argv) > 1 else "512"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "font.family": "sans-serif"})
METH = [("DIP (optimal it.)", "DIP opt.", "#7570b3"), ("DIP (30 it.)", "DIP 30 it.", "#a6761d"), ("EPRY", "EPRY", "#1f78b4"), ("BLIS-FPM", "BLIS-FPM", "#1b9e77")]
ROWS = [("bf", "aff", "BF (rings 0-2)", "before focus loop\n(initial pupil z -1.52 mm)"), ("bf", "aff2", "BF (rings 0-2)", "after focus loop\n(initial pupil z -2.72 mm)"),
        ("df", "aff", "BF+DF (+ rings 6-7)", "before focus loop"), ("df", "aff2", "BF+DF (+ rings 6-7)", "after focus loop")]
D = {}
for s, t, _, _ in ROWS:
    d = W + f"d32a/out_32a{s}{t}_{G}/"
    D[(s, t)] = (np.load(d + "recs.npz"), json.load(open(d + "compare_summary.json")))
dx = float(D[("bf", "aff2")][0]["dx"])
def scalebar(ax, n_px, um, col="w"):
    L = um / dx; x0 = n_px * 0.06; y0 = n_px * 0.07                       # top-left corner (numbers sit bottom-right)
    ax.plot([x0, x0 + L], [y0, y0], color=col, lw=2.2, solid_capstyle="butt")
    ax.text(x0 + L / 2, y0 + n_px * 0.03, f"{um:g} µm", color=col, ha="center", va="top", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.1", fc="k", ec="none", alpha=0.45))
def crop(p, cy, cx, h):
    if h is None: return p
    N = p.shape[0]; y0, y1 = int(max(cy - h, 0)), int(min(cy + h, N)); x0, x1 = int(max(cx - h, 0)), int(min(cx + h, N))
    return p[y0:y1, x0:x1]
def figure(kind):
    h = 80 if kind == "zoom" else None                                      # 160 px = 5.1 um box
    imgs = {}
    for s, t, _, _ in ROWS:
        z, S = D[(s, t)]; cy, cx = z["star"]
        for m, _, _ in METH:
            p = crop(z[f"{m}, FOV model|phi_unf"].astype(float), cy, cx, h); imgs[(s, t, m)] = p - np.median(p)
    vv = {s: tuple(np.median([np.percentile(imgs[(s_, t_, m)], [0.5, 99.5]) for s_, t_, _, _ in ROWS if s_ == s for m, _, _ in METH], 0)) for s in ("bf", "df")}
    fig, axs = plt.subplots(4, 4, figsize=(7.0, 7.6), gridspec_kw=dict(wspace=0.04, hspace=0.06))
    out = {}
    for i, (s, t, slab, rlab) in enumerate(ROWS):
        z, S = D[(s, t)]
        for j, (m, short, col) in enumerate(METH):
            ax = axs[i, j]; img = imgs[(s, t, m)]
            ax.imshow(img, cmap="gray", vmin=vv[s][0], vmax=vv[s][1], interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(col); sp.set_linewidth(1.4 if t == "aff2" else 0.8); sp.set_linestyle("-" if t == "aff2" else (0, (2, 1.5)))
            R = S["results"][f"{m}, FOV model"]; q3 = R.get("spoke_snr3_um_inv"); rf = R["pupil_drift_check"]["object_refocus"]["z_um"] / 1e3
            pz = R["pupil_drift_check"]["pupil_defocus_um"] / 1e3
            out[f"{s}|{t}|{m}"] = dict(spoke_snr3_um_inv=q3, object_refocus_mm=rf, recovered_pupil_defocus_mm=pz)
            ax.text(0.97, 0.04, f"{q3:.2f} µm⁻¹\nrefocus {rf:+.2f} mm\npupil z {pz:+.2f} mm", transform=ax.transAxes, color="w", fontsize=6, ha="right", va="bottom",
                    linespacing=1.15, bbox=dict(boxstyle="round,pad=0.15", fc="k", ec="none", alpha=0.6))
            if j == 0 and i in (0, 2): scalebar(ax, img.shape[0], 1 if kind == "zoom" else 5)
            if i == 0: ax.set_title(short, color=col, fontsize=8)
        axs[i, 0].set_ylabel(rlab, fontsize=7.5)
    for a in axs[2:].ravel():                                               # gap between the BF and BF+DF bands
        b = a.get_position(); a.set_position([b.x0, b.y0 - 0.035, b.width, b.height])
    for i0, slab in ((0, "BF (rings 0-2)"), (2, "BF+DF (+ rings 6-7)")):
        y = axs[i0, 0].get_position().y1
        fig.text(axs[0, 0].get_position().x0, y + (0.008 if i0 == 2 else 0.03), slab + ("  —  star centre, 5.1 µm box" if kind == "zoom" else f"  —  {G} px field ({int(G) * dx:.1f} µm)"),
                 fontsize=8, fontweight="bold", ha="left", va="bottom")
    fig.text(0.5, axs[3, 0].get_position().y0 - 0.012, f"FOV model (c = 1), unfiltered phase as reconstructed; grey scale shared within BF and within BF+DF "
             f"(BF {vv['bf'][0]:+.3f} to {vv['bf'][1]:+.3f} rad, BF+DF {vv['df'][0]:+.3f} to {vv['df'][1]:+.3f} rad).\n"
             "Numbers: spoke SNR>3 limit, object refocus needed (0 = in focus) and the pupil defocus recovered by that method.\n"
             "Row label: initial pupil of all methods (posaffine calibration); EPRY keeps it, DIP and BLIS-FPM refine their own pupil. Dashed frame = before, solid frame = after.",
             fontsize=6.5, ha="center", va="top", color="0.25")
    f = W + f"d32a/fig_focusloop_{kind}_32a_{G}.png"
    fig.savefig(f, dpi=250, bbox_inches="tight"); plt.close(fig)
    return f, out, vv
res = {}
for kind in ("zoom", "field"):
    f, o, vv = figure(kind); res[kind] = dict(file=os.path.basename(f), grey_scale_rad={k: list(v) for k, v in vv.items()}, panels=o)
json.dump(res, open(W + f"d32a/fig_focusloop_images_32a_{G}.json", "w"), indent=1, default=float)
print({k: v["file"] for k, v in res.items()})
