# Figures of d32a/frc/frc_mtf_32a.py: signed star MTF (512 / 1000 px) and split-half FRC (512 px), BF vs BF+DF, FOV model.
# The refocused split-half FRC uses each half's own autofocus (compare_32a.py pupil_drift_check.object_refocus).
# usage: python d32a/frc/fig_resolution_32a.py   (after frc_mtf_32a.py and the compare runs of the halves)
import json, os, sys
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
sys.path.insert(0, W + "d32a/frc"); import frc_mtf_32a as FM
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
R = json.load(open(W + "d32a/frc/frc_mtf_32a.json")); qb, qd = R["support_limits_um_inv"]["BF"], R["support_limits_um_inv"]["BF_DF"]
METH = [("DIP (optimal it.)", "DIP opt."), ("DIP (30 it.)", "DIP 30 it."), ("EPRY", "EPRY"), ("BLIS-FPM", "BLIS-FPM")]
CB, CD = "#1f5fa6", "#d95f02"; LAB = {"BF": "BF (rings 0-2)", "BF+DF": "BF+DF (+ rings 6-7)"}
def limits(ax, label=False, ytop=1.3):
    ax.axvline(qb, color="0.5", lw=0.6, ls="--"); ax.axvline(qd, color="0.5", lw=0.6, ls="-.")
    if label:
        ax.text(qb - 0.07, ytop, "BF limit", rotation=90, ha="right", va="top", fontsize=6, color="0.4")
        ax.text(qd - 0.07, ytop, "DF limit", rotation=90, ha="right", va="top", fontsize=6, color="0.4")
def fig_mtf(g):
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.06, hspace=0.14))
    for i, (kind, nk, row) in enumerate((("signed", "noise", "as reconstructed"), ("signed_refocused", "noise_refocused", "after object refocus"))):
        for j, (m, ml) in enumerate(METH):
            ax = axs[i, j]; zz = []
            for s, col in (("BF", CB), ("BF+DF", CD)):
                d = R["mtf"][f"{g}|{s}|{m}, FOV model"]; q = np.array(d["q"]); nz = np.array(d[nk])
                ax.fill_between(q, -3 * nz, 3 * nz, color=col, alpha=0.10, lw=0); ax.plot(q, d[kind], color=col, lw=1.1, label=LAB[s]); zz.append(d["refocus_um"][0] / 1e3)
            ax.axhline(0, color="0.3", lw=0.5); limits(ax, label=(i == 0 and j == 1), ytop=1.3); ax.set_xlim(0.8, 6); ax.set_ylim(-0.65, 1.35)
            if i == 0: ax.set_title(ml, loc="left")
            if i == 1: ax.text(0.97, 0.97, f"refocus BF {zz[0]:+.2f} mm\nBF+DF {zz[1]:+.2f} mm", transform=ax.transAxes, ha="right", va="top", fontsize=6)
            if j == 0: ax.set_ylabel(f"{row}\nsigned star MTF")
    fig.supxlabel("spatial frequency (µm⁻¹)", fontsize=8, y=0.01); axs[0, 0].legend(frameon=False, loc="upper right", fontsize=6)
    fig.savefig(W + f"d32a/frc/fig_mtf_signed_32a_{g}.png", dpi=250, bbox_inches="tight")
def fig_frc():
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.06, hspace=0.14))
    for i, (src, row) in enumerate(((R["frc"], "as reconstructed"), (R["frc_refocused"], "after object refocus"))):
        for j, (m, ml) in enumerate(METH):
            ax = axs[i, j]
            for s, col, qcap in (("BF", CB, qb), ("BF+DF", CD, qd)):
                d = src[f"{s}|{m}"]; q = np.array(d["q"]); f = np.array(d["frc"]); inn = q <= qcap
                ax.plot(q[inn], f[inn], color=col, lw=1.1, label=LAB[s]); ax.plot(q[~inn], f[~inn], color=col, lw=0.7, alpha=0.3)
                c = R["frc_limits_capped"][f"{row}|{s}|{m}"]
                if not c["no_crossing_below_support_limit"]: ax.plot([c["q"]], [np.interp(c["q"], q, d["halfbit"])], "o", ms=3.2, color=col, mec="none", zorder=5)
            ax.plot(q, d["halfbit"], color="0.2", lw=0.6, ls=(0, (1, 1)), label="half-bit threshold"); limits(ax, label=(i == 0 and j == 1), ytop=1.02); ax.axhline(0, color="0.3", lw=0.4)
            ax.set_xlim(0, 6); ax.set_ylim(-0.25, 1.05)
            if i == 0: ax.set_title(ml, loc="left")
            if j == 0: ax.set_ylabel(f"{row}\nsplit-half FRC (phase)")
    fig.supxlabel("spatial frequency (µm⁻¹)", fontsize=8, y=0.01); axs[1, 0].legend(frameon=False, loc="lower left", fontsize=6)
    fig.savefig(W + "d32a/frc/fig_frc_32a_512.png", dpi=250, bbox_inches="tight")
if __name__ == "__main__":
    fig_mtf("512"); fig_mtf("1000"); fig_frc()
