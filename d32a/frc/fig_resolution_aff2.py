# 32a resolution before / after moving the residual object defocus into the pupil (posaffine focus loop, "aff2").
# Inputs: d32a/frc/frc_mtf_32a.json (aff) and d32a/frc/frc_mtf_32a_aff2.json (aff2), both from frc_mtf_32a.py [tag].
# Outputs: d32a/frc/fig_mtf_signed_aff_vs_aff2_{512,1000}.png, d32a/frc/fig_frc_aff_vs_aff2_512.png,
#          d32a/frc/resolution_table_32a_aff2.json
# usage: python d32a/frc/fig_resolution_aff2.py
import json, os
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
A = json.load(open(W + "d32a/frc/frc_mtf_32a.json")); B = json.load(open(W + "d32a/frc/frc_mtf_32a_aff2.json"))
qb, qd = A["support_limits_um_inv"]["BF"], A["support_limits_um_inv"]["BF_DF"]
METH = [("DIP (optimal it.)", "DIP opt."), ("DIP (30 it.)", "DIP 30 it."), ("EPRY", "EPRY"), ("BLIS-FPM", "BLIS-FPM")]
COL = {"BF": "#1f5fa6", "BF+DF": "#d95f02"}; LAB = {"BF": "BF (rings 0-2)", "BF+DF": "BF+DF (+ rings 6-7)"}


def limits(ax, label=False, ytop=1.3):
    ax.axvline(qb, color="0.5", lw=0.6, ls="--"); ax.axvline(qd, color="0.5", lw=0.6, ls="-.")
    if label:
        ax.text(qb - 0.07, ytop, "BF limit", rotation=90, ha="right", va="top", fontsize=6, color="0.4")
        ax.text(qd - 0.07, ytop, "DF limit", rotation=90, ha="right", va="top", fontsize=6, color="0.4")


def fig_mtf(g):
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.06, hspace=0.16))
    for i, s in enumerate(("BF", "BF+DF")):
        for j, (m, ml) in enumerate(METH):
            ax = axs[i, j]; key = f"{g}|{s}|{m}, FOV model"
            if key not in A["mtf"] or key not in B["mtf"]: ax.set_visible(False); continue
            a, b = A["mtf"][key], B["mtf"][key]; col = COL[s]
            ax.fill_between(b["q"], -3 * np.array(b["noise"]), 3 * np.array(b["noise"]), color=col, alpha=0.10, lw=0)
            ax.plot(a["q"], a["signed"], color="0.55", lw=0.8, ls="--", label="before (aff), as reconstructed")
            ax.plot(a["q"], a["signed_refocused"], color=col, lw=0.8, ls=":", label="before (aff), object refocused afterwards")
            ax.plot(b["q"], b["signed"], color=col, lw=1.2, label="focus in pupil (aff2), as reconstructed")
            ax.axhline(0, color="0.3", lw=0.5); limits(ax, label=(i == 0 and j == 1)); ax.set_xlim(0.8, 6); ax.set_ylim(-0.65, 1.35)
            ax.text(0.97, 0.97, f"refocus aff {a['refocus_um'][0] / 1e3:+.2f} mm\naff2 {b['refocus_um'][0] / 1e3:+.2f} mm",
                    transform=ax.transAxes, ha="right", va="top", fontsize=5.5)
            if i == 0: ax.set_title(ml, loc="left")
            if j == 0: ax.set_ylabel(f"{LAB[s]}\nsigned star MTF")
    fig.supxlabel("spatial frequency (µm⁻¹)", fontsize=8, y=0.01)
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc="upper center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, 1.03))
    fig.savefig(W + f"d32a/frc/fig_mtf_signed_aff_vs_aff2_{g}.png", dpi=250, bbox_inches="tight")


def fig_frc():
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.06, hspace=0.16))
    for i, (s, qcap) in enumerate((("BF", qb), ("BF+DF", qd))):
        for j, (m, ml) in enumerate(METH):
            ax = axs[i, j]; k = f"{s}|{m}"
            if k not in B["frc"]: ax.set_visible(False); continue
            col = COL[s]
            for src, lw, ls, c_, lab in ((A["frc"], 0.8, "--", "0.55", "before (aff), as reconstructed"), (A["frc_refocused"], 0.8, ":", col, "before (aff), halves refocused afterwards"),
                                         (B["frc"], 1.2, "-", col, "focus in pupil (aff2), as reconstructed")):
                d = src[k]; q = np.array(d["q"]); f = np.array(d["frc"]); inn = q <= qcap
                ax.plot(q[inn], f[inn], color=c_, lw=lw, ls=ls, label=lab); ax.plot(q[~inn], f[~inn], color=c_, lw=0.5, ls=ls, alpha=0.3)
            ax.plot(q, d["halfbit"], color="0.2", lw=0.6, ls=(0, (1, 1)), label="half-bit threshold")
            c = B["frc_limits_capped"][f"as reconstructed|{s}|{m}"]
            if not c["no_crossing_below_support_limit"]: ax.plot([c["q"]], [np.interp(c["q"], q, d["halfbit"])], "o", ms=3.2, color=col, mec="none", zorder=5)
            limits(ax, label=(i == 0 and j == 1), ytop=1.02); ax.axhline(0, color="0.3", lw=0.4); ax.set_xlim(0, 6); ax.set_ylim(-0.25, 1.05)
            if i == 0: ax.set_title(ml, loc="left")
            if j == 0: ax.set_ylabel(f"{LAB[s]}\nsplit-half FRC (phase)")
    fig.supxlabel("spatial frequency (µm⁻¹)", fontsize=8, y=0.01)
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc="upper center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, 1.03))
    fig.savefig(W + "d32a/frc/fig_frc_aff_vs_aff2_512.png", dpi=250, bbox_inches="tight")


def table():
    T = {}
    for g in ("512", "1000"):
        for s in ("BF", "BF+DF"):
            for m, _ in METH:
                key = f"{g}|{s}|{m}, FOV model"
                if key in A["mtf"] and key in B["mtf"]:
                    T[f"MTF10|{key}"] = dict(aff=A["mtf"][key]["numbers"]["q_mtf10"], aff_refocused=A["mtf"][key]["numbers_refocused"]["q_mtf10"],
                                             aff2=B["mtf"][key]["numbers"]["q_mtf10"], aff2_refocused=B["mtf"][key]["numbers_refocused"]["q_mtf10"],
                                             df_band_aff=A["mtf"][key]["numbers"]["df_band"], df_band_aff_refocused=A["mtf"][key]["numbers_refocused"]["df_band"],
                                             df_band_aff2=B["mtf"][key]["numbers"]["df_band"], df_band_noise_aff2=B["mtf"][key]["numbers"]["df_band_noise"],
                                             refocus_um_aff=A["mtf"][key]["refocus_um"], refocus_um_aff2=B["mtf"][key]["refocus_um"])
    for row in ("as reconstructed", "after object refocus"):
        for s in ("BF", "BF+DF"):
            for m, _ in METH:
                k = f"{row}|{s}|{m}"
                if k in A["frc_limits_capped"] and k in B["frc_limits_capped"]:
                    T[f"FRC_halfbit|{k}"] = dict(aff=A["frc_limits_capped"][k], aff2=B["frc_limits_capped"][k])
    json.dump(T, open(W + "d32a/frc/resolution_table_32a_aff2.json", "w"), indent=1)
    return T


if __name__ == "__main__":
    fig_mtf("512"); fig_mtf("1000"); fig_frc(); T = table()
    for k, v in T.items():
        if k.startswith("MTF10|512"): print(k, {kk: (round(x, 2) if isinstance(x, float) else x) for kk, x in v.items() if not kk.startswith("refocus")})
    for k, v in T.items():
        if k.startswith("FRC"): print(k, round(v["aff"]["q"], 2), v["aff"]["no_crossing_below_support_limit"], "->", round(v["aff2"]["q"], 2), v["aff2"]["no_crossing_below_support_limit"])
