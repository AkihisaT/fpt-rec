# 32a: moving the residual object defocus into the pupil (posaffine focus loop, "aff2") -- focus bookkeeping figure.
# (a) focus loop of posaffine (pupil fixed, object solved), (b) object refocus by method (aff vs aff2),
# (c) recovered pupil defocus by method vs the defocus that the kept parallax corresponds to,
# (d) radial phase profile of the free pupil of the posaffine check (piston, tilt, astigmatism removed).
# Inputs: fpt_output_32a_bf_aff{,2}/results/position_affine.json, d32a/out_32a{bf,df}aff{,2}_512/compare_summary.json,
#         d32a/af_test/pupil_check_{aff,aff2}.npz (d32a/af_test/pupil_basis_test.py).  Output: d32a/fig_focusloop_32a.png,
#         d32a/focusloop_summary_32a.json.   usage: python d32a/fig_focusloop_32a.py
import json, os, sys
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, W + "fpt_pipeline"); from fptrecon.optics import kgrid
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
LAM = 1.239842 / 30 * 1e-3; KC = 2.0
C0, C1 = "0.6", "#1b7837"                                  # aff (before), aff2 (focus in pupil)
PA = {t: json.load(open(W + f"fpt_pipeline/fpt_output_32a_bf_{t}/results/position_affine.json")) for t in ("aff", "aff2")}
METH = [("DIP (optimal it.)", "DIP opt."), ("DIP (30 it.)", "DIP 30"), ("EPRY", "EPRY"), ("BLIS-FPM", "BLIS")]
def cs(ds):
    f = W + f"d32a/out_{ds}/compare_summary.json"
    return json.load(open(f))["results"] if os.path.exists(f) else None
S = {(s, t): cs(f"32a{s}{t}_512") for s in ("bf", "df") for t in ("aff", "aff2")}
out = dict(focus_loop=PA["aff2"]["focus_loop"], aligned_stack=PA["aff2"]["aligned_stack"], check_aff2=PA["aff2"]["check"], check_aff=PA["aff"]["check"],
           mechanical=dict(aff=PA["aff"]["mechanical"], aff2=PA["aff2"]["mechanical"]), methods={})
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.0), gridspec_kw=dict(wspace=0.32, hspace=0.62)); axs = axs.ravel()
# (a)
ax = axs[0]; L = PA["aff2"]["focus_loop"]; x = np.arange(len(L))
ax.plot(x, [l["pupil_defocus_um"] / 1e3 for l in L], "o-", color=C1, ms=3.5, lw=1, label="pupil (fixed)")
ax.plot(x, [l["object_refocus"]["z_um"] / 1e3 for l in L], "s--", color="#b2182b", ms=3.2, lw=1, label="object refocus")
for xi, l in zip(x, L): ax.text(xi, 0.35, f"misfit\n{l['misfit']:.4f}", ha="center", va="bottom", fontsize=5.5, color="0.35")
ax.axhline(0, color="0.3", lw=0.5); ax.set_xticks(x); ax.set_xticklabels([f"pass {i}" for i in x]); ax.set_xlim(-0.5, len(L) - 0.5); ax.set_ylim(-3.2, 1.9)
ax.set_ylabel("defocus (mm)"); ax.set_title("a  posaffine focus loop", loc="left"); ax.legend(frameon=False, fontsize=5.5, loc="lower left")
# (b), (c)
for ax, key, title in ((axs[1], "refocus", "b  object refocus needed (512 px, FOV model)"), (axs[2], "pupil", "c  pupil defocus of each reconstruction")):
    xt, xl = [], []
    for si, s in enumerate(("bf", "df")):
        for mi, (m, ml) in enumerate(METH):
            xx = si * 5 + mi; xt.append(xx); xl.append(ml)
            for t, col, dx in (("aff", C0, -0.19), ("aff2", C1, 0.19)):
                R = S[(s, t)]
                if R is None or f"{m}, FOV model" not in R: continue
                p = R[f"{m}, FOV model"]["pupil_drift_check"]
                v = p["object_refocus"]["z_um"] / 1e3 if key == "refocus" else p["pupil_defocus_um"] / 1e3
                ax.bar(xx + dx, v, 0.36, color=col, label=("before (aff)" if t == "aff" else "focus in pupil (aff2)") if (si, mi) == (0, 0) else None)
                out["methods"].setdefault(f"{s}|{m}", {})[f"{key}_{t}_mm"] = v
    ax.axhline(0, color="0.3", lw=0.5); ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=90); ax.set_title(title, loc="left")
    ax.text(1.5, 0.97, "BF", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, fontweight="bold"); ax.text(6.5, 0.97, "BF+DF", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, fontweight="bold"); ax.axvline(4, color="0.8", lw=0.5)
    ax.set_ylabel("mm")
    if key == "pupil":
        for t, col in (("aff", C0), ("aff2", C1)):
            ax.axhline(PA[t]["pupil_defocus_um"] / 1e3, color=col, lw=0.8, ls="--")
        ax.text(8.6, PA["aff2"]["pupil_defocus_um"] / 1e3, "kept parallax\n(aff2)", fontsize=5.3, color=C1, va="top", ha="right")
        ax.text(8.6, PA["aff"]["pupil_defocus_um"] / 1e3, "kept parallax (aff)", fontsize=5.3, color="0.4", va="bottom", ha="right")
        ax.set_ylim(-3.1, 0.4)
    else:
        ax.set_ylim(-0.3, 1.75); ax.legend(frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(0.0, 0.9))
# (d)
ax = axs[3]; prof = {}
for t, col in (("aff", C0), ("aff2", C1)):
    z = np.load(W + f"d32a/af_test/pupil_check_{t}.npz"); Wp, Pa, dk = z["W"], z["Pamp"], float(z["dk"])
    yy, xx_ = kgrid(Wp.shape[0], dk); r = np.hypot(yy, xx_); m = np.isfinite(Wp) & (Pa > 0.5) & (r < 0.95 * KC)
    X = np.stack([np.ones(m.sum()), yy[m], xx_[m], (xx_ ** 2 - yy ** 2)[m], (2 * xx_ * yy)[m], (yy ** 2 + xx_ ** 2)[m]], 1)
    c = np.linalg.lstsq(X, Wp[m], rcond=None)[0]; Wd = np.full_like(Wp, np.nan); Wd[m] = Wp[m] - X[:, :5] @ c[:5]
    e = np.arange(0, 0.951 * KC, 0.05); rc = 0.5 * (e[1:] + e[:-1])
    pr = np.array([np.nanmean(Wd[(r >= a) & (r < b) & m]) for a, b in zip(e[:-1], e[1:])]); pr -= np.interp(0.5, rc, pr)
    ax.plot(rc, pr, color=col, lw=1.2, label=f"free pupil, {t}"); prof[t] = dict(r=rc.tolist(), phase_rad=pr.tolist())
for zz, ls in ((-1.52e3, ":"), (-2.72e3, "--")):
    ax.plot(rc, np.pi * LAM * zz * (rc ** 2 - 0.25), color="k", lw=0.7, ls=ls, label=f"quadratic {zz / 1e3:.2f} mm")
kn = np.array(json.load(open(W + "fpt_pipeline/fpt_output_32a_bf_aff2/work/calibration.json"))["k_solver"]); kr = np.hypot(*kn.T)
ax.axvspan(kr.min(), kr.max(), color="0.9", lw=0, zorder=0); ax.text(0.5 * (kr.min() + kr.max()), 0.95, "direct beams\n(BF rings)", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.3, color="0.4")
ax.set_xlabel("pupil radius |k| (µm⁻¹)"); ax.set_ylabel("phase (rad)"); ax.set_title("d  free pupil of the posaffine check", loc="left"); ax.legend(frameon=False, fontsize=5, loc="lower left")
ax.set_xlim(0, 1.95)
out["free_pupil_profile"] = prof
fig.savefig(W + "d32a/fig_focusloop_32a.png", dpi=250, bbox_inches="tight")
json.dump(out, open(W + "d32a/focusloop_summary_32a.json", "w"), indent=1, default=float)
print(json.dumps(out["methods"], indent=0))
