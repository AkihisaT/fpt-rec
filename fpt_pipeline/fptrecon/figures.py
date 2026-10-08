"""Diagnostic / summary figures (matplotlib only)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi

from .optics import ZLIST, kgrid, zernike_fit

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False})
MODEL_COLORS = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]
_FIXED = {0.0: "#d95f02", 1.0: "#1b9e77"}


def col(c):
    """Colour bound to a model (FOV factor c): plane wave = orange, c = 1 = teal, others cycle."""
    c = float(c)
    if c in _FIXED:
        return _FIXED[c]
    return MODEL_COLORS[2 + int(abs(hash(round(c, 3)))) % 3]


def _rings(pos):
    r = np.round(np.hypot(pos[:, 0], pos[:, 1]), -1)
    u = np.unique(r)
    return np.searchsorted(u, r) + 1


def fig_calibration(path, R, meta, calib, cfg, twin):
    dx, kc = cfg["pixel_um"], cfg["kc"]
    kn = np.array(calib["k_solver"]); ring = _rings(meta["positions"])
    kpp = cfg["k_per_pulse_nominal"]
    N = R.shape[-1]; c = N // 2; m = int(5.5 * N * dx)
    win = np.outer(np.hanning(N), np.hanning(N))
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.5)
    th = np.linspace(0, 2 * np.pi, 300)
    picks = [int(np.where(ring == r)[0][0]) for r in np.unique(ring)[:3]]
    for j, i in enumerate(picks):
        ax = fig.add_subplot(gs[0, j])
        F = np.fft.fftshift(np.fft.fft2((R[i] - R[i].mean()) * win))
        L = ndi.gaussian_filter(np.log10(np.abs(F) ** 2 + 1e-6), 3)[c - m:c + m, c - m:c + m]
        e = m / (N * dx)
        ax.imshow(L, cmap="magma", vmin=np.percentile(L, 40), vmax=np.percentile(L, 99.8), extent=[-e, e, e, -e])
        for sg in (1, -1):
            ax.plot(sg * kn[i, 1] + kc * np.cos(th), sg * kn[i, 0] + kc * np.sin(th), "c--", lw=0.6)
        ax.set_xlim(-5.5, 5.5); ax.set_ylim(5.5, -5.5)
        ax.set_title(f"ring {ring[i]} (pos {i+1}), |k|/k$_c$={np.hypot(*kn[i])/kc:.2f}", loc="left")
        ax.set_xlabel("$q_x$ (µm$^{-1}$)")
        if j == 0: ax.set_ylabel("$q_y$ (µm$^{-1}$)")
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(kc * np.cos(th), kc * np.sin(th), "k", lw=0.8)
    cols = plt.cm.Blues(np.linspace(0.4, 1.0, ring.max()))
    x, y = meta["positions"][:, 0], meta["positions"][:, 1]
    ph = np.deg2rad(calib["rotation_deg"]); ysg = -1 if calib["mirror"] else 1
    ang = np.arctan2(ysg * y, x) + ph; r = np.hypot(x, y) * kpp
    knom = np.stack([r * np.sin(ang), r * np.cos(ang)], 1)
    for rr in range(1, ring.max() + 1):
        ii = ring == rr
        ax.scatter(twin * knom[ii, 1], twin * knom[ii, 0], s=7, facecolors="none", edgecolors=cols[rr - 1], lw=0.6)
        ax.scatter(twin * kn[ii, 1], twin * kn[ii, 0], s=7, color=cols[rr - 1], label=f"ring {rr}")
    ax.set_aspect("equal"); ax.set_xlim(-6.8, 6.8); ax.set_ylim(6.8, -6.8)
    ax.set_xlabel("$k_x$ (µm$^{-1}$)"); ax.set_ylabel("$k_y$ (µm$^{-1}$)")
    ax.set_title("illumination k (open: nominal)", loc="left"); ax.legend(frameon=False, loc="upper left")
    ax = fig.add_subplot(gs[1, 1:])
    if len(meta["Psig"]):
        kr = meta["k_radial"]; s = (kr > 0.1) & (kr < 11)
        for rr in range(1, ring.max() + 1):
            ii = ring == rr
            ax.semilogy(kr[s], ndi.uniform_filter1d(meta["Psig"][ii].mean(0), 5)[s], color=cols[rr - 1], lw=1, label=f"ring {rr}")
            ax.semilogy(kr[s], ndi.uniform_filter1d(meta["Pnoi"][ii].mean(0), 5)[s], color=cols[rr - 1], lw=0.8, ls=":")
    ax.axvspan(*cfg["nonlinear"]["q_band"], color="0.9", zorder=0); ax.axvline(kc, color="k", lw=0.6, ls="--")
    ax.set_xlabel("|q| (µm$^{-1}$)"); ax.set_ylabel("power (arb.)")
    ax.set_title("flat-fielded image (solid) vs " + ("repeat-frame noise" if cfg.get("n_repeat", 2) >= 2 else "noise floor (|q| > 2.1 kc)")
                 + " (dotted); grey = band used", loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)


def fig_reconstruction(path, phys_by_c, cfg, vT=(0.985, 1.015), vP=(-0.08, 0.08)):
    dx = cfg["pixel_um"]
    cs = sorted(phys_by_c)
    fig, axs = plt.subplots(len(cs), 4, figsize=(7.2, 1.95 * len(cs) + 0.3), gridspec_kw=dict(wspace=0.05, hspace=0.12), squeeze=False)
    H = next(iter(phys_by_c.values()))["phi"].shape[-1]
    zc = (slice(H // 2 - 200, H // 2 + 200), slice(H // 2 - 200, H // 2 + 200)); zp = (slice(60, 460), slice(60, 460))
    for i, cval in enumerate(cs):
        p = phys_by_c[cval]; cc = col(cval)
        axs[i, 0].imshow(p["T"], cmap="gray", vmin=vT[0], vmax=vT[1])
        axs[i, 1].imshow(p["phi"], cmap="gray", vmin=vP[0], vmax=vP[1])
        axs[i, 2].imshow(p["phi"][zc], cmap="gray", vmin=vP[0], vmax=vP[1])
        axs[i, 3].imshow(p["phi"][zp], cmap="gray", vmin=vP[0], vmax=vP[1])
        for ax in axs[i]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(True); s.set_color(cc); s.set_linewidth(1.2)
        for ax in axs[i, :2]:
            ax.add_patch(Rectangle((zc[1].start, zc[0].start), 400, 400, fill=False, ec="#377eb8", lw=0.8))
            ax.add_patch(Rectangle((zp[1].start, zp[0].start), 400, 400, fill=False, ec="#e7298a", lw=0.8))
        lbl = "plane-wave model" if cval == 0 else f"FOV effect c={cval:g}"
        axs[i, 0].set_ylabel(lbl, color=cc)
        for ax, L in ((axs[i, 0], 10), (axs[i, 2], 2), (axs[i, 3], 2)):
            n = ax.get_images()[0].get_array().shape[0]
            ax.plot([0.06 * n, 0.06 * n + L / dx], [0.93 * n, 0.93 * n], "w", lw=2)
            ax.text(0.06 * n + L / dx / 2, 0.90 * n, f"{L} µm", color="w", ha="center", va="bottom", fontsize=6)
    axs[0, 0].set_title(f"transmission [{vT[0]}, {vT[1]}]", loc="left"); axs[0, 1].set_title(f"phase [{vP[0]}, {vP[1]}] rad", loc="left")
    axs[0, 2].set_title("phase, centre", loc="left", color="#377eb8"); axs[0, 3].set_title("phase, FOV corner", loc="left", color="#e7298a")
    fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)


def fig_pupil(path, phys_by_c, res_by_c, cfg):
    kc = cfg["kc"]; cs = sorted(phys_by_c)
    fig = plt.figure(figsize=(7.2, 2.6))
    n = len(cs); w = 0.52 / n
    coefs = {}
    for i, cval in enumerate(cs):
        W = phys_by_c[cval]["W"]; M, dk = res_by_c[cval]["M"], res_by_c[cval]["dk"]
        coef, _ = zernike_fit(W, dk, kc); coefs[cval] = coef
        yy, xx = kgrid(M, dk); sup = np.hypot(yy, xx) <= kc
        c0, rr = M // 2, int(1.08 * kc / dk)
        ax = fig.add_axes([0.05 + i * (w + 0.01), 0.18, w, 0.66])
        im = ax.imshow(np.where(sup, W - coef[0], np.nan)[c0 - rr:c0 + rr, c0 - rr:c0 + rr], cmap="RdBu_r", vmin=-3, vmax=3,
                       extent=[-rr * dk, rr * dk, rr * dk, -rr * dk])
        ax.set_title("plane-wave model" if cval == 0 else f"FOV effect c={cval:g}", loc="left", color=col(cval))
        ax.set_xlabel("$k_x$ (µm$^{-1}$)")
        if i == 0: ax.set_ylabel("$k_y$ (µm$^{-1}$)")
        else: ax.set_yticklabels([])
    cax = fig.add_axes([0.06 + n * (w + 0.01), 0.18, 0.012, 0.66]); cb = fig.colorbar(im, cax=cax); cb.set_label("pupil phase (rad)")
    ax = fig.add_axes([0.76, 0.3, 0.235, 0.54])
    js = [3, 4, 5, 6, 7, 8, 9, 10]; x = np.arange(len(js)); bw = 0.8 / n
    for i, cval in enumerate(cs):
        ax.bar(x + (i - (n - 1) / 2) * bw, [coefs[cval][j] for j in js], bw, color=col(cval),
               label="plane wave" if cval == 0 else f"c={cval:g}")
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(x); ax.set_xticklabels([ZLIST[j][2] for j in js], rotation=45, ha="right")
    ax.set_ylabel("Zernike coeff. (rad RMS)"); ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)


def fig_fov(path, misfit_by_c, tiles_by_c):
    cs = sorted(misfit_by_c)
    fig = plt.figure(figsize=(7.2, 2.5))
    ax = fig.add_axes([0.06, 0.2, 0.25, 0.63])
    ax.plot(cs, [misfit_by_c[c] for c in cs], "o-", color="#444444", ms=3.5, lw=0.8)
    ax.set_xlabel("FOV factor c  (κ = c/(λp))"); ax.set_ylabel("band-limited misfit (lower = better)")
    ax.margins(0.1); ax.set_title("misfit vs FOV-effect strength", loc="left")
    show = [c for c in cs if c in tiles_by_c][:3]
    for j, cval in enumerate(show):
        ax = fig.add_axes([0.38 + j * 0.19, 0.2, 0.17, 0.63])
        tm = tiles_by_c[cval]
        ax.imshow(tm, cmap="viridis", vmin=0.2, vmax=1.1)
        for (ii, jj), v in np.ndenumerate(tm):
            ax.text(jj, ii, f"{v:.2f}", ha="center", va="center", fontsize=4.5, color="w" if v < 0.8 else "k")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("plane wave" if cval == 0 else f"c={cval:g}", loc="left", color=col(cval))
    fig.text(0.38, 0.08, "local misfit on 6x6 tiles", fontsize=6)
    fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)
