"""Export (physical frame) and figures for EPRY results."""
import os

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dipfpm.evaluate import band_pass, flip_k
from dipfpm.physics import ZERN, zernike_nm


def pupil_phase_unwrapped(P, W0):
    """Pupil phase relative to the initial (calibrated) phase map, added back: W = W0 + arg(P e^{-i W0})."""
    return W0 + np.angle(P * np.exp(-1j * W0))


def zernike_fit(W, dk, kc):
    M = W.shape[-1]
    yy, xx = np.mgrid[-(M // 2):M - M // 2, -(M // 2):M - M // 2] * dk
    rho = np.hypot(yy, xx) / kc; th = np.arctan2(yy, xx); m = rho <= 1
    B = np.stack([np.ones(m.sum())] + [zernike_nm(n, mm, rho[m], th[m]) for n, mm, _ in ZERN], 1)
    return np.linalg.lstsq(B, W[m], rcond=None)[0][1:]


def zern_map(zc, M, dk, kc):
    yy, xx = np.mgrid[-(M // 2):M - M // 2, -(M // 2):M - M // 2] * dk
    rho = np.hypot(yy, xx) / kc; th = np.arctan2(yy, xx)
    return sum(z * zernike_nm(n, m, np.minimum(rho, 1), th) for z, (n, m, _) in zip(zc, ZERN))


def export_epry(out_dir, tag, res, twin, cfg, W0, which="best"):
    r = res[which]; dxo = res["dxo"]; M, dk, kc = res["M"], res["dk"], cfg["kc"]
    kmin, kmax = cfg["export_band"]
    phi_raw = twin * r["phi"]; a_raw = r["a"]
    phi_bp = band_pass(phi_raw, dxo, kmin, kmax); a_bp = band_pass(a_raw, dxo, kmin, kmax)
    P = np.asarray(r["P"])
    W_s = pupil_phase_unwrapped(P, W0)
    A_s = np.abs(P)
    if twin < 0:                                  # (O, P(k)) -> (O*, P*(-k))
        W_p, A_p = -flip_k(W_s), flip_k(A_s)
    else:
        W_p, A_p = W_s, A_s
    yy, xx = np.mgrid[-(M // 2):M - M // 2, -(M // 2):M - M // 2] * dk
    inside = np.hypot(yy, xx) <= kc
    zc_phys = zernike_fit(W_p, dk, kc)
    tags = dict(resolution=(1 / dxo, 1 / dxo), metadata={"unit": "um"})
    for name, img in [("phase_rad", phi_bp), ("transmission", np.exp(-a_bp)), ("phase_rad_unfiltered", phi_raw),
                      ("transmission_unfiltered", np.exp(-a_raw))]:
        tifffile.imwrite(os.path.join(out_dir, f"{tag}_{name}.tif"), img.astype(np.float32), **tags)
    tifffile.imwrite(os.path.join(out_dir, f"{tag}_pupil_phase_rad.tif"), np.where(inside, W_p, 0).astype(np.float32))
    tifffile.imwrite(os.path.join(out_dir, f"{tag}_pupil_amplitude.tif"), A_p.astype(np.float32))
    return dict(phi=phi_bp, T=np.exp(-a_bp), a_bp=a_bp, phi_bp_solver=twin * phi_bp, zc_phys=zc_phys,
                W_phys=np.where(inside, W_p, np.nan), A_phys=A_p)


def fig_sweeps(path, res_by_tag, styles, n_fit=32, n_val=4):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.3))
    for tag, res in res_by_tag.items():
        h = res["hist"]; lb, col = styles.get(tag, (tag, "0.4"))
        axs[0].plot(h["it"], h["train_band"], color=col, lw=1.1)
        axs[0].plot(h["it"], h["val_band"], color=col, lw=1.1, ls="--")
        ib = int(np.searchsorted(h["it"], res["best"]["it"]))
        axs[0].plot(h["it"][ib], h["val_band"][ib], "o", ms=4, mfc="white", mec=col, mew=1.2)
        axs[1].semilogy(h["it"], h["alpha"], color=col, lw=1.1, label=lb)
    ends = sorted([(float(r["hist"]["val_band"][-1]), float(r["hist"]["it"][-1]), styles.get(t, (t, "0.4"))) for t, r in res_by_tag.items()])
    ys = []
    for y, _, _ in ends:
        ys.append(max(y, ys[-1] + 0.03) if ys else y)
    for (y, xe, (lb, col)), yl in zip(ends, ys):
        axs[0].text(xe * 1.02 + 0.5, yl, lb, color=col, fontsize=5, va="center")
    itmax = max(float(r["hist"]["it"][-1]) for r in res_by_tag.values())
    axs[0].set_xlim(0, itmax * 1.3 + 1)
    axs[0].set_xlabel("EPRY sweep"); axs[0].set_ylabel("in-band misfit (0.3\u20133.5 \u00b5m$^{-1}$)")
    axs[0].set_title("held-out misfit versus sweep", loc="left")
    from matplotlib.lines import Line2D
    axs[0].legend(handles=[Line2D([], [], color="0.3", lw=1.1, label=f"fitted images ({n_fit})"),
                           Line2D([], [], color="0.3", lw=1.1, ls="--", label=f"held-out images ({n_val})"),
                           Line2D([], [], color="0.3", marker="o", mfc="white", ls="", label="selected sweep")],
                  frameon=False, fontsize=5, loc="upper right")
    axs[1].set_xlabel("EPRY sweep"); axs[1].set_ylabel("step size \u03b1 (= \u03b2)")
    axs[1].set_title("adaptive step size", loc="left"); axs[1].legend(frameon=False, fontsize=5)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_pupils(path, pupils, kc):
    """pupils: label -> (W_phys, A_phys, dk, color)."""
    n = len(pupils)
    fig, axs = plt.subplots(2, n, figsize=(1.8 * n, 3.4), gridspec_kw=dict(hspace=0.15, wspace=0.08))
    axs = np.atleast_2d(axs).reshape(2, n)
    for j, (lb, (W, A, dk, col)) in enumerate(pupils.items()):
        M = W.shape[0]; h = int(1.1 * kc / dk); c = M // 2; sl = (slice(c - h, c + h), slice(c - h, c + h))
        im0 = axs[0, j].imshow(W[sl], cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        im1 = axs[1, j].imshow(A[sl], cmap="gray", vmin=0, vmax=1.3)
        axs[0, j].set_title(lb, loc="left", color=col, fontsize=6)
        for a_ in axs[:, j]: a_.set_xticks([]); a_.set_yticks([])
    axs[0, 0].set_ylabel("phase"); axs[1, 0].set_ylabel("amplitude")
    fig.colorbar(im0, ax=axs[0, :], fraction=0.02, pad=0.01, label="rad")
    fig.colorbar(im1, ax=axs[1, :], fraction=0.02, pad=0.01)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
