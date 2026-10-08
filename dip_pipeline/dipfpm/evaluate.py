"""Physical-frame conversion, export, common data-misfit metric, FRC and comparison figures."""
import json
import os

import numpy as np
import tifffile
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

from .physics import ZERN, FPMPhysics, zernike_nm

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False})
COL = {"DIP": "#7570b3", "prev_plane": "#d95f02", "prev_fov": "#1b9e77"}


def band_pass(img, dx, kmin=0.3, kmax=3.5, tlo=0.1, thi=0.3):
    N = img.shape[-1]
    k = np.fft.fftfreq(N, d=dx); KY, KX = np.meshgrid(k, k, indexing="ij"); KR = np.hypot(KY, KX)
    fh = np.clip((kmax + thi - KR) / (2 * thi), 0, 1); fh = 0.5 - 0.5 * np.cos(np.pi * fh)
    fl = np.clip((KR - (kmin - tlo)) / (2 * tlo), 0, 1); fl = 0.5 - 0.5 * np.cos(np.pi * fl)
    return np.fft.ifft2(np.fft.fft2(img) * fh * fl).real


def zern_to_physical(zc, twin):
    """Twin (O,P(k),k)->(O*,P*(-k),-k): Z_n^m(-k) = (-1)^m Z_n^m(k)  =>  z -> -(-1)^m z."""
    if twin > 0:
        return np.asarray(zc).copy()
    return np.array([-((-1) ** abs(m)) * z for z, (n, m, _) in zip(zc, ZERN)])


def pupil_map(zc, M, dk, kc):
    yy, xx = np.mgrid[-(M // 2):M - M // 2, -(M // 2):M - M // 2] * dk
    rho = np.hypot(yy, xx) / kc; th = np.arctan2(yy, xx)
    W = sum(z * zernike_nm(n, m, np.minimum(rho, 1), th) for z, (n, m, _) in zip(zc, ZERN))
    return np.where(rho <= 1, W, np.nan)


WMASK = None       # per-image validity masks (n, M, M) of dark-mode data, set by train.prepare (None = all valid)


def band_misfit(phys, a, phi, zc, shifts_um, I_meas, band_um, tuk=0.2, W_map=None, dk_illum=None, weights=None, dark_gain=False):
    """Common metric: sum_n sum_{q in band} |F{w(m_n)} - F{w(I_n - 1)}|^2 / sum |F{w(I_n-1)}|^2, m_n = I/<I> - 1.
    W_map: optional centred (M, M) pupil phase map used instead of the Zernike expansion."""
    Wm = W_map                                            # centred numpy map (or None)
    with torch.no_grad():
        I = phys(torch.tensor(a, dtype=torch.float32), torch.tensor(phi, dtype=torch.float32),
                 torch.tensor(zc, dtype=torch.float32), torch.tensor(shifts_um, dtype=torch.float32), W_map=Wm,
                 dk_illum=None if dk_illum is None else torch.tensor(dk_illum, dtype=torch.float32)).numpy()
    M = I.shape[-1]; dk = phys.dk
    w = np.outer(tukey(M, tuk), tukey(M, tuk)) if weights is None else weights
    q = np.fft.fftfreq(M, d=1.0 / (M * dk)); QY, QX = np.meshgrid(q, q, indexing="ij"); QR = np.hypot(QY, QX)
    sel = (QR >= band_um[0]) & (QR <= band_um[1])
    if dark_gain and WMASK is not None and WMASK.shape == I.shape:
        # 036 partial-coherence comparison: window x mask weighted means; dark-field images with a profiled contrast
        # factor (absorbs an additive background and the unknown illumination level); same metric for every model
        w = w * WMASK
        mu_m = (I * w).sum((1, 2), keepdims=True) / w.sum((1, 2), keepdims=True)
        mu_d = (I_meas * w).sum((1, 2), keepdims=True) / w.sum((1, 2), keepdims=True)
        dk_ = np.asarray(phys.dark)[:, None, None]
        Fm = np.fft.fft2(np.where(dk_, I - mu_m, I / mu_m - 1) * w)[:, sel]
        Fd = np.fft.fft2((I_meas / mu_d - 1) * w)[:, sel]
        cg = np.clip((np.conj(Fm) * Fd).real.sum(1) / ((np.abs(Fm) ** 2).sum(1) + 1e-30), 0, None)
        Fm = Fm * np.where(np.asarray(phys.dark), cg, 1.0)[:, None]
    elif WMASK is not None and WMASK.shape == I.shape:     # dark mode: masked weights, contrast about the masked mean
        w = w * WMASK
        mu_m = (I * WMASK).sum((1, 2), keepdims=True) / WMASK.sum((1, 2), keepdims=True)
        mu_d = (I_meas * WMASK).sum((1, 2), keepdims=True) / WMASK.sum((1, 2), keepdims=True)
        Fm = np.fft.fft2((I / mu_m - 1) * w)[:, sel]
        Fd = np.fft.fft2((I_meas / mu_d - 1) * w)[:, sel]
    else:
        Fm = np.fft.fft2((I / I.mean((1, 2), keepdims=True) - 1) * w)[:, sel]
        Fd = np.fft.fft2((I_meas - 1) * w)[:, sel]
    per = (np.abs(Fm - Fd) ** 2).sum(1) / (np.abs(Fd) ** 2).sum(1)
    return float((np.abs(Fm - Fd) ** 2).sum() / (np.abs(Fd) ** 2).sum()), per


def frc(img1, img2, dx, nbins=40, apod=0.2):
    N = img1.shape[-1]; w = np.outer(tukey(N, apod), tukey(N, apod))
    F1 = np.fft.fftshift(np.fft.fft2((img1 - img1.mean()) * w)); F2 = np.fft.fftshift(np.fft.fft2((img2 - img2.mean()) * w))
    yy, xx = np.mgrid[-(N // 2):N - N // 2, -(N // 2):N - N // 2] / (N * dx); qr = np.hypot(yy, xx)
    edges = np.linspace(0, 0.5 / dx, nbins + 1); qc = 0.5 * (edges[1:] + edges[:-1]); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (qr >= lo) & (qr < hi)
        out.append(np.real((F1[m] * np.conj(F2[m])).sum()) / np.sqrt((np.abs(F1[m]) ** 2).sum() * (np.abs(F2[m]) ** 2).sum() + 1e-30))
    return qc, np.array(out)


def export_dip(out_dir, tag, res, data, cfg, which="best"):
    os.makedirs(out_dir, exist_ok=True)
    r = res[which]; twin = data["twin"]; dx = cfg["pixel_um"]; dxo = res["dxo"]
    kmin, kmax = cfg["export_band"]
    phi_raw = twin * r["phi"]; a_raw = r["a"]
    phi_bp = band_pass(phi_raw, dxo, kmin, kmax); a_bp = band_pass(a_raw, dxo, kmin, kmax)
    Wf = np.asarray(r.get("W_free", 0.0))
    has_free = Wf.ndim == 2 and np.abs(Wf).max() > 0
    W_solver = np.nan_to_num(pupil_map(r["zc"], res["M"], res["dk"], cfg["kc"])) + (Wf if has_free else 0.0)
    W_phys = W_solver if twin > 0 else -flip_k(W_solver)
    if has_free:            # report the Zernike content of the total pupil phase
        from .train import zernike_fit_map
        zc_phys = zernike_fit_map(W_phys, res["dk"], cfg["kc"])
    else:
        zc_phys = zern_to_physical(r["zc"], twin)
    tags = dict(resolution=(1 / dxo, 1 / dxo), metadata={"unit": "um"})
    tifffile.imwrite(os.path.join(out_dir, f"{tag}_phase_rad.tif"), phi_bp.astype(np.float32), **tags)
    tifffile.imwrite(os.path.join(out_dir, f"{tag}_transmission.tif"), np.exp(-a_bp).astype(np.float32), **tags)
    tifffile.imwrite(os.path.join(out_dir, f"{tag}_phase_rad_unfiltered.tif"), phi_raw.astype(np.float32), **tags)
    tifffile.imwrite(os.path.join(out_dir, f"{tag}_transmission_unfiltered.tif"), np.exp(-a_raw).astype(np.float32), **tags)
    M = res["M"]; yy, xx = np.mgrid[-(M // 2):M - M // 2, -(M // 2):M - M // 2] * res["dk"]
    W = np.where(np.hypot(yy, xx) <= cfg["kc"], W_phys, np.nan)
    tifffile.imwrite(os.path.join(out_dir, f"{tag}_pupil_phase_rad.tif"), np.nan_to_num(W).astype(np.float32))
    return dict(phi=phi_bp, T=np.exp(-a_bp), phi_raw=phi_raw, a_raw=a_raw, zc_phys=zc_phys, W=W,
                W_solver_total=W_solver if has_free else None)


def flip_k(W):
    """W(k) -> W(-k) on a centred even grid (index i <-> (M - i) mod M)."""
    return np.roll(W[::-1, ::-1], 1, axis=(0, 1))


def load_previous(path, cfg, crop, twin):
    """BLIS-FPM (fptrecon) results: full-FOV, physical frame, band-limited, camera grid."""
    d = np.load(path)
    cy, cx, Nc = crop
    sl = (slice(cy - Nc // 2, cy + Nc // 2), slice(cx - Nc // 2, cx + Nc // 2))
    zl = ["piston", "tilt x", "tilt y", "defocus", "astig 0deg", "astig 45deg", "coma x", "coma y",
          "trefoil x", "trefoil y", "spherical", "2nd astig 0deg", "2nd astig 45deg"]
    out = {}
    for key, pk, tk, zk in [("prev_plane", "phase_planewave", "transmission_planewave", "zernike_planewave"),
                            ("prev_fov", "phase_FOV", "transmission_FOV", "zernike_FOV")]:
        zp = dict(zip(zl, d[zk]))
        z_phys = np.array([zp.get(name, 0.0) for _, _, name in ZERN])
        out[key] = dict(phi=d[pk][sl].astype(np.float64), T=d[tk][sl].astype(np.float64), zc_phys=z_phys,
                        zc_solver=zern_to_physical(z_phys, twin))
    return out


def fig_training(path, res_by_tag, styles, n_fit=32, n_val=4):
    """styles: tag -> (label, color)."""
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.32, width_ratios=[1.15, 1]))
    for tag, res in res_by_tag.items():
        h = res["hist"]; lb, col = styles.get(tag, (tag, "0.4"))
        for ax, ktr, kva in [(axs[0], "train_band", "val_band"), (axs[1], "train", "val")]:
            ax.plot(h["it"], h[ktr], color=col, lw=1.1)
            ax.plot(h["it"], h[kva], color=col, lw=1.1, ls="--")
            ib = int(np.searchsorted(h["it"], res["best"]["it"]))
            ax.plot(h["it"][ib], h[kva][ib], "o", ms=4, mfc="white", mec=col, mew=1.2)
    # direct labels at the right end of the held-out curves, de-overlapped
    ends = sorted([(float(r["hist"]["val_band"][-1]), styles.get(t, (t, "0.4"))) for t, r in res_by_tag.items()])
    ys = []
    for y, _ in ends:
        ys.append(max(y, ys[-1] + 0.03) if ys else y)
    itmax = max(float(r["hist"]["it"][-1]) for r in res_by_tag.values())
    for (y, (lb, col)), yl in zip(ends, ys):
        axs[0].text(itmax * 1.02, yl, lb, color=col, fontsize=5, va="center")
    axs[0].set_xlim(0, itmax * 1.34); axs[1].set_xlim(0, itmax * 1.03)
    hi = max(float(np.max(r["hist"]["val_band"][2:])) if len(r["hist"]["val_band"]) > 2 else 0.0 for r in res_by_tag.values())
    axs[0].set_ylim(0.15 if hi <= 0.8 else 0.0, max(0.8, 1.1 * hi))      # data-driven when held-out misfits exceed 0.8
    axs[0].set_xlabel("iteration"); axs[0].set_ylabel("in-band misfit (0.3\u20133.5 \u00b5m$^{-1}$)")
    sel = " / ".join(str(int(r["best"]["it"])) for r in res_by_tag.values())
    axs[0].set_title(f"held-out misfit is lowest at iteration {sel}", loc="left")
    axs[1].set_xlabel("iteration"); axs[1].set_ylabel("loss, data term"); axs[1].set_yscale("log")
    axs[1].set_title("training loss keeps decreasing", loc="left")
    from matplotlib.lines import Line2D
    axs[1].legend(handles=[Line2D([], [], color="0.3", lw=1.1, label=f"fitted images ({n_fit})"),
                           Line2D([], [], color="0.3", lw=1.1, ls="--", label=f"held-out images ({n_val})"),
                           Line2D([], [], color="0.3", marker="o", mfc="white", ls="", label="selected iteration")],
                  frameon=False, fontsize=5, loc="upper right")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)

def fig_compare(path, recs, cfg, frcs, metrics, vT=None, vP=None, frc_title="DIP vs BLIS-FPM", n_fit=32, n_val=4):
    """recs: label -> dict(phi, T, zc_phys, color) (physical frame, band-limited, same crop).
    frcs: label -> (q, frc, color, linestyle).
    metrics: label -> (train_mean, val_mean, color, val_is_heldout)."""
    dx = cfg["pixel_um"]; labels = list(recs); nr = len(labels)
    ref = recs[labels[-1]]
    if vT is None: vT = tuple(np.percentile(ref["T"], [0.5, 99.5]))
    if vP is None: vP = tuple(np.percentile(ref["phi"], [0.5, 99.5]))
    H = 1.62 * nr + 2.3
    fig = plt.figure(figsize=(7.2, H))
    top = fig.add_gridspec(nr, 3, left=0.2, right=0.99, top=1 - 0.25 / H, bottom=2.45 / H, hspace=0.05, wspace=0.03)
    for i, lb in enumerate(labels):
        r = recs[lb]; N = r["phi"].shape[-1]; z = slice(N // 2 - 128, N // 2 + 128)
        for j, (img, vr, ttl) in enumerate([(r["T"], vT, "transmission"), (r["phi"], vP, "phase"),
                                            (r["phi"][z, z], vP, "phase, central 8.2 \u00b5m")]):
            ax = fig.add_subplot(top[i, j]); ax.imshow(img, cmap="gray", vmin=vr[0], vmax=vr[1]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(r["color"]); sp.set_linewidth(1.4)
            if i == 0: ax.set_title(ttl, loc="left")
            if j == 0: ax.set_ylabel(lb.replace(", ", ",\n", 1), color=r["color"], rotation=0, ha="right", va="center", labelpad=4)
            n_ = img.shape[0]; L = 2.0
            ax.plot([0.06 * n_, 0.06 * n_ + L / dx], [0.93 * n_] * 2, color="w", lw=2)
            ax.text(0.06 * n_ + L / dx / 2, 0.905 * n_, "2 \u00b5m", color="w", ha="center", va="bottom", fontsize=6)
            if i == nr - 1 and j < 2:
                lo, hi = vr; unit = "" if j == 0 else " rad"
                ax.text(0.99, -0.03, f"grey scale {lo:.3f} \u2013 {hi:.3f}{unit}", transform=ax.transAxes, ha="right", va="top", fontsize=5)
    bot = fig.add_gridspec(1, 3, left=0.08, right=0.99, top=1.75 / H, bottom=0.62 / H, wspace=0.45,
                           width_ratios=[1.1, 0.9, max(1.0, 0.25 * len(metrics))])  # room for "BLIS-FPM" tick labels
    ax = fig.add_subplot(bot[0])
    for lb, (q, f, col, ls) in frcs.items():
        ax.plot(q, f, color=col, ls=ls, lw=1, label=lb)
    ax.axhline(0.143, color="0.6", lw=0.5, ls=":"); ax.set_xlim(0, 5); ax.set_ylim(-0.1, 1.02)
    ax.axvspan(*cfg["export_band"], color="0.93", zorder=0)
    ax.set_xlabel("spatial frequency |q| (\u00b5m$^{-1}$)"); ax.set_ylabel("FRC of phase")
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.42, 0.45), fontsize=5, handlelength=1.5,
              title=frc_title, title_fontsize=5)
    ax.set_title("methods agree within 0.3\u20133.5 \u00b5m$^{-1}$", loc="left")
    ax = fig.add_subplot(bot[1])
    js = [0, 1, 2, 3, 4, 7]; x = np.arange(len(js)); bw = 0.8 / nr
    for i, lb in enumerate(labels):
        ax.bar(x + (i - (nr - 1) / 2) * bw, [recs[lb]["zc_phys"][j] for j in js], bw, color=recs[lb]["color"])
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(x); ax.set_xticklabels([ZERN[j][2] for j in js], rotation=40, ha="right")
    ax.set_ylabel("Zernike coefficient (rad RMS)"); ax.set_title("pupil: defocus dominates", loc="left")
    ax = fig.add_subplot(bot[2])
    ml = list(metrics); x = np.arange(len(ml))
    for i, lb in enumerate(ml):
        tr, va, col, held = metrics[lb]
        ax.bar(i - 0.19, tr, 0.36, color=col)
        ax.bar(i + 0.19, va, 0.36, color="white", edgecolor=col, hatch="////" if held else "....", lw=0.8)
    short = [lb.replace(", ", "\n", 1)
             .replace(" model", "").replace("+ free pupil", "+pupil").replace("plane-wave", "plane") for lb in ml]
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=5)
    ax.set_ylabel("in-band misfit")
    ax.text(0.02, 0.98, "lower = better", transform=ax.transAxes, ha="left", va="top", fontsize=5, color="0.35")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="0.5", label=f"fitted images ({n_fit})"),
                       Patch(facecolor="white", edgecolor="0.5", hatch="////", label=f"held-out images ({n_val})"),
                       Patch(facecolor="white", edgecolor="0.5", hatch="....", label=f"same {n_val}, used in fit")],
              frameon=False, fontsize=5, loc="upper right")
    ax.set_ylim(0, max(max(v[0], v[1]) for v in metrics.values()) * 1.75)
    ax.set_title("data misfit, same crop and metric", loc="left")
    fig.savefig(path, dpi=300); plt.close(fig)

def load_previous_native(path, crop, Ncam, pixel_um, M, dk):
    """Unfiltered solver-frame result of the BLIS-FPM (fptrecon / run_full4) nonlinear solver:
    object (a, phi) resampled to the camera grid and cropped, free pupil-phase map interpolated to the crop k-grid."""
    from scipy.interpolate import RegularGridInterpolator
    from .physics import fourier_resample
    d = np.load(path); cy, cx, Nc = crop
    sl = (slice(cy - Nc // 2, cy + Nc // 2), slice(cx - Nc // 2, cx + Nc // 2))
    a = fourier_resample(torch.tensor(d["a"], dtype=torch.float64), Ncam).numpy()[sl]
    phi = fourier_resample(torch.tensor(d["phi"], dtype=torch.float64), Ncam).numpy()[sl]
    Mf, dkf = int(d["M"]), float(d["dk"])
    kf = (np.arange(Mf) - Mf // 2) * dkf
    f = RegularGridInterpolator((kf, kf), d["W"].astype(np.float64), bounds_error=False, fill_value=0.0)
    kq = (np.arange(M) - M // 2) * dk; KY, KX = np.meshgrid(kq, kq, indexing="ij")
    W = f(np.stack([KY.ravel(), KX.ravel()], -1)).reshape(M, M)
    return dict(a=a, phi=phi, W=W, kappa_solver=float(d["kappa"]), final_misfit_fullFOV=float(d["hist"][-1]))
