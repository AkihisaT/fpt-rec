"""Step 5: twin selection, physical-frame conversion, band limiting, metrics and export."""
import json
import os

import numpy as np
import tifffile
from scipy import ndimage as ndi

from .optics import ZLIST, band_limit, flip_k, fresample, kgrid, zernike_fit


def twin_sign(a, phi, frac=0.27):
    """+1 if the solver solution is already the physical one, -1 if it is the conjugate twin.
    Physical: material absorbs (a > 0) and retards (phi < 0)  ->  corr(a, phi) < 0."""
    H = a.shape[-1]; m = int(frac * H)
    A = ndi.gaussian_filter(a[m:H - m, m:H - m], 2); P = ndi.gaussian_filter(phi[m:H - m, m:H - m], 2)
    c = np.corrcoef(A.ravel(), P.ravel())[0, 1]
    return (1 if c < 0 else -1), float(c)


def to_physical(res, twin, cfg, Ncam):
    """Solver result -> physical frame, band-limited, resampled to the camera grid."""
    kmin, kmax = cfg["export_band"]
    a = band_limit(res["a"], res["dxo"], kmin, kmax)
    phi = twin * band_limit(res["phi"], res["dxo"], kmin, kmax)
    a_c, phi_c = fresample(a, Ncam), fresample(phi, Ncam)
    W = res["W"] if twin > 0 else -flip_k(res["W"])
    W0 = res["W0"] if twin > 0 else -flip_k(res["W0"])
    Pamp = res["Pamp"] if twin > 0 else flip_k(res["Pamp"])
    return dict(T=np.exp(-a_c).astype(np.float32), a=a_c.astype(np.float32), phi=phi_c.astype(np.float32),
                W=W, W0=W0, Pamp=Pamp)


def tile_misfit(Imod, Imeas, band_px, nt=6):
    M = Imod.shape[-1]; T = M // nt
    wt = np.outer(np.hanning(T), np.hanning(T))
    qy, qx = np.mgrid[-(T // 2):T - T // 2, -(T // 2):T - T // 2] * (M / T)      # also correct for odd T
    qm = np.fft.ifftshift(np.hypot(qy, qx)); sel = (qm >= band_px[0]) & (qm <= band_px[1])
    out = np.zeros((nt, nt))
    for i in range(nt):
        for j in range(nt):
            sl = (slice(None), slice(i * T, (i + 1) * T), slice(j * T, (j + 1) * T))
            d, m = Imeas[sl], Imod[sl]
            Fd = np.fft.fft2((d - d.mean((1, 2), keepdims=True)) * wt)[:, sel]
            Fm = np.fft.fft2((m - m.mean((1, 2), keepdims=True)) * wt)[:, sel]
            out[i, j] = (np.abs(Fd - Fm) ** 2).sum() / (np.abs(Fd) ** 2).sum()
    return out


def export(out_dir, tag, phys, res, cfg, extra=None):
    """Write TIFFs (transmission, phase, pupil phase/amplitude) and a JSON summary for one model."""
    os.makedirs(out_dir, exist_ok=True)
    dx, kc = cfg["pixel_um"], cfg["kc"]
    res_tag = dict(resolution=(1 / dx, 1 / dx), metadata={"unit": "um"})
    tifffile.imwrite(os.path.join(out_dir, f"{tag}_transmission.tif"), phys["T"], **res_tag)
    tifffile.imwrite(os.path.join(out_dir, f"{tag}_phase_rad.tif"), phys["phi"], **res_tag)
    M, dk = res["M"], res["dk"]
    yy, xx = kgrid(M, dk)
    sup = np.hypot(yy, xx) <= kc
    coef, _ = zernike_fit(phys["W"], dk, kc)
    c0, rr = M // 2, int(np.ceil(1.08 * kc / dk))
    Wc = np.where(sup, phys["W"] - coef[0], 0)[c0 - rr:c0 + rr, c0 - rr:c0 + rr].astype(np.float32)
    Ac = phys["Pamp"][c0 - rr:c0 + rr, c0 - rr:c0 + rr].astype(np.float32)
    tifffile.imwrite(os.path.join(out_dir, f"{tag}_pupil_phase_rad.tif"), Wc)
    tifffile.imwrite(os.path.join(out_dir, f"{tag}_pupil_amplitude.tif"), Ac)
    tm = tile_misfit(res["Imod"], res["Imeas"], res["band"])
    lam = cfg["lam_um"]
    summary = dict(tag=tag, final_misfit=float(res["hist"][-1]) if len(res["hist"]) else float(res["L0"]),
                   initial_misfit=float(res["L0"]), n_iter=int(len(res["hist"])), images_used=res["keep"].tolist(),
                   zernike_rad_rms={ZLIST[j][2]: float(coef[j]) for j in range(len(ZLIST))},
                   defocus_equiv_um=float(coef[3] * 2 * np.sqrt(3) / (np.pi * lam * kc ** 2)),
                   tile_misfit_6x6=tm.round(4).tolist(),
                   pupil_pixel_um_inv=float(dk), pupil_crop_px=int(2 * rr), pixel_um=dx)
    if extra:
        summary.update(extra)
    with open(os.path.join(out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary, tm, coef
