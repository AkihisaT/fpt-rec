"""Data preparation and Deep-Image-Prior training loop."""
import copy
import json
import os
import time

import numpy as np
import torch
from scipy.fft import next_fast_len
from scipy.signal.windows import tukey

from .models import ParamMLP, UNet
from .physics import ZERN, FPMPhysics, fourier_resample, zernike_nm


def fast_even(n):
    n = int(np.ceil(n))
    while True:
        m = next_fast_len(n)
        if m % 2 == 0:
            return m
        n = m + 1


def fcrop(I, M):
    """Fourier-crop real images (..., N, N) -> (..., M, M), values preserved."""
    N = I.shape[-1]
    F = np.fft.fftshift(np.fft.fft2(I, norm="ortho"), axes=(-2, -1))
    c, h = N // 2, M // 2
    Fc = F[..., c - h:c - h + M, c - h:c - h + M]
    return (np.fft.ifft2(np.fft.ifftshift(Fc, axes=(-2, -1)), norm="ortho").real * (M / N)).astype(np.float32)


def zernike_fit_map(W, dk, kc):
    M = W.shape[-1]
    yy, xx = np.mgrid[-(M // 2):M - M // 2, -(M // 2):M - M // 2] * dk
    rho = np.hypot(yy, xx) / kc; th = np.arctan2(yy, xx); m = rho <= 1
    B = np.stack([np.ones(m.sum())] + [zernike_nm(n, mm, rho[m], th[m]) for n, mm, _ in ZERN], 1)
    coef, *_ = np.linalg.lstsq(B, W[m], rcond=None)
    return coef[1:]


def prepare(cfg, log=print):
    """Load fptrecon preprocessing/calibration, crop, split train/validation, build grids."""
    d = cfg["dip"]; lam, dx, kc = cfg["lam_um"], cfg["pixel_um"], cfg["kc"]
    R = np.load(os.path.join(cfg["fpt_work_dir"], "preprocessed.npy"), mmap_mode="r")
    cal = json.load(open(os.path.join(cfg["fpt_work_dir"], "calibration.json")))
    kn = np.array(cal["k_solver"]); twin = int(cal.get("twin", 1))
    H = R.shape[-1]; Nc = d["crop_px"]
    cy, cx = d["crop_center"] or (H // 2, H // 2)
    Rc = np.array(R[:, cy - Nc // 2:cy + Nc // 2, cx - Nc // 2:cx + Nc // 2], dtype=np.float32)
    dark_mode = bool(d.get("dark_mode", False))
    # dark_mode (036 Phase C): images with |k_n| > pupil_radius kc (direct beam outside the pupil) are dark-field data on
    # a common scale (units of the bright-field level) -> not normalised per image; validity masks from dfmask.npy
    darkall = (np.hypot(kn[:, 0], kn[:, 1]) > d["pupil_radius"] * kc) if dark_mode else np.zeros(len(kn), bool)
    Rc = np.clip(Rc, *cfg["clip"])
    Rc[~darkall] /= Rc[~darkall].mean((1, 2), keepdims=True)
    keep = np.where(np.hypot(kn[:, 0], kn[:, 1]) / kc <= d["max_k_over_kc"])[0]
    if d.get("exclude_images"):          # e.g. 036 Phase C: ring-2 edge images (direct-beam tail inside the pupil)
        keep = np.setdiff1d(keep, np.asarray(d["exclude_images"], int))
    val = keep[::d["val_every"]] if d["val_every"] else np.array([], int)
    train = np.setdiff1d(keep, val)
    dk = 1.0 / (Nc * dx)
    M = d["M"] or fast_even(4 * d["pupil_radius"] * kc / dk)             # alias-free |psi|^2
    smax = np.abs(kn[keep] / dk).max()
    if d.get("partial_coherence"):       # room for the illumination modes (offsets up to ~1.1 kc)
        smax += 1.1 * kc / dk
    No = d["No"] or fast_even(max(M + 2 * np.ceil(smax) + 8, Nc))
    Nu = d["unet_px"]
    I_meas = fcrop(Rc[keep], M)
    dk_ = darkall[keep]
    I_meas[~dk_] /= I_meas[~dk_].mean((1, 2), keepdims=True)
    x_in = fcrop(Rc[train], Nu)
    if dark_mode:        # per-image contrast, standardised per image (bright and dark-field inputs on equal footing)
        x_in = x_in / x_in.mean((1, 2), keepdims=True) - 1.0; x_in /= x_in.std((1, 2), keepdims=True)
    else:
        x_in = (x_in - 1.0) / x_in.std()
    wmask = None
    if dark_mode:
        from scipy.ndimage import zoom
        mk = np.load(os.path.join(cfg["fpt_work_dir"], "dfmask.npy"), mmap_mode="r")
        wmask = np.stack([zoom(np.asarray(mk[k, cy - Nc // 2:cy + Nc // 2, cx - Nc // 2:cx + Nc // 2], np.float32), M / Nc, order=1)
                          for k in keep]).astype(np.float32)
        wmask[~dk_] = 1.0
        log(f"dark mode: {int(dk_.sum())} dark-field images (|k| > {d['pupil_radius']} kc), masked fraction (mask < 0.5) "
            f"max {float((wmask < 0.5).mean((1, 2)).max()):.3f}")
    img_w = None
    if dark_mode and d.get("image_weights") == "noise":
        # Gaussian likelihood weights: lambda_n = sigma2_ref / sigma2_n (common intensity units, from the half averages);
        # ring-1 (bright) images keep weight 1
        nz = json.load(open(os.path.join(cfg["fpt_work_dir"], "noise.json")))
        s2 = np.array(nz["sigma2_512" if Nc <= 512 else "sigma2_1024"])
        ref = np.median(s2[~darkall])
        img_w = np.where(darkall, ref / s2, 1.0)[keep].astype(np.float32)
        log(f"dark mode: image weights (noise) dark median {float(np.median(img_w[dk_])):.1f}, range {float(img_w[dk_].min()):.1f}-{float(img_w[dk_].max()):.1f}")
    from . import evaluate as _E
    _E.WMASK = wmask
    # calibrated pupil -> Zernike initial coefficients (solver frame)
    yy, xx = np.mgrid[-(M // 2):M - M // 2, -(M // 2):M - M // 2] * dk
    z, (a0, a1) = cal["defocus_um"], cal["astig_um"]
    W0 = np.pi * lam * z * (yy ** 2 + xx ** 2) + np.pi * lam * (a0 * (xx ** 2 - yy ** 2) + a1 * 2 * xx * yy)
    z0 = zernike_fit_map(W0, dk, kc)
    axis_off = ((cy - H / 2) * dx, (cx - H / 2) * dx)
    log(f"crop {Nc} px at ({cy},{cx}); images: train {len(train)}, validation {len(val)} (of {len(kn)}); "
        f"M={M}, No={No} (object pixel {Nc*dx/No*1e3:.1f} nm), U-Net grid {Nu}; twin={twin:+d}")
    return dict(Rc=Rc, keep=keep, train=train, val=val, kn=kn, twin=twin, dk=dk, M=M, No=No, Nu=Nu, wmask=wmask, dark=dk_, img_w=img_w,
                I_meas=I_meas, x_in=x_in, z0=z0, axis_off=axis_off, cal=cal, crop=(cy, cx, Nc), H=H)


def run_dip(cfg, data, kappa_solver, log=print, checkpoint=None):
    d = cfg["dip"]; torch.manual_seed(d["seed"]); np.random.seed(d["seed"])
    torch.set_num_threads(int(d["threads"]))
    keep, train, val = data["keep"], data["train"], data["val"]
    pos = {k: i for i, k in enumerate(keep)}
    itr = np.array([pos[k] for k in train]); iva = np.array([pos[k] for k in val], int)
    n = len(keep)
    from .coherence import pc_modes
    md, mw = pc_modes(d.get("partial_coherence"), data["kn"][keep], cfg["kc"], d["pupil_radius"])
    phys = FPMPhysics(data["kn"][keep], data["dk"], data["M"], data["No"], cfg["kc"], cfg["lam_um"],
                      kappa=kappa_solver, axis_offset_um=data["axis_off"], pupil_radius=d["pupil_radius"], modes=md, mode_w=mw)
    pc_every = int(d.get("pc_direct_every", 5))
    if phys.pc:
        phys.cache_on = pc_every > 1
        log(f"  partial coherence {d['partial_coherence']}: modes per image mean {float((mw > 0).sum(1).mean()):.1f} "
            f"(min {int((mw > 0).sum(1).min())}, max {int((mw > 0).sum(1).max())}); direct beam refreshed every {pc_every} it")
    net = UNet(len(train), base=d["unet_base"], depth=d["unet_depth"], bn=d["batchnorm"])
    mlp_z = ParamMLP(len(ZERN), init=data["z0"], scale=d["zernike_scale"])
    mlp_s = ParamMLP(2 * n, scale=d["shift_scale_px"])            # training images
    mlp_c = ParamMLP(n, scale=d["intensity_scale"])
    mlp_sv = ParamMLP(2 * n, scale=d["shift_scale_px"])           # held-out images: fitted with the object frozen
    mlp_cv = ParamMLP(n, scale=d["intensity_scale"])
    tmask = torch.zeros(n); tmask[itr] = 1.0
    vmask = torch.zeros(n); vmask[iva] = 1.0
    smax = float(d.get("shift_max", 5.0))
    x_in = torch.tensor(data["x_in"][None])
    I_meas = torch.tensor(data["I_meas"])
    M = data["M"]
    w = torch.tensor(np.outer(tukey(M, d["loss_tukey"]), tukey(M, d["loss_tukey"])).astype(np.float32))
    # FOV model: pixels where the model puts the direct beam outside the pupil (I_Q < dir_floor * max) cannot be
    # represented by the flat-field-normalised model -> excluded from loss and metric (smoothed mask)
    wn = vignetting_weights(phys, torch.tensor(data["z0"], dtype=torch.float32), w) \
        if (kappa_solver != 0 and d.get("mask_vignetted", True)) else w[None].expand(n, M, M)
    if data.get("wmask") is not None:
        wn = wn * torch.tensor(data["wmask"])
    if kappa_solver != 0:
        log(f"  vignetting mask: excluded fraction per image max {float(1 - (wn > 0.5 * w).float().mean((1, 2)).min()):.3f}, "
            f"mean {float(1 - (wn > 0.5 * w).float().mean()):.3f}")
    params = [{"params": net.parameters(), "lr": d["lr_unet"]},
              {"params": list(mlp_z.parameters()) + list(mlp_s.parameters()) + list(mlp_c.parameters()), "lr": d["lr_mlp"]},
              {"params": list(mlp_sv.parameters()) + list(mlp_cv.parameters()), "lr": d["lr_mlp"]}]
    use_free = bool(d.get("pupil_free", False))
    Wf = torch.zeros(data["M"], data["M"], requires_grad=True)          # ifftshifted layout
    yy_, xx_ = np.mgrid[-(data["M"] // 2):data["M"] - data["M"] // 2, -(data["M"] // 2):data["M"] - data["M"] // 2] * data["dk"]
    fmask = torch.tensor(np.fft.ifftshift(np.hypot(yy_, xx_) <= 1.06 * cfg["kc"]).astype(np.float32))
    if use_free:
        params.append({"params": [Wf], "lr": d.get("lr_pupil_free", 1e-3)})
    opt = torch.optim.Adam(params)
    dx = cfg["pixel_um"]
    shift_mode = d.get("shift_mode", "illumination")
    zero_sh = torch.zeros(n, 2)

    def outputs(noise=0.0):
        xi = x_in + noise * torch.randn_like(x_in) if noise > 0 else x_in
        raw = net(xi)[0]
        a = fourier_resample(d["absorption_scale"] * raw[0], data["No"])
        phi = fourier_resample(d["phase_scale"] * raw[1], data["No"])
        zc = mlp_z()
        ut = smax * torch.tanh(mlp_s().view(n, 2) / smax)                     # bounded |u| < shift_max
        ut = (ut - (ut * tmask[:, None]).sum(0) / tmask.sum()) * tmask[:, None]  # zero mean over training images
        # (a common shift is degenerate with an object translation / phase ramp)
        u = ut + smax * torch.tanh(mlp_sv().view(n, 2) / smax) * vmask[:, None]
        if shift_mode == "image":
            sh, dki = u * dx, None                                     # lateral image shift, um
        else:
            sh, dki = zero_sh, u * data["dk"]                          # illumination wave-vector correction, um^-1
        c = torch.exp(mlp_c() * tmask + mlp_cv() * vmask)
        return a, phi, zc, sh, c, dki, (Wf * fmask if use_free else None)

    lam_img = torch.tensor(data["img_w"]) if data.get("img_w") is not None else None
    # dark_gain "profile": the unknown illumination level of a dark-field position is profiled out, i.e. its intensity
    # factor is the least-squares optimum c_n = sum(w I_pred I_meas) / sum(w I_pred^2) at every step
    dark_t = torch.tensor(np.asarray(data["dark"])) if data.get("dark") is not None else None
    profile = d.get("dark_gain") == "profile" and dark_t is not None and bool(dark_t.any())

    def data_term(Ip, Im, c, ii):
        ww = wn[ii]
        if profile:
            dki_ = dark_t[ii]
            if bool(dki_.any()):
                if d.get("dark_offset") == "profile":   # least-squares gain and additive offset per dark image
                    s0 = ww.sum((1, 2)); s1 = (ww * Ip).sum((1, 2)); s2 = (ww * Ip * Ip).sum((1, 2))
                    t0_ = (ww * Im).sum((1, 2)); t1_ = (ww * Ip * Im).sum((1, 2)); det = s0 * s2 - s1 * s1 + 1e-30
                    cp = torch.clamp((s0 * t1_ - s1 * t0_) / det, min=0.0)
                    bp = (t0_ - cp * s1) / (s0 + 1e-30)
                    Ip = torch.where(dki_[:, None, None], Ip + (bp / torch.clamp(cp, min=1e-12))[:, None, None], Ip)
                else:
                    cp = torch.clamp((ww * Ip * Im).sum((1, 2)) / ((ww * Ip * Ip).sum((1, 2)) + 1e-20), min=0.0)
                c = torch.where(dki_, cp, c)
        if d["loss"] == "amplitude":
            r = torch.sqrt(torch.clamp(c[:, None, None] * Ip, min=1e-8)) - torch.sqrt(torch.clamp(Im, min=0))
        else:                                                   # slide: (c_n I_pred - I_meas)^2
            r = c[:, None, None] * Ip - Im
        if lam_img is not None:
            return (ww * r ** 2).sum((1, 2)) * lam_img[ii]
        return (ww * r ** 2).sum((1, 2))

    # band-limited spectral misfit (same definition as evaluate.band_misfit) -> signal-sensitive validation
    q = np.fft.fftfreq(M, d=1.0 / (M * data["dk"])); QR = np.hypot(*np.meshgrid(q, q, indexing="ij"))
    bsel = torch.tensor((QR >= cfg["export_band"][0]) & (QR <= cfg["export_band"][1]))
    if data.get("wmask") is not None:        # contrast relative to the (masked) mean of each image
        wm_ = torch.tensor(data["wmask"])
        wmean = lambda I: (I * wm_).sum((1, 2), keepdim=True) / wm_.sum((1, 2), keepdim=True)
        Dm = torch.fft.fft2((I_meas / wmean(I_meas) - 1.0) * wn)[:, bsel]
    else:
        wmean = lambda I: I.mean((1, 2), keepdim=True)
        Dm = torch.fft.fft2((I_meas - 1.0) * wn)[:, bsel]

    gain_metric = phys.pc and dark_t is not None and bool(dark_t.any())
    if gain_metric:      # partial coherence: window-weighted means; dark images with a profiled contrast factor
        wwm = (wm_ if data.get("wmask") is not None else torch.ones_like(I_meas)) * w[None]
        wmean_w = lambda I: (I * wwm).sum((1, 2), keepdim=True) / wwm.sum((1, 2), keepdim=True)
        Dm = torch.fft.fft2((I_meas / wmean_w(I_meas) - 1.0) * wn)[:, bsel]

    def band_per_image(I):
        if gain_metric:
            mu = wmean_w(I)
            X = torch.where(dark_t[:, None, None], I - mu, I / mu - 1.0)
            F = torch.fft.fft2(X * wn)[:, bsel]
            cg = torch.clamp((F.conj() * Dm).real.sum(1) / (F.abs().pow(2).sum(1) + 1e-30), min=0.0)
            F = F * torch.where(dark_t, cg, torch.ones_like(cg))[:, None]
            return (F - Dm).abs().pow(2).sum(1) / Dm.abs().pow(2).sum(1)
        F = torch.fft.fft2((I / wmean(I) - 1.0) * wn)[:, bsel]
        return (F - Dm).abs().pow(2).sum(1) / Dm.abs().pow(2).sum(1)
    sel_key = d.get("select_by", "val_band")
    hist = dict(it=[], train=[], val=[], train_band=[], val_band=[], t=[])

    def u_rms(sh, dki, ii):
        v = sh / dx if dki is None or shift_mode == "image" else dki / data["dk"]
        return v[ii].norm(dim=1).pow(2).mean().sqrt()
    best = dict(score=np.inf, val=np.inf, it=-1)
    snaps, best_min = [], np.inf
    sel_tol = float(d.get("select_tol", 0.0))
    t0 = time.time()
    stop_file = os.path.join(os.path.dirname(checkpoint), "STOP") if checkpoint else None
    n_done = d["n_iter"]
    for it in range(d["n_iter"]):
        if stop_file and it % d["eval_every"] == 0 and os.path.exists(stop_file):
            log(f"  STOP file found -> ending training at iteration {it}"); n_done = it; break
        opt.zero_grad()
        a, phi, zc, sh, c, dki, wf = outputs(d["input_noise"])
        if phys.pc and phys.cache_on and it % pc_every == 0:
            phys.refresh_direct(zc.detach(), None if dki is None else dki.detach(), None if wf is None else wf.detach())
        Ip = phys(a, phi, zc, sh, idx=itr, dk_illum=dki, W_free=wf)
        L = data_term(Ip, I_meas[itr], c[itr], itr).mean()
        if len(iva):   # nuisance parameters of held-out images; object, pupil and training parameters frozen
            Iv = phys(a.detach(), phi.detach(), zc.detach(), sh, idx=iva, dk_illum=dki, W_free=None if wf is None else wf.detach())
            L = L + data_term(Iv, I_meas[iva], c[iva], iva).mean()
        if d["l2_weight"] > 0:
            L = L + d["l2_weight"] * sum((p ** 2).sum() for p in net.parameters())
        if use_free and d.get("pupil_free_l2", 0) > 0:
            L = L + d["pupil_free_l2"] * (wf ** 2).sum()
        L.backward()
        opt.step()
        if it % d["eval_every"] == 0 or it == d["n_iter"] - 1:
            with torch.no_grad():
                net.eval() if d["eval_mode_bn"] else None
                a, phi, zc, sh, c, dki, wf = outputs()
                if phys.pc: phys.refresh_direct(zc, dki, wf)
                Ia = phys(a, phi, zc, sh, dk_illum=dki, W_free=wf)
                tr = float(data_term(Ia[itr], I_meas[itr], c[itr], itr).mean())
                if len(iva):
                    va = float(data_term(Ia[iva], I_meas[iva], c[iva], iva).mean())
                else:
                    va = tr
                bp = band_per_image(Ia)
                trb = float(bp[itr].mean()); vab = float(bp[iva].mean()) if len(iva) else trb
                net.train()
            hist["it"].append(it); hist["train"].append(tr); hist["val"].append(va); hist["t"].append(time.time() - t0)
            hist["train_band"].append(trb); hist["val_band"].append(vab)
            score = {"val_band": vab, "val": va, "train_band": trb, "train": tr}[sel_key]
            snap = dict(score=score, val=va, val_band=vab, it=it, a=a.numpy().copy(), phi=phi.numpy().copy(), zc=zc.numpy().copy(),
                        shifts_um=sh.numpy().copy(), c=c.numpy().copy(),
                        dk_illum=(dki.numpy().copy() if dki is not None else np.zeros((n, 2))),
                        W_free=(np.fft.fftshift(wf.numpy()) if wf is not None else np.zeros((M, M))))
            snaps.append(snap)
            if score < best_min:
                best_min = score
            # early stopping rule: earliest evaluated iteration whose score is within select_tol (relative) of the minimum
            best = next(sn for sn in snaps if sn["score"] <= best_min * (1 + sel_tol))
            if it % (d["eval_every"] * 5) == 0 or it == d["n_iter"] - 1:
                shr = float(u_rms(sh, dki, itr))
                log(f"  it {it:5d}  loss train {tr:.2f} val {va:.2f} | band misfit train {trb:.4f} val {vab:.4f}"
                    f" | rms {'shift' if shift_mode == 'image' else 'dk'} {shr:.2f} px, defocus {float(zc[0]):+.3f} rad  best@{best['it']}  ({time.time()-t0:.0f} s)")
                if checkpoint:                                   # resumable/inspectable partial result
                    save_state(checkpoint, dict(best=best, hist={k: np.array(v) for k, v in hist.items()},
                                                kappa_solver=kappa_solver, M=data["M"], No=data["No"], dk=data["dk"],
                                                dxo=phys.dxo, n_done=it + 1))
    with torch.no_grad():
        a, phi, zc, sh, c, dki, wf = outputs()
    final = dict(a=a.numpy(), phi=phi.numpy(), zc=zc.numpy(), shifts_um=sh.numpy(), c=c.numpy(), it=n_done - 1,
                 dk_illum=(dki.numpy() if dki is not None else np.zeros((n, 2))),
                 W_free=(np.fft.fftshift(wf.detach().numpy()) if wf is not None else np.zeros((M, M))))
    res = dict(best=best, final=final, hist={k: np.array(v) for k, v in hist.items()}, kappa_solver=kappa_solver,
               M=data["M"], No=data["No"], dk=data["dk"], dxo=phys.dxo, n_done=n_done)
    if checkpoint:
        save_state(checkpoint, res)
    return dict(best=best, final=final, hist={k: np.array(v) for k, v in hist.items()}, keep=keep, train=train, val=val,
                kappa_solver=kappa_solver, M=data["M"], No=data["No"], dk=data["dk"], dxo=phys.dxo)


def save_state(path, res):
    out = {}
    for grp in ("best", "final"):
        for k, v in res.get(grp, {}).items():
            out[f"{grp}_{k}"] = np.asarray(v)
    for k, v in res["hist"].items():
        out[f"hist_{k}"] = v
    for k in ("kappa_solver", "M", "No", "dk", "dxo", "n_done"):
        out[k] = np.asarray(res[k])
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **out); os.replace(tmp, path)


def load_state(path):
    z = np.load(path); res = dict(best={}, final={}, hist={})
    for k in z.files:
        g, _, name = k.partition("_")
        if g in ("best", "final", "hist"):
            res[g][name] = z[k] if z[k].ndim else z[k].item()
        else:
            res[k] = z[k].item()
    return res


def vignetting_weights(phys, zc0, w, sigma_px=3.0):
    """Per-image loss weights: window x smoothed mask of pixels where the image of Q alone is above the floor."""
    from scipy.ndimage import gaussian_filter
    with torch.no_grad():
        if getattr(phys, "pc", False):
            Id = phys.direct(zc0).numpy()
        else:
            P = phys.pupil(zc0, phys.kres)[0]
            Id = phys._image(phys.Qh, P, np.arange(phys.n), None).numpy()
    m = (Id >= phys.dir_floor * Id[~phys.dark].max()).astype(np.float32) if (~phys.dark).any() else np.ones_like(Id)
    m[phys.dark] = 1.0                   # dark-field images: no direct-beam mask (validity from dfmask in dark mode)
    m = np.stack([np.clip(gaussian_filter(mi, sigma_px), 0, 1) * mi for mi in m])
    return torch.tensor(m) * w[None]
