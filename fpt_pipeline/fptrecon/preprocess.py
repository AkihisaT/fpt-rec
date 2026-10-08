"""Step 1: raw TIFF / ITEX .img -> flat-fielded, normalised, flattened intensity stack (one image per position)."""
import glob
import os

import numpy as np
import tifffile
from scipy import ndimage as ndi


def read_itex(path):
    """Hamamatsu ITEX .img: 64-byte header ('IM', comment length, width, height, x/y offset, type) + comment + data."""
    with open(path, "rb") as f:
        b = f.read()
    if b[:2] != b"IM":
        raise ValueError(f"{path}: not an ITEX image")
    clen, w, h, _, _, typ = (int(v) for v in np.frombuffer(b[2:14], "<u2"))
    dt = {0: np.uint8, 2: np.uint16, 3: np.uint32}.get(typ)
    if dt is None:
        raise ValueError(f"{path}: unsupported ITEX data type {typ}")
    return np.frombuffer(b, dt, count=w * h, offset=64 + clen).reshape(h, w)


def load_stack(folder):
    """All *.tif / *.tiff (tifffile) or, if none, all *.img (ITEX) files of a folder, sorted by name."""
    files = sorted(glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.tiff")))
    reader = tifffile.imread
    if not files:
        files, reader = sorted(glob.glob(os.path.join(folder, "*.img"))), read_itex
    if not files:
        raise FileNotFoundError(f"no TIFF or ITEX .img files in {folder}")
    return np.stack([reader(f).astype(np.float32) for f in files]), files


def load_positions(path):
    P = np.loadtxt(path, delimiter=",", ndmin=2)
    return P[:, 1:3].astype(float)                        # (x, y) in pulses


def group_repeats(stack, npos, nrep, order):
    if stack.shape[0] != npos * nrep:
        raise ValueError(f"{stack.shape[0]} frames != {npos} positions x {nrep} repeats")
    if order == "interleaved":
        return stack.reshape(npos, nrep, *stack.shape[1:])
    if order == "blocked":
        return stack.reshape(nrep, npos, *stack.shape[1:]).transpose(1, 0, 2, 3)
    raise ValueError(order)


def radial_power(img, win, rbin):
    F = np.fft.fftshift(np.fft.fft2((img - img.mean()) * win))
    cnt = np.bincount(rbin.ravel())
    return np.bincount(rbin.ravel(), (np.abs(F) ** 2).ravel()) / np.maximum(cnt, 1)


def preprocess(cfg, log=print):
    d = cfg["data_dir"]
    pos = load_positions(os.path.join(d, cfg["positions_csv"]))
    npos, nrep = len(pos), cfg["n_repeat"]
    S, fs = load_stack(os.path.join(d, cfg["sample_subdir"]))
    D, fd = load_stack(os.path.join(d, cfg["direct_subdir"]))
    log(f"loaded {len(fs)} sample + {len(fd)} direct frames {S.shape[1:]}, {npos} positions x {nrep} repeats")
    S -= cfg["dark_offset"]; D -= cfg["dark_offset"]
    S = group_repeats(S, npos, nrep, cfg["repeat_order"])
    D = group_repeats(D, npos, nrep, cfg["repeat_order"])
    Sm, Dm = S.mean(1), D.mean(1)
    R = Sm / np.maximum(Dm, 1.0)
    R /= R.mean((1, 2), keepdims=True)
    # zinger / hot-pixel removal
    nfix = 0
    for i in range(npos):
        med = ndi.median_filter(R[i], 3)
        m = np.abs(R[i] - med) > cfg["zinger_threshold"]
        R[i][m] = med[m]; nfix += int(m.sum())
    # low-frequency flattening (residual flat-field structure)
    Rf = np.empty_like(R)
    for i in range(npos):
        Rf[i] = R[i] / ndi.gaussian_filter(R[i], cfg["flatten_sigma_px"], mode="reflect")
    log(f"zinger pixels replaced: {nfix}; flattened with sigma={cfg['flatten_sigma_px']} px")
    # signal / noise power spectra (repeat-frame differences of sample and direct images)
    N = R.shape[-1]
    win = np.outer(np.hanning(N), np.hanning(N)).astype(np.float32)
    yy, xx = np.mgrid[-(N // 2):N - N // 2, -(N // 2):N - N // 2]
    rbin = np.hypot(yy, xx).astype(int)
    Psig, Pnoi = [], []
    if nrep >= 2:
        for i in range(npos):
            ra = S[i, 0] / Dm[i]; rb = S[i, 1] / Dm[i]
            ra /= ra.mean(); rb /= rb.mean()
            da, db = D[i, 0] / D[i, 0].mean(), D[i, 1] / D[i, 1].mean()
            Psig.append(radial_power(0.5 * (ra + rb), win, rbin))
            Pnoi.append(radial_power(0.5 * (ra - rb), win, rbin) + radial_power(0.5 * (da - db), win, rbin))
    else:
        # single exposure per position: noise from the white plateau of the flat-fielded image beyond the
        # coherent-imaging pass band (|q| > 2.1 kc, below 0.95 Nyquist), assumed flat in |q|
        kr = np.arange(N) / (N * cfg["pixel_um"]); kc = cfg["kc"]
        sel = (kr > 2.1 * kc) & (kr < 0.95 * 0.5 / cfg["pixel_um"])
        for i in range(npos):
            ps = radial_power(R[i], win, rbin); Psig.append(ps)
            Pnoi.append(np.full_like(ps, ps[:len(kr)][sel[:len(ps)]].mean()))
        log(f"n_repeat=1: noise power from the |q| > {2.1 * kc:.1f} um^-1 plateau of each image")
    meta = dict(positions=pos, direct_mean=Dm.mean((1, 2)), Psig=np.array(Psig), Pnoi=np.array(Pnoi),
                k_radial=np.arange(len(Psig[0])) / (N * cfg["pixel_um"]) if Psig else np.array([]))
    return Rf.astype(np.float32), meta
