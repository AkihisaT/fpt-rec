# -*- coding: utf-8 -*-
"""Shared helpers for the 036 partial-coherence scripts: data at the M grid, FPMBand construction from a saved BLIS-FPM
solution, object re-gridding for mode shifts."""
import json, os, sys
import numpy as np
import torch
from scipy.ndimage import zoom
from scipy.signal.windows import tukey
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon.config import load_config
from fptrecon import nonlinear as nlr
from fptrecon.optics import fcrop_intensity, fresample, kgrid
cfg = load_config(W + "fpt_pipeline/config_036bf.json"); nl = cfg["nonlinear"]
WD = W + "d036/work_df/"
cal = json.load(open(WD + "calibration.json")); ring = np.array(cal["ring"]); kn_all = np.array(cal["k_solver"])
kc, dx = cfg["kc"], cfg["pixel_um"]

def load_data(idx, M, half=""):
    R = np.load(WD + f"preprocessed{'_' + half if half else ''}.npy", mmap_mode="r"); mk = np.load(WD + "dfmask.npy", mmap_mode="r")
    Ncam = R.shape[-1]
    I = fcrop_intensity(np.asarray(R[idx], np.float32), M)
    dark = np.hypot(kn_all[idx, 0], kn_all[idx, 1]) > 1.1 * kc
    wm = np.stack([np.clip(zoom(np.asarray(mk[i], np.float32), M / Ncam, order=1), 0, 1) for i in idx]).astype(np.float32)
    wm[~dark] = 1.0
    return I, dark, wm, Ncam

def build(sol, idx, modes_k=None, mode_w=None, No=None, use_mask=True, half="", dark_gain=None):
    """FPMBand for images idx with the object and pupil of a saved solution (npz of blis_df.py / nonlinear_*.npz).
    modes_k: (J, 2) mode offsets in um^-1 (solver frame, added to k_n); No: object grid (default: enlarged for the modes)."""
    M, dk = int(sol["M"]), float(sol["dk"]); No0 = int(sol["No"]); dxo0 = float(sol["dxo"])
    I, dark, wm, Ncam = load_data(idx, M, half)
    if not use_mask: wm = np.where(dark[:, None, None], 1.0, wm).astype(np.float32)
    kn = kn_all[idx]
    if No is None:
        extra = 0 if modes_k is None else int(np.ceil(np.abs(modes_k).max() / dk)) + 2
        No = No0 + 2 * extra
    dxo = Ncam * dx / No
    a = fresample(np.asarray(sol["a"], np.float64), No).astype(np.float32); phi = fresample(np.asarray(sol["phi"], np.float64), No).astype(np.float32)
    ks = float(sol["kappa"]); y = (np.arange(No) - No / 2) * dxo; Y, X = np.meshgrid(y, y, indexing="ij")
    Q = np.exp(1j * np.pi * ks * (X ** 2 + Y ** 2)) if ks != 0 else None
    yy, xx = kgrid(M, dk)
    P0 = (np.hypot(yy, xx) <= nl["pupil_init_scale"] * kc) * np.exp(1j * np.asarray(sol["W"]))
    win = np.outer(tukey(M, nl["tukey_alpha"]), tukey(M, nl["tukey_alpha"]))
    band = (nl["q_band"][0] / dk, nl["q_band"][1] / dk)
    md = None if modes_k is None else -np.asarray(modes_k) / dk
    fb = nlr.FPMBand(I, -kn / dk, No, M, kc / dk, band, Q=Q, P0=P0, window=win, support_scale=nl["support_scale"], a0=a, phi0=phi,
                     direct_norm=True, dir_floor=nl["dir_floor"], dark=dark, wmask=wm, img_weights=np.ones(len(idx)), modes=md, mode_w=mode_w, dark_gain=dark_gain)
    with torch.no_grad():
        fb.W.copy_(torch.tensor(np.fft.ifftshift(np.asarray(sol["W"])), dtype=torch.float32))
    return fb
