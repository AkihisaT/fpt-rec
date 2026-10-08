#!/usr/bin/env python
"""BLIS-FPM driver for 32a (parallel-beam objective-FZP scan): the fptrecon solver (FPMBand, L-BFGS, pupil stage +
joint stage, band 0.3-3.5 um^-1) on a pipeline work folder, with optional held-out images and the dark-field
extensions of FPMBand (dark images: no direct-beam normalisation; validity masks from dfmask.npy; image weights
lambda_n = 1 / (noise variance of the normalised contrast), from noise.json, normalised to median 1 over the bright images).

usage: python blis_32a.py --work <dir> --dr 0.25 --c 1 [--exclude 0,9,...|val9] [--init linear.npz|zero] [--n-joint 150]
                          [--threads 4] --out <file.npz>
The solver frame is the physical frame (twin +1, see make_calib_32a.py); kappa_solver = c / (lambda p), p = 0.75 m.
Output npz has the layout of fptrecon nonlinear_*.npz (+ keys keep, heldout, heldout_per_image, dark, weights)."""
import argparse, json, os, sys, time
import numpy as np, torch
from scipy.ndimage import zoom
from scipy.signal.windows import tukey
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon.config import load_config
from fptrecon import nonlinear as nlr
from fptrecon.optics import analytic_W, fcrop_intensity, fresample, kgrid

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True); ap.add_argument("--dr", type=float, default=0.25); ap.add_argument("--c", type=float, default=0.0)
ap.add_argument("--exclude", default=""); ap.add_argument("--init", default="zero"); ap.add_argument("--n-pupil", type=int, default=20)
ap.add_argument("--n-joint", type=int, default=150); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--out", required=True)
ap.add_argument("--dark-thr", type=float, default=1.1)
a = ap.parse_args()
cfg = load_config(W + "fpt_pipeline/config_32a_bf.json", dict(fzp_outer_zone_um=a.dr)); nl = cfg["nonlinear"]
torch.set_num_threads(a.threads)
WD = a.work.rstrip("/") + "/"
R = np.load(WD + "preprocessed.npy", mmap_mode="r"); cal = json.load(open(WD + "calibration.json"))
assert abs(cal["kc"] - cfg["kc"]) < 1e-6, f"calibration kc {cal['kc']} != config kc {cfg['kc']}"
kn_all = np.array(cal["k_solver"]); lam_um, dx, kc = cfg["lam_um"], cfg["pixel_um"], cfg["kc"]
n_all = len(kn_all)
if a.exclude.startswith("val"):
    ev = int(a.exclude[3:]); excl = list(range(0, n_all, ev))           # same held-out set as DIP/EPRY (keep[::val_every])
else:
    excl = [int(v) for v in a.exclude.split(",")] if a.exclude else []
use = np.array([i for i in range(n_all) if i not in excl]); kn = kn_all[use]
darkall = np.hypot(kn_all[:, 0], kn_all[:, 1]) > a.dark_thr * kc; dark = darkall[use]
has_dark = bool(darkall.any())
Ncam = R.shape[-1]; dk = 1 / (Ncam * dx)
M, No = nlr.grid_sizes(cfg, kn_all, dk)
dxo = Ncam * dx / No
t0 = time.time(); log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
def load_I(ii):
    return fcrop_intensity(np.asarray(R[ii], np.float32), M)
def load_wm(ii, dk_):
    if not has_dark: return None
    mk = np.load(WD + "dfmask.npy", mmap_mode="r")
    wm = np.stack([np.clip(zoom(np.asarray(mk[i], np.float32), M / Ncam, order=1), 0, 1) for i in ii]).astype(np.float32)
    wm[~dk_] = 1.0; return wm
lamw_all = np.ones(n_all)
if has_dark:
    nz = json.load(open(WD + "noise.json"))
    s2c = np.array(nz["sigma2_1024"]) / np.array(nz["level_1024"]) ** 2
    lamw_all = 1.0 / s2c; lamw_all /= np.median(lamw_all[~darkall])
ks = a.c * cfg["kappa_nom"] * int(cal.get("twin", 1))
y = (np.arange(No) - No / 2) * dxo; Y, X = np.meshgrid(y, y, indexing="ij")
Q = np.exp(1j * np.pi * ks * (X ** 2 + Y ** 2)) if ks != 0 else None
W0 = analytic_W(M, dk, lam_um, cal["defocus_um"], cal["astig_um"])
yy, xx = kgrid(M, dk)
P0 = (np.hypot(yy, xx) <= nl["pupil_init_scale"] * kc) * np.exp(1j * W0)
win = np.outer(tukey(M, nl["tukey_alpha"]), tukey(M, nl["tukey_alpha"]))
band = (nl["q_band"][0] / dk, nl["q_band"][1] / dk)
if a.init == "zero":
    a0 = np.zeros((No, No), np.float32); p0 = np.zeros((No, No), np.float32)
else:
    li = np.load(a.init); assert abs(float(li["kappa_solver"]) - ks) < 1e-9, "linear start does not match the FOV model"
    a0 = fresample(li["a"], No).astype(np.float32); p0 = fresample(li["phi"], No).astype(np.float32)
ext = dict(dark=dark, wmask=load_wm(use, dark), img_weights=lamw_all[use]) if has_dark else {}
fb = nlr.FPMBand(load_I(use), -kn / dk, No, M, kc / dk, band, Q=Q, P0=P0, window=win, support_scale=nl["support_scale"],
                 a0=a0, phi0=p0, direct_norm=True, dir_floor=nl["dir_floor"], **ext)
with torch.no_grad():
    fb.W.copy_(torch.tensor(np.fft.ifftshift(W0), dtype=torch.float32))
L0 = float(fb.loss().detach())
log(f"c={a.c:g} (kappa_solver {ks:+.5f}), dr {a.dr} (kc {kc}): images {len(use)} ({int(dark.sum())} dark), held out {len(excl)}; "
    f"M={M}, No={No} (object pixel {dxo * 1e3:.1f} nm); init {a.init}; initial misfit {L0:.4f}")
if a.n_pupil: fb.run(a.n_pupil, ("pupil_phase",), log=log, every=10)
fb.run(a.n_joint, ("obj", "pupil_phase"), log=log, every=25)
with torch.no_grad():
    Imod = (fb.normalised() + 1.0).numpy()
per = fb.per_image_loss()
ho_per = None
if excl:
    ho = np.array(excl); kh = kn_all[ho]; dh = darkall[ho]
    ext_h = dict(dark=dh, wmask=load_wm(ho, dh), img_weights=np.ones(len(ho))) if has_dark else {}
    fh = nlr.FPMBand(load_I(ho), -kh / dk, No, M, kc / dk, band, Q=Q, P0=P0, window=win, support_scale=nl["support_scale"],
                     a0=fb.a.detach().numpy(), phi0=fb.phi.detach().numpy(), direct_norm=True, dir_floor=nl["dir_floor"], **ext_h)
    with torch.no_grad():
        fh.W.copy_(fb.W.detach())
    ho_per = fh.per_image_loss()
    with torch.no_grad():
        Imod_ho = (fh.normalised() + 1.0).numpy().astype(np.float32)
    log(f"held-out {ho.tolist()}: misfit mean {ho_per.mean():.4f} (bright {ho_per[~dh].mean() if (~dh).any() else float('nan'):.4f}, "
        f"dark {ho_per[dh].mean() if dh.any() else float('nan'):.4f})")
res = dict(a=fb.a.detach().numpy(), phi=fb.phi.detach().numpy(), W=np.fft.fftshift(fb.W.detach().numpy()), W0=W0,
           Pamp=np.fft.fftshift(fb.Pamp.numpy() * fb.sup.numpy()), hist=np.array(fb.hist), L0=L0, per_image=per,
           Imod=Imod.astype(np.float32), keep=use, kappa=ks, dxo=dxo, M=M, No=No, dk=dk, band=np.array(band), dark=dark,
           weights=lamw_all[use], heldout=np.array(excl, int), heldout_per_image=np.array([]) if ho_per is None else ho_per,
           Imod_heldout=np.zeros((0, M, M), np.float32) if ho_per is None else Imod_ho)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez_compressed(a.out, **res)
json.dump(dict(work=WD, dr=a.dr, kc=kc, c=a.c, kappa_solver=ks, init=a.init, n_joint=a.n_joint, misfit=float(fb.hist[-1]),
               per_image=[float(v) for v in per], mean_bright=float(per[~dark].mean()), mean_dark=float(per[dark].mean()) if dark.any() else None,
               heldout=excl, heldout_per_image=None if ho_per is None else [float(v) for v in ho_per],
               heldout_mean=None if ho_per is None else float(ho_per.mean()), seconds=time.time() - t0, M=M, No=No, dxo_um=dxo),
          open(a.out.replace(".npz", ".json"), "w"), indent=1)
log(f"done: fitted misfit {fb.hist[-1]:.4f} (bright mean {per[~dark].mean():.4f}{', dark mean %.4f' % per[dark].mean() if dark.any() else ''})")
