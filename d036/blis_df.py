"""BLIS-FPM for 036 Phase C: all 44 images (bright field ring 1 + edge/dark field rings 2-3).

Same solver (fptrecon.nonlinear.FPMBand, L-BFGS, pupil stage n_pupil + joint stage n_joint), same start (tile-WOTF
linear solution of the ring-1 images, fpt_output_036bf/work/linear_*.npz), same band (0.3-3.5 um^-1) as Phase B, with
the dark-field extensions of FPMBand:
  * dark images (|k_n| > 1.1 kc): no direct-beam normalisation; data from d036/work_df (common intensity scale);
  * per-image validity masks (work_df/dfmask.npy, resampled to the M grid) multiply the window; contrast about the
    masked mean;
  * image weights lambda_n = 1 / (noise power of the normalised contrast), from the half averages
    (work_df/noise.json: sigma2_1024 / level_1024^2), normalised to median 1 over the ring-1 images.
usage: [BLIS_DF_RINGS=1,3] python blis_df.py <s_solver> [threads] [excluded indices e.g. 0,25,34,43] [n_joint]
output: d036/blis_df/s<s>[_ho].npz (layout of fptrecon nonlinear_*.npz) and .json"""
import json, os, sys, time
import numpy as np
import torch
from scipy.ndimage import zoom
from scipy.signal.windows import tukey
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon.config import load_config
from fptrecon import nonlinear as nlr
from fptrecon.optics import analytic_W, fcrop_intensity, fresample, kgrid

if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/blis_df/STOP_CHAIN"):
    sys.exit(0)                           # (stops a queued chain of the superseded 44-image runs)
s = float(sys.argv[1]); threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4
excl = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] not in ("", "-") else []
cfg = load_config(W + "fpt_pipeline/config_036bf.json"); nl = cfg["nonlinear"]
PR = os.environ.get("BLIS_DF_PUPIL")                   # diagnostic: pupil radius / kc (amplitude disc), default 1.02
if PR:
    nl["pupil_init_scale"] = float(PR); nl["support_scale"] = max(nl["support_scale"], float(PR) + 0.03)
n_joint = int(sys.argv[4]) if len(sys.argv) > 4 else nl["n_joint"]
torch.set_num_threads(threads)
WD = W + "d036/work_df/"; BF = W + "fpt_pipeline/fpt_output_036bf/work/"
OUT = os.environ.get("BLIS_DF_OUT", W + "d036/blis_df/").rstrip("/") + "/"; os.makedirs(OUT, exist_ok=True)
HALF = os.environ.get("BLIS_DF_DATA", "")            # "", "A" or "B" (half averages, for FRC)
R = np.load(WD + f"preprocessed{'_' + HALF if HALF else ''}.npy", mmap_mode="r"); cal = json.load(open(WD + "calibration.json"))
noise = json.load(open(WD + "noise.json")); mk = np.load(WD + "dfmask.npy", mmap_mode="r")
kn_all = np.array(cal["k_solver"]); lam_um, dx, kc = cfg["lam_um"], cfg["pixel_um"], cfg["kc"]
rings_used = [int(v) for v in os.environ.get("BLIS_DF_RINGS", "1,2,3").split(",")]
# partial coherence (036): BLIS_DF_PC = halo fraction eta of S = (1-eta) Gauss(0.02 kc) + eta Cauchy(0.1 kc, beta 2)
# ("1mode" = coherent through the same mixed-state code path); BLIS_DF_PCMODES = adaptive mode set (F default, E fine);
# with BLIS_DF_PC set, dark images get a profiled contrast factor (BLIS_DF_GAIN=profile, default) and the model (I - I_direct)
PC = os.environ.get("BLIS_DF_PC"); PCM = os.environ.get("BLIS_DF_PCMODES", "F"); GAIN = os.environ.get("BLIS_DF_GAIN", "profile" if PC else None)
ks3 = float(os.environ.get("BLIS_DF_KSCALE3", "1"))       # diagnostic: scale |k| of the ring-3 images
if ks3 != 1.0:
    kn_all = kn_all.copy(); kn_all[np.array(cal["ring"]) == 3] *= ks3
use = np.array([i for i in range(len(kn_all)) if i not in excl and cal["ring"][i] in rings_used]); kn = kn_all[use]
dark = np.hypot(kn[:, 0], kn[:, 1]) > 1.1 * kc
Ncam = R.shape[-1]; dk = 1 / (Ncam * dx)
M, No = nlr.grid_sizes(cfg, kn, dk)
MODES = None
if PC:
    from fptrecon import coherence as coh
    PCSETS = dict(F=dict(core=0.08, sec_in=8, sec_out=2), A=dict(core=0.13, sec_in=8, sec_out=4),
                  E=dict(core=0.13, sec_in=16, sec_out=6, rad_edges=(0.13, 0.22, 0.35, 0.5, 0.7, 1.0, 1.3)))
    if PC == "1mode":
        MODES = (np.zeros((1, 2)), np.ones(1))
    else:
        e_ = np.array([0, .03, .07, .12, .2, .3, .45, .65, .9, 1.2]); na_ = [1, 6, 8, 12, 16, 20, 24, 28, 32]
        dref, wref = coh.modes_from_radial(e_ * kc, coh.cauchy_core_fractions(e_, float(PC), 0.02, 0.1, 2.0), na_)
        MODES = (dref, wref)
    No = No + 2 * (int(np.ceil(1.25 * kc / dk)) + 2)
    No += No % 2
def modes_for(kk):
    if MODES is None: return None, None
    if PC == "1mode": return -MODES[0] / dk, MODES[1]
    d_, w_ = coh.adaptive_modes(kk, MODES[0], MODES[1], kc, **PCSETS[PCM])
    return -d_ / dk, w_
dxo = Ncam * dx / No
t0 = time.time(); log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
I = fcrop_intensity(np.asarray(R[use], np.float32), M)
wm = np.stack([np.clip(zoom(np.asarray(mk[i], np.float32), M / Ncam, order=1), 0, 1) for i in use]).astype(np.float32)
wm[~dark] = 1.0
sig2c = np.array(noise["sigma2_1024"]) / np.array(noise["level_1024"]) ** 2
lamw = 1.0 / sig2c; ring = np.array(cal["ring"])
lamw = lamw / np.median(lamw[ring == 1]); lamw = lamw[use]
ks = s * cfg["kappa_nom"]
y = (np.arange(No) - No / 2) * dxo; Y, X = np.meshgrid(y, y, indexing="ij")
Q = np.exp(1j * np.pi * ks * (X ** 2 + Y ** 2)) if ks != 0 else None
W0 = analytic_W(M, dk, lam_um, cal["defocus_um"], cal["astig_um"])
yy, xx = kgrid(M, dk)
P0 = (np.hypot(yy, xx) <= nl["pupil_init_scale"] * kc) * np.exp(1j * W0)
win = np.outer(tukey(M, nl["tukey_alpha"]), tukey(M, nl["tukey_alpha"]))
band = (nl["q_band"][0] / dk, nl["q_band"][1] / dk)
if HALF or os.environ.get("BLIS_DF_LININIT") == "self":
    # independent start for the half data sets: tile-WOTF linear solution of the ring-1 images of the SAME data
    from fptrecon import wotf
    b1 = np.where(np.array(cal["ring"]) == 1)[0]
    a_l, p_l, _ = wotf.tile_linear_recon(np.asarray(R[b1]), kn_all[b1], cfg, ks, cal["defocus_um"], cal["astig_um"], log=lambda m: None)
    lin = dict(a=a_l, phi=p_l, kappa_solver=ks)
else:
    lin = np.load(BF + ("linear_cp0_00.npz" if s == 0 else "linear_cp1_00.npz"))
assert abs(float(lin["kappa_solver"]) - ks) < 1e-9 or s not in (0, -1), "linear start does not match the FOV model"
fb = nlr.FPMBand(I, -kn / dk, No, M, kc / dk, band, Q=Q, P0=P0, window=win, support_scale=nl["support_scale"],
                 a0=fresample(lin["a"], No).astype(np.float32), phi0=fresample(lin["phi"], No).astype(np.float32),
                 direct_norm=True, dir_floor=nl["dir_floor"], dark=dark, wmask=wm, img_weights=lamw,
                 modes=modes_for(kn)[0], mode_w=modes_for(kn)[1], dark_gain=GAIN)
# (the mixed-state direct beam is recomputed at every evaluation: it depends strongly on the pupil phase under the FOV curvature)
with torch.no_grad():
    fb.W.copy_(torch.tensor(np.fft.ifftshift(W0), dtype=torch.float32))
L0 = float(fb.loss().detach())
log(f"s={s:+g}: images {len(use)} ({int(dark.sum())} dark), excluded {excl}; M={M}, No={No} (object pixel {dxo * 1e3:.1f} nm); "
    f"weights ring1/2/3 median {[round(float(np.median(lamw[ring[use] == r])), 5) for r in (1, 2, 3)]}; initial misfit {L0:.4f}")
fb.run(nl["n_pupil"], ("pupil_phase",), log=log, every=10)
fb.run(n_joint, ("obj", "pupil_phase"), log=log, every=25)
with torch.no_grad():
    Imod = (fb.normalised() + 1.0).numpy()
per = fb.per_image_loss()
ho_per = None
if excl:      # prediction of the held-out images with the fitted object and pupil (same metric as per_image)
    ho = np.array([i for i in excl if cal["ring"][i] in rings_used]); kh = kn_all[ho]
    dh = np.hypot(kh[:, 0], kh[:, 1]) > 1.1 * kc
    Ih = fcrop_intensity(np.asarray(R[ho], np.float32), M)
    wh = np.stack([np.clip(zoom(np.asarray(mk[i], np.float32), M / Ncam, order=1), 0, 1) for i in ho]).astype(np.float32); wh[~dh] = 1.0
    fh = nlr.FPMBand(Ih, -kh / dk, No, M, kc / dk, band, Q=Q, P0=P0, window=win, support_scale=nl["support_scale"],
                     a0=fb.a.detach().numpy(), phi0=fb.phi.detach().numpy(), direct_norm=True, dir_floor=nl["dir_floor"],
                     dark=dh, wmask=wh, img_weights=np.ones(len(ho)), modes=modes_for(kh)[0], mode_w=modes_for(kh)[1], dark_gain=GAIN)
    with torch.no_grad():
        fh.W.copy_(fb.W.detach())
    ho_per = fh.per_image_loss()
    log(f"held-out images {ho.tolist()}: misfit {np.round(ho_per, 3).tolist()}")
tag = f"s{s:+.0f}" + ("_ho" if excl else "") + ("" if n_joint == nl["n_joint"] else f"_j{n_joint}") + (f"_half{HALF}" if HALF else "") + (f"_k3x{ks3:g}" if ks3 != 1.0 else "") + (f"_P{float(PR):g}" if PR else "") + (f"_pc{PC}{PCM if PC != '1mode' else ''}" if PC else "")
res = dict(a=fb.a.detach().numpy(), phi=fb.phi.detach().numpy(), W=np.fft.fftshift(fb.W.detach().numpy()), W0=W0,
           Pamp=np.fft.fftshift(fb.Pamp.numpy() * fb.sup.numpy()), hist=np.array(fb.hist), L0=L0, per_image=per,
           Imod=Imod.astype(np.float32), Imeas=I.astype(np.float32), keep=use, kappa=ks, dxo=dxo, M=M, No=No, dk=dk,
           band=np.array(band), dark=dark, weights=lamw)
np.savez_compressed(OUT + tag + ".npz", **res)
rs = {r: float(np.mean(per[ring[use] == r])) for r in (1, 2, 3)}
json.dump(dict(s_solver=s, kappa_solver=ks, excluded=excl, misfit=float(fb.hist[-1]), per_image=[float(v) for v in per],
               mean_per_ring=rs, n_joint=n_joint, kscale_ring3=ks3, partial_coherence=PC, pc_modes=PCM if PC else None, dark_gain=GAIN,
               n_modes_mean=None if MODES is None else float(np.atleast_2d(modes_for(kn)[1] > 0).sum(1).mean()),
               dark_gains=None if GAIN is None else fb.dark_gains().tolist(),
               heldout_images=[int(i) for i in excl], heldout_per_image=None if ho_per is None else [float(v) for v in ho_per], seconds=time.time() - t0, M=M, No=No, dxo_um=dxo),
          open(OUT + tag + ".json", "w"), indent=1)
log(f"{tag}: weighted misfit {fb.hist[-1]:.4f}; mean per-image misfit ring1/2/3 {rs[1]:.4f}/{rs[2]:.4f}/{rs[3]:.4f}")
