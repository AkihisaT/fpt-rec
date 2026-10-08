# -*- coding: utf-8 -*-
"""Synthetic check of the mixed-state BLIS-FPM model: data simulated with partial coherence (S = core 0.02 kc + Cauchy halo,
eta = 0.3) for the 036 geometry of rings 1 + 3 (28 images, FOV c = 1, calibrated defocus), 256-px field (8.2 um), 1 %
Gaussian noise.  The object (phase and absorption from the BLIS-FPM rings 1+3 solution, central field) is reconstructed from
the true object and from the true object at half contrast, with the pupil fixed to the truth,, (a) with the coherent model and (b) with the correct mixed-state model.
Reports the phase error (0.3-3.5 um^-1 band) and the per-ring data misfit.  usage: python d036/pc/synthetic_test.py"""
import json, sys, time, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import pc_common as pcm
sys.path.insert(0, pcm.W + "fpt_pipeline")
from fptrecon import nonlinear as nlr, coherence as coh
from fptrecon.optics import analytic_W, fresample, kgrid, fast_even
from scipy.signal.windows import tukey
torch.set_num_threads(2); rng = np.random.default_rng(0)
kc, dx = pcm.kc, pcm.dx; Ncam = 256; dk = 1 / (Ncam * dx)
idx = np.array([i for i in range(44) if pcm.ring[i] != 2]); kn = pcm.kn_all[idx]; dark = np.hypot(*kn.T) > 1.1 * kc
M = fast_even((2 * 1.05 * kc + 3.5) / dk); smax = np.abs(kn / dk).max()
No = fast_even(M + 2 * np.ceil(smax) + 8 + 2 * (np.ceil(1.25 * kc / dk) + 2)); dxo = Ncam * dx / No
sol = np.load(pcm.W + "d036/blis_df13/s-1.npz"); N0 = int(sol["No"]); c0 = N0 // 2; hw = int(round(N0 * 256 / 1024 / 2))
a_t = fresample(np.asarray(sol["a"])[c0 - hw:c0 + hw, c0 - hw:c0 + hw], No).astype(np.float32)
p_t = fresample(np.asarray(sol["phi"])[c0 - hw:c0 + hw, c0 - hw:c0 + hw], No).astype(np.float32)
ks = float(sol["kappa"]); y = (np.arange(No) - No / 2) * dxo; Y, X = np.meshgrid(y, y, indexing="ij"); Q = np.exp(1j * np.pi * ks * (X ** 2 + Y ** 2))
W0 = analytic_W(M, dk, pcm.cfg["lam_um"], pcm.cal["defocus_um"], pcm.cal["astig_um"]); yy, xx = kgrid(M, dk)
P0 = (np.hypot(yy, xx) <= 1.02 * kc) * np.exp(1j * W0)
win = np.outer(tukey(M, 0.2), tukey(M, 0.2)); band = (0.3 / dk, 3.5 / dk)
e_ = coh.FINE_EDGES; dref, wref = coh.modes_from_radial(e_ * kc, coh.cauchy_core_fractions(e_, 0.3, 0.02, 0.1, 2.0), coh.FINE_NAZ)
dm, wm_ = coh.adaptive_modes(kn, dref, wref, kc, **coh.PC_SETS["F"])
def band_fb(I, modes, mw, a0, p0):
    fb = nlr.FPMBand(I, -kn / dk, No, M, kc / dk, band, Q=Q, P0=P0, window=win, support_scale=1.05, a0=a0, phi0=p0,
                     direct_norm=True, dir_floor=0.1, dark=dark, wmask=np.ones((len(idx), M, M), np.float32), img_weights=np.ones(len(idx)),
                     modes=modes, mode_w=mw, dark_gain="profile")
    with torch.no_grad(): fb.W.copy_(torch.tensor(np.fft.ifftshift(W0), dtype=torch.float32))
    return fb
# simulate: normalised mixed-state model + 1 (mean 1), plus 1 % noise of each image's contrast rms
gen = band_fb(np.ones((len(idx), M, M), np.float32), -dm / dk, wm_, a_t, p_t)
with torch.no_grad(): Ic = (gen.normalised() + 1.0).numpy()
Ic = Ic + 0.01 * Ic.std((1, 2), keepdims=True) * rng.standard_normal(Ic.shape)
zero = np.zeros((No, No), np.float32); qb = np.fft.fftfreq(No, dxo); QB = np.hypot(*np.meshgrid(qb, qb, indexing="ij")); bm = (QB >= 0.3) & (QB <= 3.5)
def bp(x): return np.real(np.fft.ifft2(np.fft.fft2(x) * bm))
cen = np.s_[No // 8:-No // 8, No // 8:-No // 8]
def err(p): d = bp(p) - bp(p_t); return float(np.sqrt((d[cen] ** 2).mean()) / np.sqrt((bp(p_t)[cen] ** 2).mean()))
out = {}
for start, (a0, p0) in {"true object": (a_t, p_t), "true object at half contrast": (0.5 * a_t, 0.5 * p_t)}.items():
  for name, (modes, mw) in {"coherent model": (np.zeros((1, 2)), np.ones(1)), "mixed-state model (true S)": (-dm / dk, wm_)}.items():
    fb = band_fb(Ic.astype(np.float32), modes, mw, a0.astype(np.float32), p0.astype(np.float32)); t0 = time.time()
    per0 = fb.per_image_loss()
    fb.run(100, ("obj",), log=lambda m: None, every=0)
    per = fb.per_image_loss(); pe = err(fb.phi.detach().numpy())
    key = f"{start} | {name}"
    out[key] = dict(phase_error_rel=pe, misfit_ring1=float(per[pcm.ring[idx] == 1].mean()), misfit_ring3=float(per[pcm.ring[idx] == 3].mean()),
                    start_misfit_ring1=float(per0[pcm.ring[idx] == 1].mean()), start_misfit_ring3=float(per0[pcm.ring[idx] == 3].mean()), seconds=time.time() - t0)
    print(f"{key:60s} phase error {pe:.3f} | misfit ring1 {per0[pcm.ring[idx] == 1].mean():.4f}->{out[key]['misfit_ring1']:.4f} ring3 {per0[pcm.ring[idx] == 3].mean():.4f}->{out[key]['misfit_ring3']:.4f} | {time.time() - t0:.0f}s", flush=True)
json.dump(dict(field_px=Ncam, M=int(M), No=int(No), eta=0.3, noise=0.01, results=out), open(pcm.W + "d036/pc/synthetic_test.json", "w"), indent=1)
