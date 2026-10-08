"""Does the BLIS-FPM solution rely on object content outside the exported band (0.3-3.5 um^-1)?
Re-evaluate the full-field fptrecon loss with the fitted pupil W and (a) the raw object, (b) the object band-passed
to 0.3-3.5 um^-1 (what is exported and compared), (c) band-passed to 0-3.5 (keep low frequencies).
usage: python blis_bandcheck.py <config.json> <nonlinear.npz> [threads]"""
import json, os, sys
import numpy as np, torch
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon.config import load_config
from fptrecon import nonlinear as nlr
from fptrecon.optics import kgrid
from scipy.signal.windows import tukey
cfg = load_config(sys.argv[1]); z = dict(np.load(sys.argv[2])); torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 2)
nl = cfg["nonlinear"]; kc = cfg["kc"]
work = sys.argv[4] if len(sys.argv) > 4 else os.path.dirname(os.path.abspath(sys.argv[2]))
R = np.load(os.path.join(work, "preprocessed.npy"), mmap_mode="r"); cal = json.load(open(os.path.join(work, "calibration.json")))
kn = np.array(cal["k_solver"]); keep = z["keep"]; M, No, dk, dxo = int(z["M"]), int(z["No"]), float(z["dk"]), float(z["dxo"])
I = nlr.fcrop_intensity(np.clip(np.asarray(R)[keep], *cfg["clip"]), M); I = I / I.mean((1, 2), keepdims=True)
y = (np.arange(No) - No / 2) * dxo; Y, X = np.meshgrid(y, y, indexing="ij"); kap = float(z["kappa"])
Q = np.exp(1j * np.pi * kap * (X ** 2 + Y ** 2)) if kap != 0 else None
win = np.outer(tukey(M, nl["tukey_alpha"]), tukey(M, nl["tukey_alpha"]))
yy_, xx_ = kgrid(M, dk)
Pamp = z["Pamp"] if "Pamp" in z else (np.hypot(yy_, xx_) <= nl["pupil_init_scale"] * kc).astype(float)   # older runs: fixed NA disc
P0 = Pamp * np.exp(1j * z["W"])
def bp(x, lo, hi):
    q = np.hypot(*np.meshgrid(np.fft.fftfreq(No, dxo), np.fft.fftfreq(No, dxo), indexing="ij"))
    return np.real(np.fft.ifft2(np.fft.fft2(x) * ((q >= lo) & (q <= hi)))).astype(np.float32)
out = {}
for name, (a, p) in {"raw object": (z["a"], z["phi"]), "object 0.3-3.5": (bp(z["a"], .3, 3.5), bp(z["phi"], .3, 3.5)),
                     "object 0-3.5": (bp(z["a"], 0, 3.5), bp(z["phi"], 0, 3.5)), "object 0.3-inf": (bp(z["a"], .3, 99), bp(z["phi"], .3, 99))}.items():
    fb = nlr.FPMBand(I, -kn[keep] / dk, No, M, kc / dk, tuple(z["band"]), Q=Q, P0=P0, window=win, support_scale=nl["support_scale"],
                     a0=a.astype(np.float32), phi0=p.astype(np.float32), direct_norm=True, dir_floor=nl["dir_floor"])
    with torch.no_grad():
        fb.W.copy_(torch.tensor(np.fft.ifftshift(z["W"]), dtype=torch.float32))
        L = float(fb.loss())
    q = np.hypot(*np.meshgrid(np.fft.fftfreq(No, dxo), np.fft.fftfreq(No, dxo), indexing="ij"))
    Fp = np.abs(np.fft.fft2(p)) ** 2
    out[name] = dict(misfit=L)
    print(f"{name:16s} misfit {L:.4f}", flush=True)
Fp = np.abs(np.fft.fft2(z["phi"] - z["phi"].mean())) ** 2; Fa = np.abs(np.fft.fft2(z["a"] - z["a"].mean())) ** 2
q = np.hypot(*np.meshgrid(np.fft.fftfreq(No, dxo), np.fft.fftfreq(No, dxo), indexing="ij"))
for lab, F in [("phase", Fp), ("absorption", Fa)]:
    tot = F.sum(); print(f"{lab}: power fraction <0.3: {F[q < .3].sum() / tot:.3f}, 0.3-3.5: {F[(q >= .3) & (q <= 3.5)].sum() / tot:.3f}, >3.5: {F[q > 3.5].sum() / tot:.3f}")
    out[f"{lab}_power_fraction"] = dict(below=float(F[q < .3].sum() / tot), band=float(F[(q >= .3) & (q <= 3.5)].sum() / tot), above=float(F[q > 3.5].sum() / tot))
json.dump(out, open(os.path.splitext(sys.argv[2])[0] + "_bandcheck.json", "w"), indent=1)
