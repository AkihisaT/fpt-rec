"""Data vs BLIS-FPM model (FOV) for example images on the full-field M grid (band-passed contrast 0.3-3.5 um^-1):
ring-2 image fitted in the 44-image run, ring-3 image fitted (28-image run) and ring-3 image held out (hold-out run).
Output d036/df_examples.npz"""
import json, os, sys
import numpy as np, torch
from scipy.ndimage import zoom
from scipy.signal.windows import tukey
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon.config import load_config
from fptrecon import nonlinear as nlr
from fptrecon.optics import fcrop_intensity
cfg = load_config(W + "fpt_pipeline/config_036bf.json"); nl = cfg["nonlinear"]; kc = cfg["kc"]
WD = W + "d036/work_df/"
R = np.load(WD + "preprocessed.npy", mmap_mode="r"); cal = json.load(open(WD + "calibration.json")); mk = np.load(WD + "dfmask.npy", mmap_mode="r")
kn_all = np.array(cal["k_solver"])
def model(npz, idx):
    z = np.load(npz); M, No, dk = int(z["M"]), int(z["No"]), float(z["dk"]); ks = float(z["kappa"]); dxo = float(z["dxo"])
    kh = kn_all[idx]; dh = np.hypot(kh[:, 0], kh[:, 1]) > 1.1 * kc
    Ih = fcrop_intensity(np.asarray(R[idx], np.float32), M)
    wh = np.stack([np.clip(zoom(np.asarray(mk[i], np.float32), M / 1024, order=1), 0, 1) for i in idx]).astype(np.float32); wh[~dh] = 1
    y = (np.arange(No) - No / 2) * dxo; Y, X = np.meshgrid(y, y, indexing="ij")
    Q = np.exp(1j * np.pi * ks * (X ** 2 + Y ** 2)) if ks != 0 else None
    win = np.outer(tukey(M, nl["tukey_alpha"]), tukey(M, nl["tukey_alpha"]))
    band = (nl["q_band"][0] / dk, nl["q_band"][1] / dk)
    P0 = np.fft.fftshift(np.fft.ifftshift(z["Pamp"]))
    fb = nlr.FPMBand(Ih, -kh / dk, No, M, kc / dk, band, Q=Q, P0=z["Pamp"].astype(complex), window=win, support_scale=nl["support_scale"],
                     a0=z["a"], phi0=z["phi"], direct_norm=True, dir_floor=nl["dir_floor"], dark=dh, wmask=wh, img_weights=np.ones(len(idx)))
    with torch.no_grad():
        fb.W.copy_(torch.tensor(np.fft.ifftshift(z["W"]), dtype=torch.float32))
        m = fb.normalised().numpy()
    q = np.fft.fftfreq(M, 1 / (M * dk)); QR = np.hypot(*np.meshgrid(q, q, indexing="ij")); b = (QR >= 0.3) & (QR <= 3.5)
    bp = lambda x: np.fft.ifft2(np.fft.fft2(x * win) * b).real
    d = np.stack([bp(Ih[i] / ((Ih[i] * wh[i]).sum() / wh[i].sum()) - 1) for i in range(len(idx))])
    mm = np.stack([bp(m[i]) for i in range(len(idx))])
    return d, mm, fb.per_image_loss(), wh
out = {}
d, m, per, wh = model(W + "d036/blis_df/s-1.npz", [13]); out.update(r2_data=d[0], r2_model=m[0], r2_misfit=per[0], r2_mask=wh[0])
d, m, per, _ = model(W + "d036/blis_df13/s-1.npz", [30]); out.update(r3fit_data=d[0], r3fit_model=m[0], r3fit_misfit=per[0])
d, m, per, _ = model(W + "d036/blis_df13/s-1_ho.npz", [34]); out.update(r3ho_data=d[0], r3ho_model=m[0], r3ho_misfit=per[0])
np.savez_compressed(W + "d036/df_examples.npz", **out)
print({k: float(v) for k, v in out.items() if np.ndim(v) == 0})
