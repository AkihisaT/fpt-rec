"""32a registration step 3: model-based refinement. Cross-correlates each measured image (Fourier-cropped M grid) with
the BLIS-FPM model image of the same illumination (band 0.3-3.5 um^-1, Tukey window); the model contains the pupil
parallax, so the measured offset is a registration error. Applies the shifts to the full-resolution stacks.
usage: python register_to_model.py <blis.npz> <work_in> <work_out> [<df_work_in> <df_work_out>]"""
import json, os, shutil, sys
import numpy as np
from scipy import ndimage as ndi
from scipy.signal.windows import tukey
z = np.load(sys.argv[1]); win, wout = sys.argv[2].rstrip("/") + "/", sys.argv[3].rstrip("/") + "/"
Im, Imod, keep, M, dk = z["Imeas"] if "Imeas" in z.files else None, z["Imod"], z["keep"], int(z["M"]), float(z["dk"])
R = np.load(win + "preprocessed.npy", mmap_mode="r"); N = R.shape[-1]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/fpt_pipeline")
from fptrecon.optics import fcrop_intensity
q = np.fft.fftfreq(M, 1 / (M * dk)); QR = np.hypot(*np.meshgrid(q, q, indexing="ij")); bmask = (QR >= 0.3) & (QR <= 3.5)
w = np.outer(tukey(M, 0.3), tukey(M, 0.3))
def xs(A, B):
    FA = np.fft.fft2((A - A.mean()) * w) * bmask; FB = np.fft.fft2((B - B.mean()) * w) * bmask
    up = 4; Mu = M * up; C = np.zeros((Mu, Mu), complex)
    X = np.conj(FA) * FB; Xs = np.fft.fftshift(X); c0 = Mu // 2 - M // 2; C[c0:c0 + M, c0:c0 + M] = Xs
    c = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(C))))
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    par = lambda m, o, p: 0.5 * (m - p) / (m - 2 * o + p)
    return np.array([iy + par(c[iy - 1, ix], c[iy, ix], c[iy + 1, ix]) - Mu // 2, ix + par(c[iy, ix - 1], c[iy, ix], c[iy, ix + 1]) - Mu // 2]) / up
sh = np.zeros((R.shape[0], 2)); done = np.zeros(R.shape[0], bool)
pairs = list(zip(keep, Imod)) + (list(zip(z["heldout"], z["Imod_heldout"])) if "Imod_heldout" in z.files else [])
for i, Im_ in pairs:
    Imeas = fcrop_intensity(np.asarray(R[i], np.float32), M); Imeas /= Imeas.mean()
    sh[i] = xs(Im_, Imeas) * (N / M); done[i] = True          # camera px; measured = model shifted by sh
sh[done] -= sh[done].mean(0)                                        # common shift = object translation (degenerate)
print(f"model registration: {int(done.sum())} images, shift rms {np.sqrt((sh[done] ** 2).mean(0)).round(2)} px, max {np.abs(sh[done]).max(0).round(1)} px")
def apply(src, dst, shifts, mask=False):
    os.makedirs(dst, exist_ok=True)
    Rin = np.load(src + "preprocessed.npy", mmap_mode="r")
    out = np.stack([ndi.shift(np.asarray(Rin[i], np.float32), -shifts[i], order=3, mode="nearest") if np.any(shifts[i]) else np.asarray(Rin[i], np.float32)
                    for i in range(Rin.shape[0])])
    np.save(dst + "preprocessed.npy", out)
    for f in os.listdir(src):
        if f.endswith(".json") or f.endswith(".npz"): shutil.copy(src + f, dst + f)
    if os.path.exists(src + "dfmask.npy"):
        mk = np.load(src + "dfmask.npy", mmap_mode="r")
        np.save(dst + "dfmask.npy", np.stack([np.clip(ndi.shift(np.asarray(mk[i], np.float32), -shifts[i], order=1, mode="nearest"), 0, 1)
                                              for i in range(mk.shape[0])]).astype(np.float16))
apply(win, wout, sh)
json.dump(dict(source=sys.argv[1], shifts_px=sh.tolist(), rms_px=np.sqrt((sh[done] ** 2).mean(0)).tolist()), open(wout + "register_to_model.json", "w"), indent=1)
if len(sys.argv) > 5:                  # bright part of the dark-field work folder gets the same shifts
    dwin, dwout = sys.argv[4].rstrip("/") + "/", sys.argv[5].rstrip("/") + "/"
    nd = np.load(dwin + "preprocessed.npy", mmap_mode="r").shape[0]; shd = np.zeros((nd, 2)); shd[:len(sh)] = sh
    apply(dwin, dwout, shd)
print("done")
