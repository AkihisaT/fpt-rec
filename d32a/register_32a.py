"""32a registration step 2: residual per-image shifts from a pairwise cross-correlation network.

The FZP stage does not return exactly to the tabulated positions: neighbouring images (same ring +-1, nearest azimuths
in the adjacent ring) are cross-correlated (band-passed Gaussian 1.5-12 px, Hann window), the pairwise shifts are
solved for absolute shifts s_n (least squares, zero mean; closure residual ~0.25 px), and s_n is split into
  (i)  a curl-free polynomial field of k_n (gradient of a degree-2..4 polynomial: image shifts caused by the pupil phase,
       i.e. defocus / astigmatism / coma parallax; physical, kept for the pupil of the reconstruction),
  (ii) a rotation field e (k_col, -k_row) (antisymmetric; cannot come from any pupil phase -> stage/detector geometry),
  (iii) a common offset, and (iv) the random remainder.
The images are shifted back by (ii) + (iv). Bright field (rings 0-2) and dark field (rings 6-7) are solved separately
(the two sets cannot be linked by correlation); the dark set keeps its mean offset from the rounding correction.
usage: python register_32a.py   -> fpt_output_32a_bf_reg{,_dr200}/work, d32a/work_df_reg, d32a/register_32a.json"""
import json, os, shutil, sys
import numpy as np
from scipy import ndimage as ndi
from concurrent.futures import ThreadPoolExecutor
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
FP = W + "fpt_pipeline/"
win2 = np.outer(np.hanning(1000), np.hanning(1000))
hp2 = lambda im: (ndi.gaussian_filter(im, 1.5) - ndi.gaussian_filter(im, 12)) * win2

def xcorr_shift(A, B):
    """shift s such that B(x) ~ A(x - s)"""
    c = np.fft.fftshift(np.real(np.fft.ifft2(np.conj(np.fft.fft2(A)) * np.fft.fft2(B))))
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    par = lambda m, o, p: 0.5 * (m - p) / (m - 2 * o + p)
    return np.array([iy + par(c[iy - 1, ix], c[iy, ix], c[iy + 1, ix]) - c.shape[0] // 2,
                     ix + par(c[iy, ix - 1], c[iy, ix], c[iy, ix + 1]) - c.shape[1] // 2])

def network(H, groups, az):
    pairs = []
    for r, ii in enumerate(groups):
        ii = np.asarray(ii)
        pairs += list(zip(ii, np.roll(ii, -1)))
        if r + 1 < len(groups):
            jj = np.asarray(groups[r + 1])
            for a in ii:
                dd = np.abs(((az[jj] - az[a]) + 180) % 360 - 180); o = np.argsort(dd)
                pairs += [(a, jj[o[0]]), (a, jj[o[1]])]
    with ThreadPoolExecutor(8) as ex:
        meas = np.array(list(ex.map(lambda p: xcorr_shift(H[p[0]], H[p[1]]), pairs)))
    n = len(H); A = np.zeros((len(pairs) + 1, n))
    for k, (a, b) in enumerate(pairs): A[k, a] = -1; A[k, b] = 1
    A[-1] = 1; w = np.ones(len(pairs))
    for it in range(4):
        Aw = A.copy(); Aw[:-1] *= w[:, None]
        s = np.stack([np.linalg.lstsq(Aw, np.r_[meas[:, d] * w, 0], rcond=None)[0] for d in range(2)], 1)
        res = np.array([s[b] - s[a] for a, b in pairs]) - meas; rn = np.hypot(*res.T)
        w = (rn < 3 * np.median(rn) + 0.5).astype(float)
    return s, dict(n_pairs=len(pairs), closure_rms_px=float(np.sqrt((rn[w > 0] ** 2).mean())), outliers=int((w == 0).sum()))

def grad_basis(k):
    ky, kx = k[:, 0], k[:, 1]; cols = []
    for deg in (2, 3, 4):
        for p in range(deg + 1):
            q = deg - p
            gy = p * ky ** (p - 1) * kx ** q if p > 0 else np.zeros_like(ky)
            gx = q * ky ** p * kx ** (q - 1) if q > 0 else np.zeros_like(kx)
            cols.append(np.r_[gy, gx])
    return np.stack(cols, 1)

def split(k, s):
    n = len(k); Gb = grad_basis(k); Ga = np.r_[k[:, 1], -k[:, 0]][:, None]
    G0 = np.stack([np.r_[np.ones(n), np.zeros(n)], np.r_[np.zeros(n), np.ones(n)]], 1)
    G = np.c_[Gb, Ga, G0]; coef = np.linalg.lstsq(G, np.r_[s[:, 0], s[:, 1]], rcond=None)[0]
    un = lambda v: np.stack([v[:n], v[n:]], 1)
    cf, an, f0 = un(Gb @ coef[:Gb.shape[1]]), un(Ga[:, 0] * coef[Gb.shape[1]]), un(G0 @ coef[-2:])
    return cf, an, f0, float(coef[Gb.shape[1]])

def shift_stack(R, corr, masks=None):
    out = np.empty(R.shape, np.float32)
    for i in range(len(R)):
        out[i] = ndi.shift(np.asarray(R[i], np.float32), -corr[i], order=3, mode="nearest")
    return out

P = np.loadtxt(W + "d32a/data_reg/positions_um.csv", delimiter=","); az = np.degrees(np.arctan2(P[:, 2], P[:, 1])) % 360
ring = np.repeat(np.arange(8), [15, 20, 25, 30, 35, 40, 45, 50])
summary = {}
# ---- bright field (rings 0-2)
Rb = np.load(FP + "fpt_output_32a_bf/work/preprocessed.npy", mmap_mode="r")
kb = np.array(json.load(open(FP + "fpt_output_32a_bf/work/calibration.json"))["k_solver"])
Hb = [hp2(np.asarray(Rb[i])) for i in range(len(Rb))]
sb, infb = network(Hb, [np.where(ring[:60] == r)[0] for r in range(3)], az[:60])
cf, an, f0, e = split(kb, sb); corr_b = sb - cf - f0
summary["bright"] = dict(infb, rotation_coef_px_per_um_inv=e, raw_rms_px=np.sqrt((sb ** 2).mean(0)).tolist(),
                         curlfree_rms_px=np.sqrt((cf ** 2).mean(0)).tolist(), correction_rms_px=np.sqrt((corr_b ** 2).mean(0)).tolist(),
                         correction_max_px=np.abs(corr_b).max(0).tolist(), correction_px=corr_b.tolist(), shifts_px=sb.tolist())
print("bright:", {k: v for k, v in summary["bright"].items() if not k.endswith("_px") or "rms" in k or "max" in k})
Rbr = shift_stack(Rb, corr_b)
for t in ("bf", "bf_dr200"):
    src, dst = FP + f"fpt_output_32a_{t}/work/", FP + f"fpt_output_32a_{t}_reg/work/"
    os.makedirs(dst, exist_ok=True)
    np.save(dst + "preprocessed.npy", Rbr)
    for f in ("preprocess_meta.npz", "calibration.json"): shutil.copy(src + f, dst + f)
# ---- dark field (rings 6-7) in work_df
WD = W + "d32a/work_df/"; WR = W + "d32a/work_df_reg/"; os.makedirs(WR, exist_ok=True)
Rd = np.load(WD + "preprocessed.npy", mmap_mode="r"); mk = np.load(WD + "dfmask.npy", mmap_mode="r")
cal = json.load(open(WD + "calibration.json")); kd_all = np.array(cal["k_solver"]); use = np.array(cal["positions_index"])
dsel = np.where(np.isin(ring[use], [6, 7]))[0]
Hd = [hp2(np.asarray(Rd[i]) * np.asarray(mk[i], np.float32)) for i in dsel]
sd, infd = network(Hd, [np.where(ring[use][dsel] == r)[0] for r in (6, 7)], az[use][dsel])
cfd, and_, f0d, ed = split(kd_all[dsel], sd); corr_d = sd - cfd - f0d
summary["dark"] = dict(infd, rotation_coef_px_per_um_inv=ed, raw_rms_px=np.sqrt((sd ** 2).mean(0)).tolist(),
                       curlfree_rms_px=np.sqrt((cfd ** 2).mean(0)).tolist(), correction_rms_px=np.sqrt((corr_d ** 2).mean(0)).tolist(),
                       correction_max_px=np.abs(corr_d).max(0).tolist(), correction_px=corr_d.tolist(), shifts_px=sd.tolist())
print("dark:", {k: v for k, v in summary["dark"].items() if not k.endswith("_px") or "rms" in k or "max" in k})
corr_all = np.zeros((len(use), 2)); corr_all[:60] = corr_b; corr_all[dsel] = corr_d
Rdr = shift_stack(Rd, corr_all)
Rdr[:60] = Rbr
np.save(WR + "preprocessed.npy", Rdr)
mkr = np.stack([np.clip(ndi.shift(np.asarray(mk[i], np.float32), -corr_all[i], order=1, mode="nearest"), 0, 1) for i in range(len(use))]).astype(np.float16)
np.save(WR + "dfmask.npy", mkr)
for f in ("calibration.json", "noise.json", "prep_df.json"): shutil.copy(WD + f, WR + f)
summary["note"] = "corrections applied as ndi.shift(image, -correction); bright correction also used for the bright part of work_df_reg"
json.dump(summary, open(W + "d32a/register_32a.json", "w"), indent=1)
print("done")
