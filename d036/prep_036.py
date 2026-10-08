"""036: split the 528 ITEX frames, correct sample drift between the 6 repeats, and median-average.

Acquisition order (scan table 3層高エネczpx6.csv, confirmed from the images): 12 blocks of 44 frames,
block 2r = sample in (repeat r), block 2r+1 = sample out / direct (repeat r), r = 0..5.

Steps
 1. background map B(x): per-pixel median of all direct frames at dark-field positions (ring 3, 19 x 6 frames);
    it contains the camera offset and the stray-light floor. All frames are background-subtracted, and hot pixels
    (B > 250 counts, about 18 % in 036) are replaced by the mean of the good pixels in their 5x5 neighbourhood.
 2. sample drift d(t) (t = frame index): plain cross-correlation of median-filtered, high-passed ring-1 ratio images
    (S_r - B)/(D_r - B) of the same position in repeats r and 0 gives d(t_rp) - d(t_0p); d(t) = a1 t + a2 t^2 + a3 t^3
    is fitted by least squares and every sample frame is shifted back to the t = 0 position (this also removes the
    drift within one 44-frame scan).  Direct frames are not shifted (they record the illumination at that time).
 3. averaging: per-pixel median over repeats (removes zingers).  Bright positions (ring 1, or direct median
    > 20 % of the ring-1 level): median of the ratios R_r, stored as sample = R_med * D_med, direct = D_med, so
    that sample / direct = R_med.  Other positions: median of the shifted, background-subtracted sample frames.
    Halves A = repeats {0,2,4}, B = {1,3,5} are formed the same way (for noise and FRC).
Outputs (d036/data/): sample_tif/ and direct_tif/ with a001..a088 = (A, B) interleaved per position
(pipeline: n_repeat 2, interleaved, dark_offset 0), prep_036.npz (full medians, halves, B, shifts, direct levels),
prep_036.json (summary).
"""
import glob, json, os, shutil, sys, time
import numpy as np
import tifffile
from scipy import ndimage as ndi
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon.preprocess import read_itex

SRC = os.environ.get("FPT036_RAW", "").rstrip("/") + "/"   # raw a001-a528.img folder (override with FPT036_RAW)
if not os.environ.get("FPT036_RAW"): raise SystemExit("生データのフォルダを環境変数 FPT036_RAW で指定してください（paths_local.example.sh を参照）")
OUT = W + "d036/data/"; os.makedirs(OUT + "sample_tif", exist_ok=True); os.makedirs(OUT + "direct_tif", exist_ok=True)
NP, NR = 44, 6
RING = np.r_[[1] * 9, [2] * 16, [3] * 19]
fs = sorted(glob.glob(SRC + "a*.img")); assert len(fs) == 2 * NP * NR, len(fs)
fid = lambda r, p, direct: 2 * NP * r + (NP if direct else 0) + p          # 0-based frame index
load = lambda r, p, d: read_itex(fs[fid(r, p, d)]).astype(np.float32)
t0 = time.time(); log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

# 1. background
dark = [load(r, p, True) for p in np.where(RING == 3)[0] for r in range(NR)]
B = np.median(np.stack(dark), 0).astype(np.float32)
del dark
bg_rep = [float(np.median(np.stack([load(r, p, True) for p in np.where(RING == 3)[0][:6]]))) for r in range(NR)]
log(f"background map: median {np.median(B):.1f}, 1-99% {np.percentile(B, 1):.1f}-{np.percentile(B, 99):.1f}; per-repeat level {np.round(bg_rep, 1)}")
# hot pixels: ~16 % of the pixels have a dark level > 300 counts in 036 (stable offsets, 4x the noise of normal
# pixels; not the same pixels as in 018).  Pixels with B > HOT are replaced in every frame by the mean of the good
# pixels in a 5x5 neighbourhood (normalised convolution) after background subtraction.
HOT = 250.0
good = (B <= HOT).astype(np.float32)
den = ndi.uniform_filter(good, 5)
def fixpix(img):
    num = ndi.uniform_filter(img * good, 5)
    out = img.copy(); bad = good < 0.5
    out[bad] = (num / np.maximum(den, 1e-3))[bad]
    return out
log(f"hot pixels (B > {HOT:.0f}): {100 * (1 - good.mean()):.1f} % of the frame; 5x5 neighbourhood without good pixels: {int((den[good < 0.5] == 0).sum())}")
_load_raw = load
load = lambda r, p, d: fixpix(_load_raw(r, p, d) - B) + B            # corrected frame, same offset convention as before

# 2. drift: sample displacement d(t) versus acquisition time t (frame index), from ring-1 bright-field ratio images
#    of the same position in different repeats: m(r,p) = d(t_rp) - d(t_0p); d(t) = a1 t + a2 t^2 + a3 t^3 (d(0) = 0).
#    Plain (not whitened) cross-correlation of median-filtered, high-passed ratio images (whitened phase correlation
#    locks onto the fixed detector pattern and returns zero).
def hp_ratio(s, d):
    r = ndi.median_filter((s - B) / np.maximum(d - B, 50.0), 3)
    r = r / np.maximum(ndi.gaussian_filter(r, 15), 1e-3) - 1
    return np.clip(r, -1, 1)[256:-256, 256:-256]

def xcorr_shift(a, b, maxs=120):
    """shift (dy, dx) that moves b onto a (cross-correlation, parabolic sub-pixel peak, search |s| <= maxs)."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    a = (a - a.mean()) * win; b = (b - b.mean()) * win
    c = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real)
    h0, h1 = c.shape[0] // 2, c.shape[1] // 2
    cw = c[h0 - maxs:h0 + maxs + 1, h1 - maxs:h1 + maxs + 1]
    iy, ix = np.unravel_index(np.argmax(cw), cw.shape); iy += h0 - maxs; ix += h1 - maxs
    dy = 0.5 * (c[iy - 1, ix] - c[iy + 1, ix]) / (c[iy - 1, ix] - 2 * c[iy, ix] + c[iy + 1, ix])
    dx = 0.5 * (c[iy, ix - 1] - c[iy, ix + 1]) / (c[iy, ix - 1] - 2 * c[iy, ix] + c[iy, ix + 1])
    cc = float(c[iy, ix] / np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return np.array([iy + dy - h0, ix + dx - h1]), cc

ring1 = np.where(RING == 1)[0]
meas = []                                          # (t_r, t_0, dy, dx, cc)
for p in ring1:
    ref = hp_ratio(load(0, p, False), load(0, p, True))
    for r in range(1, NR):
        v, cc = xcorr_shift(ref, hp_ratio(load(r, p, False), load(r, p, True)))
        meas.append((fid(r, p, False), fid(0, p, False), -v[0], -v[1], cc))   # displacement of the sample = -shift
meas = np.array(meas)
T = 2 * NP * NR
def basis(t):
    x = np.asarray(t, float) / T
    return np.stack([x, x ** 2, x ** 3], -1)
Adesign = basis(meas[:, 0]) - basis(meas[:, 1])
coef = np.stack([np.linalg.lstsq(Adesign, meas[:, 2 + k], rcond=None)[0] for k in range(2)], 1)     # (3, 2)
resid = meas[:, 2:4] - Adesign @ coef
drift = lambda t: basis(t) @ coef                                                                  # (.., 2) px
log(f"drift model: d(end) = ({drift(T - 1)[0]:+.1f}, {drift(T - 1)[1]:+.1f}) px over {T} frames; "
    f"fit residual RMS {np.sqrt((resid ** 2).mean(0)).round(2)} px; median xcorr {np.median(meas[:, 4]):.2f}")
shr = np.array([drift(fid(r, 4, False)) for r in range(NR)])
def shift_at(r, p):
    return -drift(fid(r, p, False))                  # shift that brings frame (r, p) back to the t = 0 sample position

# 3. averaging
res = {k: np.zeros((NP, 1024, 1024), np.float32) for k in ("S", "D", "SA", "SB", "DA", "DB")}
bright = np.zeros(NP, bool); dlevel = np.zeros(NP); shifts = np.zeros((NP, NR, 2))
HALF = {"": list(range(NR)), "A": [0, 2, 4], "B": [1, 3, 5]}
for p in range(NP):
    S, D = [], []
    for r in range(NR):
        s = load(r, p, False) - B; d = load(r, p, True) - B
        shifts[p, r] = shift_at(r, p)
        S.append(ndi.shift(s, shifts[p, r], order=1, mode="constant", cval=np.nan)); D.append(d)
    S, D = np.stack(S), np.stack(D)
    dlevel[p] = float(np.median(D))
    ref1 = np.median(dlevel[:9]) if p >= 9 else dlevel[p]
    bright[p] = RING[p] == 1 or dlevel[p] > 0.2 * ref1
    for key, idx in HALF.items():
        Dm = np.median(D[idx], 0)
        res["D" + key][p] = Dm
        v = np.nanmedian((S / np.maximum(D, 1.0))[idx], 0) * Dm if bright[p] else np.nanmedian(S[idx], 0)
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = Dm[bad] if bright[p] else 0.0
        res["S" + key][p] = v
    if p % 11 == 0 or p == NP - 1:
        log(f"position {p + 1}/{NP}: ring {RING[p]}, direct median {dlevel[p]:.0f}, bright={bool(bright[p])}")
for p in range(NP):
    for h, key in ((0, "A"), (1, "B")):
        tifffile.imwrite(OUT + f"sample_tif/a{2 * p + h + 1:03d}.tif", res["S" + key][p])
        tifffile.imwrite(OUT + f"direct_tif/a{2 * p + h + 1:03d}.tif", res["D" + key][p])
np.savez_compressed(OUT + "prep_036.npz", S=res["S"], D=res["D"], SA=res["SA"], SB=res["SB"], DA=res["DA"], DB=res["DB"],
                    B=B, shifts=shifts, drift_ring1=shr, drift_meas=meas, drift_coef=coef, bright=bright, dlevel=dlevel, ring=RING)
json.dump(dict(hot_pixel_threshold=HOT, hot_pixel_fraction=float(1 - good.mean()), background_median=float(np.median(B)), background_per_repeat=bg_rep, drift_ring1_px=shr.tolist(), drift_fit_residual_rms_px=np.sqrt((resid ** 2).mean(0)).tolist(), drift_end_px=drift(T - 1).tolist(),
               bright_positions=np.where(bright)[0].tolist(), direct_median_counts=dlevel.round(1).tolist(),
               sample_median_counts=[float(np.median(res["S"][p])) for p in range(NP)]),
          open(OUT + "prep_036.json", "w"), indent=1)
shutil.copy(SRC + "3層高エネczp.csv", OUT + "3層高エネczp.csv")
log(f"done: bright positions {int(bright.sum())}/{NP}")
