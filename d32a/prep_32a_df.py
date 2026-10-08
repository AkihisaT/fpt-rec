"""32a: data for the reconstructions with dark-field images (036 Phase-C scheme adapted to the parallel-beam FZP scan).

Images: bright field = rings 0-2 (60 positions, 1 s) exactly as the BF set (fptrecon preprocessing of d32a/data_reg:
flat-fielded ratio S/D, mean 1, zinger fix, sigma-30 flattening); dark field = rings 6-7 (95 positions, 10 s,
|k|/kc 1.47-1.64). For a dark-field position (registered, dark-offset-subtracted frames s, d, exposure t):
    n(x) = [ s(x) - G3{d}(x) ] / ( t * g * U(x) )
  G3{d}: the sample-out frame of the same position (Gaussian sigma 3 px): background (dark current, stray light, FZP
         0th/higher orders of the empty beam) and any direct-beam tail;
  g:     median bright-field direct level per second (ring-0 direct frames);
  U:     illumination envelope (slit-limited band): Gaussian(sigma 30 px) of the per-pixel median of the 15 ring-0 direct
         frames, each normalised by its own median; normalised to mean 1 over the central 512 px.
Values are dark-field intensities in units of the bright-field level on a common scale.
Hot pixels (per-pixel median of the 10 s sample-out frames of rings 4-7 above median + 8 MAD) are replaced by the local
mean of good pixels in s and d; single-frame zingers in n (> 8 sigma from the 3x3 median) are replaced by the median.
dfmask (per image, 0..1): 1 for bright images; for dark images 0 where the sample-out frame shows direct light
(G8{d}/(t g) > 0.05..0.15 -> 1..0, smoothed) and where U < 0.08 (outside the illuminated band the division by U is noise).
noise.json: sigma2_512 / sigma2_1024 (white-noise variance of each image in its own units, from the spectral plateau
|q| = 6.5-14 um^-1 of the masked image, i.e. above the signal band of every image), level_512 / level_1024 (masked mean).
Outputs d32a/work_df/: preprocessed.npy (155, 1000, 1000) float32, dfmask.npy (float16), calibration.json, noise.json, prep_df.json."""
import json, os, subprocess, sys, time
import numpy as np, tifffile
from scipy import ndimage as ndi
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
REG = W + "d32a/data_reg/"; OUT = W + "d32a/work_df/"; os.makedirs(OUT, exist_ok=True)
BF = W + "fpt_pipeline/fpt_output_32a_bf/work/"
meta = json.load(open(REG + "prep_reg.json")); ring_all = np.array(meta["ring"]); expo = np.array(meta["exposure_s"])
bright = np.where(ring_all <= 2)[0]; darkp = np.where(ring_all >= 6)[0]
use = np.r_[bright, darkp]
t0 = time.time(); log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)
rd = lambda sub, i: tifffile.imread(REG + f"{sub}/a{i + 1:03d}.tif").astype(np.float32)
# bright level and illumination envelope from the ring-0 direct frames (1 s)
d0 = [rd("direct", i) / expo[i] for i in np.where(ring_all == 0)[0]]
lev = np.array([np.median(d) for d in d0]); g = float(np.median(lev))
U = ndi.gaussian_filter(np.median(np.stack([d / l for d, l in zip(d0, lev)]), 0), 30)
c = slice(244, 756); U = (U / U[c, c].mean()).astype(np.float32)
log(f"bright level g = {g:.1f} counts/s; envelope U: centre-512 range {U[c, c].min():.3f}-{U[c, c].max():.3f}, full {U.min():.3f}-{U.max():.3f}")
# hot pixels from the 10 s sample-out frames
Dm = np.median(np.stack([rd("direct", i) for i in np.where(ring_all >= 4)[0][::3]]), 0)
med, mad = np.median(Dm), np.median(np.abs(Dm - np.median(Dm)))
good = (Dm <= med + 8 * 1.4826 * mad).astype(np.float32); den = ndi.uniform_filter(good, 5)
log(f"hot pixels: {int((good < 0.5).sum())} (> median + 8 MAD of the 10 s sample-out median frame)")
def fix(img):
    num = ndi.uniform_filter(img * good, 5); out = img.copy(); bad = good < 0.5
    out[bad] = (num / np.maximum(den, 1e-3))[bad]; return out
Rbf = np.load(BF + "preprocessed.npy", mmap_mode="r"); assert Rbf.shape[0] == len(bright)
res = np.zeros((len(use), 1000, 1000), np.float32); mask = np.ones((len(use), 1000, 1000), np.float16)
res[:len(bright)] = Rbf
info = [dict(pos=int(p), ring=int(ring_all[p]), level=1.0, masked_frac_1000=0.0, masked_frac_512=0.0) for p in bright]
qf = np.fft.fftfreq(1000, 0.0319); QR = np.hypot(*np.meshgrid(qf, qf, indexing="ij")); plate = (QR > 6.5) & (QR < 14.0)
uband = np.clip((U - 0.08) / 0.07, 0, 1)
for j, p in enumerate(darkp):
    s = fix(rd("sample", p)); d = fix(rd("direct", p)); t = expo[p]
    n = (s - ndi.gaussian_filter(d, 3)) / (t * g * np.maximum(U, 0.02))
    md = ndi.median_filter(n, 3); sig = 1.4826 * np.median(np.abs(n - md)); z = np.abs(n - md) > 8 * sig; n[z] = md[z]
    Dn = ndi.gaussian_filter(d, 8) / (t * g)
    m = np.clip((0.15 - Dn) / 0.10, 0, 1) * uband; m = np.clip(ndi.gaussian_filter(m, 6), 0, 1)
    res[len(bright) + j] = n; mask[len(bright) + j] = m.astype(np.float16)
    info.append(dict(pos=int(p), ring=int(ring_all[p]), level=float(np.median(n[c, c])), zingers=int(z.sum()),
                     masked_frac_1000=float((m < 0.5).mean()), masked_frac_512=float((m[c, c] < 0.5).mean())))
    if j % 20 == 0: log(f"dark {j}/{len(darkp)} pos {p + 1}: level {info[-1]['level']:.5f}, masked {info[-1]['masked_frac_1000']:.3f} / {info[-1]['masked_frac_512']:.3f}")
def plateau_var(img, m, sl):
    N = img[sl, sl].shape[0]; w = np.outer(np.hanning(N), np.hanning(N)) * m[sl, sl]
    mu = (img[sl, sl] * m[sl, sl]).sum() / m[sl, sl].sum()
    F = np.fft.fft2((img[sl, sl] - mu) * w); qq = np.fft.fftfreq(N, 0.0319); qr = np.hypot(*np.meshgrid(qq, qq, indexing="ij"))
    pl = (qr > 6.5) & (qr < 14.0)
    return float((np.abs(F[pl]) ** 2).mean() / (w ** 2).sum()), float(mu)
noise = {"sigma2_512": [], "sigma2_1024": [], "level_512": [], "level_1024": []}
for i in range(len(use)):
    m = mask[i].astype(np.float32)
    for key, sl in (("512", c), ("1024", slice(0, 1000))):
        v, mu = plateau_var(res[i], m, sl); noise["sigma2_" + key].append(v); noise["level_" + key].append(mu)
noise["ring"] = [int(ring_all[p]) for p in use]
noise["note"] = "sigma2_1024 / level_1024 refer to the full 1000 px field; white-noise variance from the |q| 6.5-14 um^-1 plateau"
np.save(OUT + "preprocessed.npy", res); np.save(OUT + "dfmask.npy", mask)
json.dump(noise, open(OUT + "noise.json", "w"), indent=1)
json.dump(dict(bright_level_counts_per_s=g, positions=info, use=use.tolist()), open(OUT + "prep_df.json", "w"), indent=1)
subprocess.run([sys.executable, W + "d32a/make_calib_32a.py", OUT, sys.argv[1] if len(sys.argv) > 1 else "0.25",
                sys.argv[2] if len(sys.argv) > 2 else "5000", ",".join(str(int(v)) for v in use)], check=True)
s2 = np.array(noise["sigma2_512"]); lv = np.array(noise["level_512"]); rr = np.array(noise["ring"])
for R_ in (0, 1, 2, 6, 7):
    sel = rr == R_
    log(f"ring {R_}: level {np.median(lv[sel]):.5f}, sigma2 {np.median(s2[sel]):.3g}, sigma2/level^2 {np.median(s2[sel] / lv[sel] ** 2):.3g}")
log("done")
