"""036 Phase C: data for the 44-image reconstruction including the edge / dark-field positions (rings 2 and 3).

Bright positions (ring 1) keep exactly the Phase-B data (fpt_output_036bf/work/preprocessed.npy: flat-fielded ratio
S/D of the two half averages, normalised to mean 1, zinger-fixed and flattened).  Positions whose direct beam leaves the
objective pupil cannot be divided by the sample-out image, so for rings 2 and 3 every frame is treated as follows
(repeat r, position p; all frames background-subtracted and hot-pixel fixed as in prep_036.py):

    n_rp(x) = [ s_rp(x) - G3{d_rp}(x) ] / ( g_r * U_r(x) )

  s_rp   sample-in frame, d_rp  sample-out frame of the same position (G3: Gaussian, sigma 3 px) -> removes the
         unscattered tail of the direct beam (partial coherence / soft pupil edge; up to ~5 % of the bright level in the
         centre of the ring-2 images) together with any residual stray light;
  g_r    median ring-1 direct level of repeat r (bright-field level; follows the source intensity);
  U_r    illumination envelope of repeat r: Gaussian(sigma 30 px) of the per-pixel median of the 9 ring-1 direct frames,
         each normalised by its own median, normalised to mean 1 over the central 512 px.
n_rp is shifted by the sample-drift correction of prep_036.py and median-averaged over the half (A = {0,2,4},
B = {1,3,5}); the full data are (A + B) / 2, as for ring 1.  Values are therefore dark-field intensities in units of
the bright-field level (ring 2 ~ 1-3 %, ring 3 ~ 0.02-0.3 %), on a common scale; per-image gains are fitted.

dfmask (per image, 0..1): 1 for ring 1; for rings 2-3 pixels where the sample-out image is bright
(G8{d}/g > 0.05 .. 0.15 -> 1 .. 0, smoothed) are excluded (bright-field light there is not described by the
subtracted data).

Outputs d036/work_df/: preprocessed.npy, preprocessed_A.npy, preprocessed_B.npy (44, 1024, 1024) float32,
dfmask.npy (float16), calibration.json (Phase-B calibration with k for all 44 images from the stage->k map,
'dark_images'), prep_df.json (levels, masked fractions, band noise fractions)."""
import glob, json, os, sys, time
import numpy as np
from scipy import ndimage as ndi
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon.preprocess import read_itex
from fptrecon.config import load_config

SRC = os.environ.get("FPT036_RAW", "").rstrip("/") + "/"   # raw a001-a528.img folder (override with FPT036_RAW)
if not os.environ.get("FPT036_RAW"): raise SystemExit("生データのフォルダを環境変数 FPT036_RAW で指定してください（paths_local.example.sh を参照）")
OUT = W + "d036/work_df/"; os.makedirs(OUT, exist_ok=True)
BF = W + "fpt_pipeline/fpt_output_036bf/work/"
NP, NR = 44, 6
z = np.load(W + "d036/data/prep_036.npz")
RING = z["ring"]; B = z["B"]; SH = z["shifts"]
fs = sorted(glob.glob(SRC + "a*.img")); assert len(fs) == 2 * NP * NR
fid = lambda r, p, direct: 2 * NP * r + (NP if direct else 0) + p
t0 = time.time(); log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)
good = (B <= 250.0).astype(np.float32); den = ndi.uniform_filter(good, 5)

def load(r, p, d):
    img = read_itex(fs[fid(r, p, d)]).astype(np.float32) - B
    num = ndi.uniform_filter(img * good, 5); out = img.copy(); bad = good < 0.5
    out[bad] = (num / np.maximum(den, 1e-3))[bad]
    return out

c = slice(256, 768)
ring1 = np.where(RING == 1)[0]
g, U = np.zeros(NR), []
for r in range(NR):
    d1 = [load(r, p, True) for p in ring1]
    lev = np.array([np.median(d) for d in d1]); g[r] = np.median(lev)
    u = ndi.gaussian_filter(np.median(np.stack([d / l for d, l in zip(d1, lev)]), 0), 30)
    U.append((u / u[c, c].mean()).astype(np.float32))
log(f"bright level per repeat {g.round(0)}; envelope centre-512 range {min(u[c, c].min() for u in U):.3f}-{max(u[c, c].max() for u in U):.3f}")

cfg = load_config(W + "fpt_pipeline/config_036bf.json")
Rbf = np.load(BF + "preprocessed.npy")
assert Rbf.shape[0] == len(ring1)
def bright_ratio(Sm, Dm):
    """same steps as fptrecon.preprocess for one image"""
    R = Sm / np.maximum(Dm, 1.0); R /= R.mean()
    med = ndi.median_filter(R, 3); m = np.abs(R - med) > cfg["zinger_threshold"]; R[m] = med[m]
    return R / ndi.gaussian_filter(R, cfg["flatten_sigma_px"], mode="reflect")

res = {k: np.zeros((NP, 1024, 1024), np.float32) for k in ("", "A", "B")}
mask = np.ones((NP, 1024, 1024), np.float16)
info = []
for p in range(NP):
    if RING[p] == 1:
        i = int(np.where(ring1 == p)[0][0])
        res[""][p] = Rbf[i]
        res["A"][p] = bright_ratio(z["SA"][p], z["DA"][p]); res["B"][p] = bright_ratio(z["SB"][p], z["DB"][p])
        info.append(dict(pos=p, ring=1, level=1.0, masked_frac_1024=0.0, masked_frac_512=0.0))
        continue
    N, Dsum = [], 0
    for r in range(NR):
        s = load(r, p, False); d = load(r, p, True)
        n = (s - ndi.gaussian_filter(d, 3)) / (g[r] * U[r])
        N.append(ndi.shift(n, SH[p, r], order=1, mode="constant", cval=np.nan)); Dsum = Dsum + d / g[r]
    N = np.stack(N)
    for key, idx in (("A", [0, 2, 4]), ("B", [1, 3, 5])):
        v = np.nanmedian(N[idx], 0); v[~np.isfinite(v)] = 0.0; res[key][p] = v
    res[""][p] = 0.5 * (res["A"][p] + res["B"][p])
    Dn = ndi.gaussian_filter(Dsum / NR, 8)
    m = np.clip((0.15 - Dn) / 0.10, 0, 1); m = ndi.gaussian_filter(m, 6)
    mask[p] = m.astype(np.float16)
    info.append(dict(pos=p, ring=int(RING[p]), level=float(np.median(res[""][p][c, c])),
                     masked_frac_1024=float((m < 0.5).mean()), masked_frac_512=float((m[c, c] < 0.5).mean())))
    if p % 5 == 0: log(f"position {p + 1}: ring {RING[p]}, level {info[-1]['level']:.4f}, masked {info[-1]['masked_frac_1024']:.3f} (1024) / {info[-1]['masked_frac_512']:.3f} (512)")

# band noise fraction in normalised contrast (central 512 and full field), 0.3-3.5 um^-1
def band_noise(Nc, sl):
    dx = cfg["pixel_um"]; win = np.outer(np.hanning(Nc), np.hanning(Nc))
    q = np.fft.fftfreq(Nc, dx); QR = np.hypot(*np.meshgrid(q, q, indexing="ij")); b = (QR >= 0.3) & (QR <= 3.5)
    out = []
    for p in range(NP):
        mm = mask[p][sl, sl].astype(np.float32) * win
        f, a_, b_ = res[""][p][sl, sl], res["A"][p][sl, sl], res["B"][p][sl, sl]
        mu = (f * mm).sum() / mm.sum()
        Ps = (np.abs(np.fft.fft2((f / mu - 1) * mm)[b]) ** 2).sum(); Pn = (np.abs(np.fft.fft2((a_ - b_) / 2 / mu * mm)[b]) ** 2).sum()
        out.append(float(Pn / Ps))
    return out
nf512, nf1024 = band_noise(512, c), band_noise(1024, slice(0, 1024))
for i in range(NP): info[i]["noise_frac_512"], info[i]["noise_frac_1024"] = nf512[i], nf1024[i]
for k, v in res.items(): np.save(OUT + f"preprocessed{'_' + k if k else ''}.npy", v)
np.save(OUT + "dfmask.npy", mask)
cal = json.load(open(BF + "calibration.json"))
ill = json.load(open(W + "d036/data/illumination_036.json"))["positions"]
kbf = np.array(cal["k_solver"]); kall = np.array([q["k"] for q in ill])
assert np.abs(kall[ring1] - kbf).max() < 2e-3, "stage->k map does not reproduce the ring-1 calibration"
kall[ring1] = kbf
cal.update(k_solver=kall.tolist(), k_over_kc=(np.hypot(*kall.T) / cfg["kc"] if "kc" in cfg else np.hypot(*kall.T) / 2.5).tolist(),
           dark_images=[int(p) for p in np.where(RING > 1)[0]], ring=[int(v) for v in RING],
           per_image_misfit_ring1=cal.pop("per_image_misfit", None),
           note="Phase C: ring-1 calibration; k of rings 2-3 from the fitted stage->k affine map")
json.dump(cal, open(OUT + "calibration.json", "w"), indent=1)
json.dump(dict(bright_level_per_repeat=g.tolist(), positions=info), open(OUT + "prep_df.json", "w"), indent=1)
for R_ in (2, 3):
    sel = [x for x in info if x["ring"] == R_]
    log(f"ring {R_}: level {np.median([x['level'] for x in sel]):.4f}, band noise fraction 512 median {np.median([x['noise_frac_512'] for x in sel]):.3f}, "
        f"1024 {np.median([x['noise_frac_1024'] for x in sel]):.3f}; masked 1024 max {max(x['masked_frac_1024'] for x in sel):.3f}")
log("done")
