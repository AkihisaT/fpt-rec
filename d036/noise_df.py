"""Per-image noise variance of the Phase-C data (common intensity units) from the half averages:
sigma2_n = variance of the high-passed (A - B)/2 (Gaussian sigma 10 px removed) over valid pixels (dfmask > 0.5),
for the central 512 px and the full 1024 px field.  Output work_df/noise.json."""
import json, numpy as np
from scipy import ndimage as ndi
W = "work_df/"
A = np.load(W + "preprocessed_A.npy", mmap_mode="r"); B = np.load(W + "preprocessed_B.npy", mmap_mode="r")
F = np.load(W + "preprocessed.npy", mmap_mode="r"); mk = np.load(W + "dfmask.npy", mmap_mode="r")
ring = json.load(open(W + "calibration.json"))["ring"]
out = {"sigma2_512": [], "sigma2_1024": [], "level_512": [], "level_1024": []}
for n in range(A.shape[0]):
    d = 0.5 * (np.asarray(A[n], np.float64) - np.asarray(B[n], np.float64)); d -= ndi.gaussian_filter(d, 10)
    m = np.asarray(mk[n], np.float32) > 0.5
    for key, sl in (("512", slice(256, 768)), ("1024", slice(16, 1008))):   # 16 px border: drift-shift fill
        mm = m[sl, sl]
        out["sigma2_" + key].append(float(d[sl, sl][mm].var()))
        out["level_" + key].append(float(np.asarray(F[n])[sl, sl][mm].mean()))
out["ring"] = ring
json.dump(out, open(W + "noise.json", "w"), indent=1)
s = np.array(out["sigma2_512"]); lv = np.array(out["level_512"]); r = np.array(ring)
for R in (1, 2, 3):
    print(R, "sigma2 median %.3g" % np.median(s[r == R]), "level median %.4g" % np.median(lv[r == R]),
          "ratio sigma2/sigma2_bright %.3g" % (np.median(s[r == R]) / np.median(s[r == 1])), "sigma2/level %.3g" % np.median(s[r == R] / np.abs(lv[r == R])))
