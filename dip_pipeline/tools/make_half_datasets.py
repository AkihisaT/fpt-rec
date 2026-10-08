"""Independent half datasets for FRC: repeat 1 (A) and repeat 2 (B), each flat-fielded with its own direct frame.
Same zinger removal / flattening as fptrecon.preprocess; calibration copied from the full-data work dir."""
import argparse, json, os, shutil, sys
import numpy as np
from scipy import ndimage as ndi
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--fpt-pipeline", default="../fpt_pipeline", help="folder containing the fptrecon package")
ap.add_argument("--config", default="../fpt_pipeline/config_003.json", help="fptrecon config (data_dir etc.)")
ap.add_argument("--calibration", default="../fpt_pipeline/test_full/work/calibration.json")
ap.add_argument("--out", default="half_work")
args = ap.parse_args()
sys.path.insert(0, os.path.abspath(args.fpt_pipeline))
from fptrecon import preprocess as pre
from fptrecon.config import load_config

cfg = load_config(args.config)
pos = pre.load_positions(os.path.join(cfg["data_dir"], cfg["positions_csv"]))
npos, nrep = len(pos), cfg["n_repeat"]
S, _ = pre.load_stack(os.path.join(cfg["data_dir"], cfg["sample_subdir"]))
D, _ = pre.load_stack(os.path.join(cfg["data_dir"], cfg["direct_subdir"]))
S -= cfg["dark_offset"]; D -= cfg["dark_offset"]
S = pre.group_repeats(S, npos, nrep, cfg["repeat_order"]); D = pre.group_repeats(D, npos, nrep, cfg["repeat_order"])
for h, name in enumerate(["A", "B"]):
    R = S[:, h] / np.maximum(D[:, h], 1.0); R /= R.mean((1, 2), keepdims=True)
    nfix = 0
    for i in range(npos):
        med = ndi.median_filter(R[i], 3); m = np.abs(R[i] - med) > cfg["zinger_threshold"]; R[i][m] = med[m]; nfix += int(m.sum())
        R[i] = R[i] / ndi.gaussian_filter(R[i], cfg["flatten_sigma_px"], mode="reflect")
    out = os.path.join(args.out, name); os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "preprocessed.npy"), R.astype(np.float32))
    shutil.copy(args.calibration, os.path.join(out, "calibration.json"))
    print(name, R.shape, "zingers", nfix, "std", float(R[:, 500:1000, 500:1000].std()))
A = np.load(os.path.join(args.out, "A", "preprocessed.npy"), mmap_mode="r"); B = np.load(os.path.join(args.out, "B", "preprocessed.npy"), mmap_mode="r")
c = slice(494, 1006)
print("corr(A, B) of image 0 on the centre crop:", float(np.corrcoef(A[0, c, c].ravel(), B[0, c, c].ravel())[0, 1]))
