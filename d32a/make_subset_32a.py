"""32a: image subset folder for the fpt_pipeline preprocess step (relative symlinks into d32a/data_reg).
usage (package root):  python d32a/make_subset_32a.py [name] [a:b | comma list of 0-based indices]
default: data_reg_bf 0:60  (bright-field set = rings 0-2, 60 positions; the input of config_32a_bf*.json)
Writes d32a/<name>/{sample,direct}/a001.tif ... and d32a/<name>/positions_um.csv (rows of d32a/data_reg/positions_um.csv)."""
import glob, os, sys
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
name = sys.argv[1] if len(sys.argv) > 1 else "data_reg_bf"
sel = sys.argv[2] if len(sys.argv) > 2 else "0:60"
idx = list(range(*map(int, sel.split(":")))) if ":" in sel else [int(v) for v in sel.split(",")]
src, dst = W + "d32a/data_reg/", W + f"d32a/{name}/"
for sub in ("sample", "direct"):
    os.makedirs(dst + sub, exist_ok=True)
    for f in glob.glob(dst + sub + "/*"): os.remove(f)
    for j, i in enumerate(idx):
        os.symlink(os.path.relpath(src + f"{sub}/a{i + 1:03d}.tif", dst + sub), dst + f"{sub}/a{j + 1:03d}.tif")
L = open(src + "positions_um.csv").read().strip().split("\n")
open(dst + "positions_um.csv", "w").write("\n".join(L[i] for i in idx) + "\n")
print(name, len(idx), "images ->", dst)
