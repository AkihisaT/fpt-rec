"""Bright-field subset of 036 for the unmodified pipelines: ring-1 positions (1-9) only, halves A/B interleaved."""
import os, shutil
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
src, dst = W + "d036/data/", W + "d036/data_bf/"
for sub in ("sample_tif", "direct_tif"):
    os.makedirs(dst + sub, exist_ok=True)
    for i in range(1, 19):
        shutil.copy(src + f"{sub}/a{i:03d}.tif", dst + f"{sub}/a{i:03d}.tif")
lines = [l for l in open(src + "3層高エネczp.csv", encoding="utf-8-sig").read().splitlines() if l.strip()]
open(dst + "czp_ring1.csv", "w").write("\n".join(lines[:9]) + "\n")
print(len(lines), "positions in csv;", "ring-1 rows:", lines[:9])
