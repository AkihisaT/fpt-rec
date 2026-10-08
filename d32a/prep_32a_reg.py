"""32a preprocessing step 0: dark offset and lens-detector mismatch registration.

The objective FZP X stage moves 2.5 um per pulse (CSV column FZP X = integer pulse count = actual lens position), but the
detector values were computed for a lens position on a 1.25 um grid (CSV detector X / -527.75 is a half-integer in ~44 %
of the rows). A mismatch delta_d between the lens position and the lens position assumed by the detector moves the image
by delta_d in the sample plane, so every frame with a half-pulse mismatch is displaced by 1.25 um = 39.1 px along
the image columns (Y: 0.2 um steps, residual <= 3.2 px along the rows). The star centre in the ratio images follows
this with slope -1.0 (rings 0-2). Sample and direct frames of each position are shifted back by the residual.
Output: d32a/data_reg/{sample,direct}/a001..a260.tif (float32, dark offset 100 subtracted, counts per exposure as recorded),
positions_um.csv (idx, FZP X*2.5 um, FZP Y*0.2 um, det X, det Y, exposure s), prep_reg.json."""
import json, os, sys
import numpy as np, tifffile
from scipy import ndimage as ndi
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon.preprocess import read_itex
RAW = os.environ.get("FPT32A_RAW", "").rstrip("/") + "/"
if not os.environ.get("FPT32A_RAW"): raise SystemExit("生データのフォルダを環境変数 FPT32A_RAW で指定してください（paths_local.example.sh を参照）")
OUT = W + "d32a/data_reg/"; OFF = 100.0; PX = 0.0319
P = np.loadtxt(RAW + "平行照明計算fzp250nm.csv", delimiter=",")
ux, uy = 2.5, 0.2                              # um per FZP-stage unit (Y: 0.2 um makes the scan rings circular)
kx, ky = -527.75, -42.2191                     # detector units per FZP unit (ratio 12.5 = ux/uy)
resX = (P[:, 3] / kx - P[:, 1]) * ux / PX      # px, image columns
resY = (P[:, 4] / ky - P[:, 2]) * uy / PX      # px, image rows
for sub in ("sample", "direct"): os.makedirs(OUT + sub, exist_ok=True)
for i in range(len(P)):
    for sub, src in (("sample", "1_sample"), ("direct", "2_direct")):
        img = read_itex(RAW + f"{src}/a{i + 1:03d}.img").astype(np.float32) - OFF
        img = ndi.shift(img, (resY[i], resX[i]), order=3, mode="nearest").astype(np.float32)
        tifffile.imwrite(OUT + f"{sub}/a{i + 1:03d}.tif", img)
rows = [f"{i},{P[i,1]*ux:.4f},{P[i,2]*uy:.4f},{P[i,3]:.0f},{P[i,4]:.0f},{P[i,5]:g}" for i in range(len(P))]
open(OUT + "positions_um.csv", "w").write("\n".join(rows) + "\n")
ring = np.repeat(np.arange(8), [15, 20, 25, 30, 35, 40, 45, 50])
json.dump(dict(dark_offset=OFF, unit_x_um=ux, unit_y_um=uy, det_per_unit=[kx, ky], shift_row_px=resY.tolist(), shift_col_px=resX.tolist(),
               ring=ring.tolist(), exposure_s=P[:, 5].tolist()), open(OUT + "prep_reg.json", "w"), indent=1)
print("done", len(P), "positions; col shifts px:", sorted(set(np.round(resX, 1))))
