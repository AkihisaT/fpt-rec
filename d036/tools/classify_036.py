"""Illumination k for all 44 positions from the ring-1 calibration (affine stage->k map), bright/edge/dark classes
from the direct-beam coverage of the (median) direct images, and the direction of the vignetting in each direct image."""
import json, sys, numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, "fpt_pipeline")
from fptrecon.wotf import kvec_affine
from fptrecon.preprocess import load_positions
cal = json.load(open("fpt_pipeline/fpt_output_036bf/work/calibration.json"))
P = load_positions("d036/data/3層高エネczp.csv")
kn = kvec_affine(P, np.array(cal["Mk"]), cal["k0"]); kc = 2.5
z = np.load("d036/data/prep_036.npz"); D = z["D"]; ring = z["ring"]
ref = np.median([np.median(D[p]) for p in range(9)])
cover = np.array([float((ndi.uniform_filter(D[p], 16) > 0.2 * ref).mean()) for p in range(44)])
cls = np.where(cover > 0.95, "bright", np.where(cover > 0.02, "edge", "dark"))
# vignetting direction: gradient of the smoothed log direct image (bright side), compared with the k azimuth
yy, xx = np.mgrid[0:1024, 0:1024] - 511.5
out = []
for p in range(44):
    d = ndi.uniform_filter(np.maximum(D[p], 1), 32)
    w = d / d.sum(); cy, cx = float((w * yy).sum()), float((w * xx).sum())      # intensity centroid (px from centre)
    out.append(dict(pos=p + 1, ring=int(ring[p]), k=[round(float(kn[p, 0]), 3), round(float(kn[p, 1]), 3)], k_over_kc=round(float(np.hypot(*kn[p]) / kc), 3),
                    direct_cover=round(cover[p], 3), cls=str(cls[p]), direct_centroid_px=[round(cy, 1), round(cx, 1)]))
json.dump(dict(kc=kc, positions=out), open("d036/data/illumination_036.json", "w"), indent=1)
for r in (1, 2, 3):
    sel = [o for o in out if o["ring"] == r]
    print(f"ring {r}: |k|/kc {min(o['k_over_kc'] for o in sel):.2f}-{max(o['k_over_kc'] for o in sel):.2f}; classes", {c: sum(o['cls'] == c for o in sel) for c in ('bright', 'edge', 'dark')},
          "cover", [o["direct_cover"] for o in sel][:8])
# does the bright side of ring-1/2 direct images point along -k or +k?  (FOV effect -> depends on the k azimuth)
for o in out[:25]:
    ky, kx = o["k"]; cy, cx = o["direct_centroid_px"]
    o["cos"] = (ky * cy + kx * cx) / (np.hypot(ky, kx) * max(np.hypot(cy, cx), 1e-6))
print("ring 1 cos(k, bright-centroid):", [round(o["cos"], 2) for o in out[:9]], "| centroid radius px", [round(float(np.hypot(*o['direct_centroid_px'])), 0) for o in out[:9]])
print("ring 2 cos(k, bright-centroid):", [round(o["cos"], 2) for o in out[9:25]])
