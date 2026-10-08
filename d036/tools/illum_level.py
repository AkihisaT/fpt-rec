"""Illumination intensity of the edge / dark-field positions relative to the ring-1 bright level, from the bright part
of their sample-out images: level_n = 99th percentile of G8{D_n} / (g1 U), U = ring-1 envelope (as prep_036_df)."""
import json, numpy as np
from scipy import ndimage as ndi
z = np.load("data/prep_036.npz"); D = z["D"]; ring = z["ring"]; dl = z["dlevel"]
g1 = np.median(dl[:9])
U = ndi.gaussian_filter(np.median(np.stack([D[p] / dl[p] for p in range(9)]), 0), 30); U /= U[256:768, 256:768].mean()
out = []
for p in range(44):
    r = ndi.gaussian_filter(D[p], 8) / (g1 * np.maximum(U, 0.2))
    inner = r[16:-16, 16:-16]
    out.append(dict(pos=p, ring=int(ring[p]), p99=float(np.percentile(inner, 99)), p999=float(np.percentile(inner, 99.9)),
                    frac_above_0p5=float((inner > 0.5).mean())))
for R in (1, 2, 3):
    s = [o for o in out if o["ring"] == R]
    print(R, "p99", np.round([o["p99"] for o in s], 2), "\n   frac>0.5", np.round([o["frac_above_0p5"] for o in s], 3))
json.dump(out, open("data/illum_level.json", "w"), indent=1)
