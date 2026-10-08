# -*- coding: utf-8 -*-
# Write the DIP (optimal it.) and BLIS-FPM reconstructions of the partial-coherence comparison as TIFF (float32, physical frame),
# from the arrays compare_018.py stored in recs.npz (identical to its own TIFF export for DIP 30 it. / EPRY).
# usage (workspace root): python d036/pc/export_pc_tiffs.py [out dirs ...]   default: d036/pc/out_1024_pc d036/pc/out_1024_pc1m
import os, sys
import numpy as np, tifffile
dirs = sys.argv[1:] or ["d036/pc/out_1024_pc", "d036/pc/out_1024_pc1m"]
NAME = {"DIP (optimal it.)": "DIP", "BLIS-FPM": "BLIS"}
MOD = {"FOV model": "FOV_c1", "plane-wave model": "planewave"}
FILE = {"phi": "phase_rad", "phi_unf": "phase_rad_unfiltered", "T": "transmission"}
for d in dirs:
    r = np.load(os.path.join(d, "recs.npz")); os.makedirs(os.path.join(d, "tiff"), exist_ok=True); n = 0
    for k in r.files:
        meth, rest = k.split(", ", 1); mod, q = rest.split("|")
        if meth in NAME and q in FILE:
            tifffile.imwrite(os.path.join(d, "tiff", f"{NAME[meth]}_{MOD[mod]}_{FILE[q]}.tif"), r[k].astype(np.float32)); n += 1
    print(d, n, "files")
