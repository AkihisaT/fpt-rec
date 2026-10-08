"""Physical-frame TIFF export (fptrecon.postprocess.export) of the 036 Phase-C BLIS-FPM solutions (rings 1 + 3, 28 images)
fitted to all 28 images (the hold-out solutions are used only for the metrics); output d036/blis_df13/export/."""
import json, os, sys
import numpy as np
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon.config import load_config
from fptrecon import postprocess as post
cfg = load_config(W + "fpt_pipeline/config_036bf.json")
out = W + "d036/blis_df13/export/"
for tag, f in (("BLIS_df_planewave", "s+0"), ("BLIS_df_FOV_c1", "s-1")):
    z = dict(np.load(W + f"d036/blis_df13/{f}.npz"))
    res = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z}
    phys = post.to_physical(res, -1, cfg, 1024)
    post.export(out, tag, phys, res, cfg, extra=dict(images="rings 1+3 (28)", note="036 Phase C dark mode"))
    print(tag, "exported")
