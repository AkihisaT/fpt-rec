"""Common-metric per-ring band misfit (512 crop, dark-mode data) of a BLIS-FPM result npz.
usage: python eval_df.py <npz> [grid]"""
import os, sys, json
import numpy as np, torch
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
os.chdir(W + "epry_pipeline"); sys.path.insert(0, "."); sys.path.insert(0, "../dip_pipeline")
import run_epry as RE
from dipfpm import train as T, evaluate as E
from dipfpm.physics import FPMPhysics, ZERN
f = os.path.abspath(os.path.join(W, sys.argv[1])); grid = int(sys.argv[2]) if len(sys.argv) > 2 else 512
cfg = RE.load_cfg("config_epry_036df.json" if grid == 512 else "config_epry_036df_1024.json", {})
data = T.prepare(cfg, lambda m: None)
ring = np.array(json.load(open(cfg["fpt_work_dir"] + "/calibration.json"))["ring"])[data["keep"]]
pn = E.load_previous_native(f, data["crop"], data["H"], cfg["pixel_um"], data["M"], data["dk"])
phys = FPMPhysics(data["kn"][data["keep"]], data["dk"], data["M"], data["No"], cfg["kc"], cfg["lam_um"], kappa=pn["kappa_solver"],
                  axis_offset_um=data["axis_off"], pupil_radius=cfg["epry"]["pupil_radius"])
band = cfg["export_band"]; dx = cfg["pixel_um"]
a_bp = E.band_pass(pn["a"], dx, *band); p_bp = E.band_pass(pn["phi"], dx, *band)
for lab, (a_, p_) in (("band-passed", (a_bp, p_bp)), ("unfiltered", (pn["a"], pn["phi"]))):
    m, per = E.band_misfit(phys, a_, p_, np.zeros(len(ZERN)), np.zeros((len(data["keep"]), 2)), data["I_meas"], band, W_map=pn["W"])
    print(lab, "ring means", [round(float(per[ring == r].mean()), 4) for r in (1, 2, 3)], "val", np.round(per[[list(data['keep']).index(k) for k in data['val']]], 3))
