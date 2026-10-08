"""BLIS-FPM (fptrecon band-limited intensity-spectrum LS) on the same 512 px centre crop, for full data and the
two half datasets (for half-data FRC).  Zero initial object (no shared linear initialisation between halves).
usage: python run_prev_crop.py <work_dir> <out.npz> <fov_factor> [threads] [fpt_pipeline_dir] [fptrecon_config]"""
import json, sys, os, time
import numpy as np
FPT = os.path.abspath(sys.argv[5]) if len(sys.argv) > 5 else os.path.abspath("../fpt_pipeline")
sys.path.insert(0, FPT)
from fptrecon.config import load_config
from fptrecon import nonlinear as nlr
work, out, c = sys.argv[1], sys.argv[2], float(sys.argv[3]); th = int(sys.argv[4]) if len(sys.argv) > 4 else 4
cfg = load_config(sys.argv[6] if len(sys.argv) > 6 else os.path.join(FPT, "config_003.json"))
cfg["nonlinear"].update(M=None, No=None, threads=th)
R = np.load(os.path.join(work, "preprocessed.npy"), mmap_mode="r")
cal = json.load(open(os.path.join(work, "calibration.json")))
kn = np.array(cal["k_solver"]); tw = cal.get("twin", 1)
H, Nc = R.shape[-1], 512
Rc = np.array(R[:, H // 2 - Nc // 2:H // 2 + Nc // 2, H // 2 - Nc // 2:H // 2 + Nc // 2])
ks = c * cfg["kappa_nom"] * tw
t0 = time.time()
res = nlr.run_nonlinear(Rc, kn, cfg, ks, cal, np.zeros((Nc, Nc)), np.zeros((Nc, Nc)), log=lambda m: None)
np.savez_compressed(out, **res)
print(out, "c", c, "misfit", round(float(res["hist"][-1]), 4), "M", int(res["M"]), "No", int(res["No"]), "time %.0f s" % (time.time() - t0))
