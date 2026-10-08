"""BLIS-FPM FOV-strength scan for 018: solver-frame kappa = s * kappa_nom, 70 iterations (n_pupil 20 + n_joint 50),
each started from its own tile-WOTF linear solution (as in the pipeline).  Physical c = twin * s.
usage: python kappa_scan.py <s1,s2,...> [threads]
Environment overrides (other datasets): FPT_CFG = config json, FPT_WORK = fptrecon work dir, FPT_OUT = output dir (defaults: 018)."""
import json, os, sys, time
import numpy as np
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon.config import load_config
from fptrecon import nonlinear as nlr, wotf
cfg = load_config(os.environ.get("FPT_CFG", W + "fpt_pipeline/config_018.json"))
cfg["nonlinear"]["threads"] = int(sys.argv[2]) if len(sys.argv) > 2 else 4
work = os.environ.get("FPT_WORK", W + "fpt_pipeline/fpt_output_018/work/").rstrip("/") + "/"
R = np.asarray(np.load(work + "preprocessed.npy", mmap_mode="r")); cal = json.load(open(work + "calibration.json"))
kn = np.array(cal["k_solver"]); out = os.environ.get("FPT_OUT", W + "d018/kappa_scan/").rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
res_all = json.load(open(out + "scan.json")) if os.path.exists(out + "scan.json") else {}
for s in [float(v) for v in sys.argv[1].split(",")]:
    t0 = time.time(); ks = s * cfg["kappa_nom"]
    a0, p0, _ = wotf.tile_linear_recon(R, kn, cfg, ks, cal["defocus_um"], cal["astig_um"], log=lambda m: None)
    r = nlr.run_nonlinear(R, kn, cfg, ks, cal, a0, p0, n_pupil=20, n_joint=50, log=lambda m: None)
    res_all[f"{s:+.2f}"] = dict(s_solver=s, kappa_solver=ks, misfit=float(r["hist"][-1]), hist=[float(v) for v in r["hist"]])
    json.dump(res_all, open(out + "scan.json", "w"), indent=1)
    print(f"s={s:+.2f}: misfit {r['hist'][-1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
