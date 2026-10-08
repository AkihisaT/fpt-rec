# Is the object autofocus (posaffine.autofocus) unbiased for EPRY reconstructions of 32a?
# Synthetic data: an 'in focus' single-material object (a = g phi, from the EPRY FOV phase), the calibrated pupil
# (contrast focus -1.52 mm), the same illuminations / FOV model; EPRY with the true pupil; autofocus of the result.
import os, sys, json, time
import numpy as np, torch
W = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) + "/"
os.chdir(W + "epry_pipeline"); sys.path.insert(0, W + "epry_pipeline"); sys.path.insert(0, W + "fpt_pipeline")
import run_epry as RE
cfg = RE.load_cfg("config_epry_32abfaff.json", {"fov_factors": [1.0]})
sys.path.insert(0, cfg["dip_pipeline_dir"])
from dipfpm import train as T
from dipfpm.physics import FPMPhysics
from eprfpm.core import EPRY, run_epry
import fptrecon.posaffine as PA
log = lambda m: print(m, flush=True)
data = T.prepare(cfg, log); e = cfg["epry"]; lam = cfg["lam_um"]
st = np.load("epry_output_32abfaff/EPRY_FOV_c1_state.npz"); dxo = float(st["dxo"]); ks = float(st["kappa_solver"])
phi_r = st["best_phi"].astype(float); a_r = st["best_a"].astype(float)
af_real = PA.autofocus(a_r, phi_r, dxo, lam, band=(0.3, 3.5), frac=0.35); log(f"real EPRY object: {af_real}")
g = float(((a_r - a_r.mean()) * (phi_r - phi_r.mean())).sum() / ((phi_r - phi_r.mean()) ** 2).sum())
phi_t = phi_r - phi_r.mean(); a_t = g * phi_t
af_true = PA.autofocus(a_t, phi_t, dxo, lam, band=(0.3, 3.5), frac=0.35); log(f"truth (a = {g:.3f} phi): {af_true}")
keep = list(data["keep"]); kk = data["kn"][data["keep"]]
phys = FPMPhysics(kk, data["dk"], data["M"], data["No"], cfg["kc"], lam, kappa=ks, axis_offset_um=data["axis_off"], pupil_radius=e["pupil_radius"])
with torch.no_grad():
    I = phys.forward(torch.tensor(a_t, dtype=torch.float32), torch.tensor(phi_t, dtype=torch.float32), torch.tensor(data["z0"], dtype=torch.float32),
                     torch.zeros(len(keep), 2)).numpy().astype(np.float64)
I = I / I.mean(axis=(1, 2), keepdims=True)
out = {"real_epry": af_real, "truth": af_true, "g": g}
for noise in (0.0,):
    Im = I.copy()
    tr = [keep.index(k) for k in data["train"]]; va = [keep.index(k) for k in data["val"]]
    kabs = np.hypot(*kk.T)
    P0, W0 = RE.initial_pupil(data["z0"], data["M"], data["dk"], cfg["kc"], e["pupil_radius"] * cfg["kc"], 0.25 * data["dk"])
    eng = EPRY(Im.astype(np.float32), kk, data["dk"], data["M"], data["No"], cfg["kc"], P0, kappa=ks, axis_offset_um=data["axis_off"],
               pupil_radius=e["pupil_radius"], replace_tukey=e["replace_tukey"], loss_tukey=e["loss_tukey"], band=cfg["export_band"],
               threads=4, mask_vignetted=bool(e.get("mask_vignetted", False)), pupil_gamma=float(e.get("pupil_gamma", 0.0)), wmask=data.get("wmask"))
    t0 = time.time(); res = run_epry(eng, tr, va, e, kabs, log=lambda m: None)
    for grp in ("best", "final"):
        af = PA.autofocus(np.asarray(res[grp]["a"], float), np.asarray(res[grp]["phi"], float), dxo, lam, band=(0.3, 3.5), frac=0.35)
        out[f"epry_synthetic_{grp}"] = dict(sweep=int(res[grp]["it"]), **af)
        log(f"EPRY on synthetic data ({grp}, sweep {res[grp]['it']}): {af}  ({time.time() - t0:.0f} s)")
json.dump(out, open(W + "d32a/af_test/af_bias_test.json", "w"), indent=1, default=float)
