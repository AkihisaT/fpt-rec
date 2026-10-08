"""EPRY with dark-field images (036, 512 crop, FOV model c = 1 unless given): per-ring band misfit per sweep.
Variants (all use the dark-mode data of d036/work_df, 44 images evaluated, held-out = every 9th image):
  collapse     all 44 images updated, intensity factors re-estimated every sweep (original rule)
  firstvisit   all 44 images updated, dark-field factors estimated once at the first visit (eprfpm default in dark mode)
  brightonly   only ring-1 images updated (dark-field images predicted, not used)
  r13          rings 1 + 3 updated (ring 2 not used), first-visit factors
  r13fix       rings 1 + 3 updated, ring-3 factor fixed to 0.8 (ring-2 direct-beam level relative to ring 1)
usage: python epry_df_diag.py <variant> [c]    output: d036/epry_diag/<variant>.json"""
import os, sys, json
import numpy as np
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
os.chdir(W + "epry_pipeline"); sys.path.insert(0, "."); sys.path.insert(0, "../dip_pipeline")
import run_epry as RE
from eprfpm.core import EPRY
from dipfpm import train as T
var = sys.argv[1]; c = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
cfg = RE.load_cfg("config_epry_036df.json", {}); e = cfg["epry"]
cfg["dip"]["exclude_images"] = []                    # evaluate all 44 images
data = T.prepare(cfg, lambda m: None)
keep = list(data["keep"]); tr = [keep.index(k) for k in data["train"]]; va = [keep.index(k) for k in data["val"]]
ring = np.array(json.load(open(cfg["fpt_work_dir"] + "/calibration.json"))["ring"])[data["keep"]]
kabs = np.hypot(*data["kn"][data["keep"]].T); ks = c * cfg["kappa_nom"] * data["twin"]
P0, W0 = RE.initial_pupil(data["z0"], data["M"], data["dk"], cfg["kc"], e["pupil_radius"] * cfg["kc"], 0.25 * data["dk"])
eng = EPRY(data["I_meas"], data["kn"][data["keep"]], data["dk"], data["M"], data["No"], cfg["kc"], P0, kappa=ks,
           axis_offset_um=data["axis_off"], pupil_radius=e["pupil_radius"], replace_tukey=e["replace_tukey"],
           loss_tukey=e["loss_tukey"], band=cfg["export_band"], threads=2, wmask=data.get("wmask"))
use_rings = {"brightonly": (1,), "r13": (1, 3), "r13fix": (1, 3)}.get(var, (1, 2, 3))
upd = [n for n in tr if ring[n] in use_rings]
order = [int(n) for n in np.argsort(kabs) if int(n) in set(upd)]
if var == "collapse":
    eng.dark_gain_first_visit = False
if var == "r13fix":
    eng.visited[ring == 3] = True; eng.g[ring == 3] = 0.8
alpha, ep, hist = 1.0, None, []
for it in range(30):
    err = eng.sweep(order, alpha, alpha, update_pupil=it >= 1)
    eng.update_direct(); eng.update_gain(tr); eng.update_gain(tr + va)
    if var == "r13fix": eng.g[ring == 3] = 0.8
    bp = eng.band_per_image()
    row = dict(sweep=it, err=float(err), alpha=alpha)
    for r in (1, 2, 3):
        row[f"ring{r}_fit"] = float(bp[[n for n in tr if ring[n] == r]].mean())
        row[f"ring{r}_heldout"] = float(bp[[n for n in va if ring[n] == r]].mean())
        row[f"g_ring{r}"] = float(eng.g[ring == r].mean())
    hist.append(row)
    print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
    if ep is not None and (ep - err) / ep < 0.01: alpha *= 0.5
    ep = err
    if alpha < 0.002: break
os.makedirs(W + "d036/epry_diag", exist_ok=True)
json.dump(dict(variant=var, c=c, images_heldout=[int(keep[i]) for i in va], hist=hist, doc=__doc__),
          open(W + f"d036/epry_diag/{var}.json", "w"), indent=1)
