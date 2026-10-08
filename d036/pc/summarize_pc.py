# -*- coding: utf-8 -*-
"""036 partial coherence: collect every number used in the report / slides into d036/pc/pc_summary.json.
Per method and model: ring-1 / ring-3 misfit of fitted and held-out images (BLIS-FPM from its hold-out solution),
spoke SNR>3 limit (1024), for the mixed-state set (pc), the 1-mode reference (pc1m) and the original coherent Phase-C
solutions re-scored with the same metric (dfgain).  Plus: direct-image analysis, S candidates, prediction test, eta selection,
synthetic test, BLIS-FPM with ring 2 (44 images) and bright field only (9 images).
usage: python d036/pc/summarize_pc.py"""
import json, os, glob, numpy as np
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
P = W + "d036/pc/"
J = lambda f: json.load(open(f)) if os.path.exists(f) else None
cal = J(W + "d036/work_df/calibration.json"); ring = np.array(cal["ring"])
M = ["DIP (optimal it.)", "DIP (30 it.)", "EPRY", "BLIS-FPM"]; MOD = ["plane-wave model", "FOV model"]
def per_ring(cs, m, mod):
    v = cs["results"].get(f"{m}, {mod}")
    if v is None: return None
    tr, va = cs["image_index_train"], cs["image_index_validation"]
    pt, pv = np.array(v["band_misfit_training_per_image"]), np.array(v["band_misfit_validation_per_image"])
    o = {}
    for R in (1, 3):
        a = pt[[ring[i] == R for i in tr]]; b = pv[[ring[i] == R for i in va]]
        o[f"fit{R}"] = float(a.mean()) if len(a) else None; o[f"ho{R}"] = float(b.mean()) if len(b) else None
    o["ho_all"] = float(v["band_misfit_validation_images"]); o["fit_all"] = float(v["band_misfit_training_images"])
    o["iteration"] = v.get("iteration"); o["spoke_snr3_um_inv"] = v.get("spoke_snr3_um_inv")
    return o
out = dict(tables={})
for G in (512, 1024):
    for S in ("pc", "pc1m", "dfgain"):
        cs, ch = J(P + f"out_{G}_{S}/compare_summary.json"), J(P + f"out_{G}_{S}_ho/compare_summary.json")
        if cs is None: continue
        tab = {}
        for m in M:
            for mod in MOD:
                src = ch if (m == "BLIS-FPM" and ch is not None) else cs
                r = per_ring(src, m, mod)
                if r is None: continue
                full = per_ring(cs, m, mod)
                r["spoke_snr3_um_inv"] = full["spoke_snr3_um_inv"] if full else None      # images / spokes from the full solution
                tab[f"{m}, {mod}"] = r
        out["tables"][f"{G}_{S}"] = dict(images_train=cs["images_train"], images_validation=cs["images_validation"],
                                         image_index_validation=cs["image_index_validation"], results=tab)
# BLIS-FPM runs (full field) in d036/pc/blis
def bj(f):
    j = J(f)
    if j is None: return None
    return dict(misfit=j["misfit"], per_ring={k: (None if (v is None or v != v) else v) for k, v in j["mean_per_ring"].items()},
                heldout_images=j.get("heldout_images"), heldout_per_image=j.get("heldout_per_image"), n_modes=j.get("n_modes_mean"), seconds=j.get("seconds"))
out["blis"] = {os.path.relpath(f, P + "blis"): bj(f) for f in sorted(glob.glob(P + "blis/**/*.json", recursive=True))}
out["direct_fit"] = {k: v for k, v in J(P + "direct_fit.json").items() if k in ("best", "rms_without_k0", "coarse")}
out["S_nnls"] = {k: v for k, v in J(P + "S_fit.json").items() if k in ("r_p", "level_A", "rms_log", "encircled")}
out["prediction_test"] = {k: v["per_ring"] for k, v in J(P + "predict_pc_gain.json")["results"].items()}
out["prediction_test_nogain"] = {k: v["per_ring"] for k, v in J(P + "predict_pc.json")["results"].items()}
out["modes_convergence"] = {k: dict(J=v["J"]) for k, v in J(P + "modes_convergence.json")["sets"].items()}
ma = J(P + "modes_adaptive_eta0.3.json")
out["modes_adaptive"] = {k: dict(J_per_ring=v["J_per_ring"], rel_diff_median={R: float(np.median(np.array(v["rel_diff"])[ring == R])) for R in (1, 2, 3)})
                         for k, v in ma["results"].items()}
out["synthetic_test"] = J(P + "synthetic_test.json")
sel = {}
for t in ("1mode", "0.1F", "0.3F"):
    j = J(P + f"blis/s-1_ho_pc{t}.json"); sel[t] = dict(misfit=j["misfit"], heldout_images=j["heldout_images"], heldout_per_image=j["heldout_per_image"],
                                                       per_ring=j["mean_per_ring"])
out["eta_selection"] = sel; out["eta"] = 0.1
out["S_model"] = dict(core_sigma_kc=0.02, halo_ell_kc=0.1, halo_beta=2.0, eta=0.1, modes="adaptive F (core 0.08 kc individually, 8 in-pupil / 2 out-of-pupil sectors)")
json.dump(out, open(P + "pc_summary.json", "w"), indent=1, default=float)
print("tables:", list(out["tables"]), "| blis runs:", len(out["blis"]))
