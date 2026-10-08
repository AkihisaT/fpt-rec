# -*- coding: utf-8 -*-
"""Configs for the 036 partial-coherence runs (rings 1 + 3, 28 images) from the Phase-C (coherent dark-mode) configs.
  036pc   : mixed state, S = (1-eta) Gauss(0.02 kc) + eta Cauchy(0.1 kc, 2), adaptive modes F
  036pc1m : coherent through the same code path (1 mode), same gain/offset handling (reference)
DIP: dark_offset profile (gain + additive offset per dark image), direct beam refreshed every 5 iterations (exact at each
evaluation), 300 iterations at 512 (Phase C: 600; selected 80), 150 at 1024 (as Phase C).
usage: python d036/pc/make_pc_configs.py <eta>"""
import json, sys, os
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
eta = float(sys.argv[1])
for ds, pc, blis in (("036pc", {"eta": eta, "modes": "F", "sigma": 0.02, "ell": 0.1, "beta": 2.0}, f"pc{eta:g}F"), ("036pc1m", {"eta": "1mode"}, "pc1mode")):
    prev = [{"label": f"BLIS-FPM planewave (rings 1+3, {ds})", "file": f"../d036/pc/blis/s+0_{blis}.npz"},
            {"label": f"BLIS-FPM FOV c1 (rings 1+3, {ds})", "file": f"../d036/pc/blis/s-1_{blis}.npz"}]
    for sfx in ("", "_it30", "_1024", "_1024_it30"):
        c = json.load(open(W + f"dip_pipeline/config_dip_036df{sfx}.json"))
        c["_comment"] = f"DIP, 036 partial coherence ({ds}): rings 1 + 3 (28 images), dark mode, " + ("mixed-state illumination" if ds == "036pc" else "1 mode (coherent reference)")
        c["output_dir"] = f"dip_output_{ds}{sfx}"
        d = c["dip"]; d["partial_coherence"] = pc; d["dark_offset"] = "profile"; d["pc_direct_every"] = 5
        if sfx == "": d["n_iter"] = 300
        c["previous_native"] = prev
        json.dump(c, open(W + f"dip_pipeline/config_dip_{ds}{sfx}.json", "w"), indent=1)
    for sfx in ("", "_1024"):
        c = json.load(open(W + f"epry_pipeline/config_epry_036df{sfx}.json"))
        c["_comment"] = f"EPRY, 036 partial coherence ({ds}): rings 1 + 3 (28 images), dark mode"
        c["output_dir"] = f"epry_output_{ds}{sfx}"; c["dip_results_dir"] = f"../dip_pipeline/dip_output_{ds}{sfx}"
        c["epry"]["partial_coherence"] = pc; c["previous_native"] = prev
        json.dump(c, open(W + f"epry_pipeline/config_epry_{ds}{sfx}.json", "w"), indent=1)
print("written for eta", eta)
