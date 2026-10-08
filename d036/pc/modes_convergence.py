# -*- coding: utf-8 -*-
"""036 partial coherence: how many illumination modes are needed.  For S = core(0.02 kc) + Cauchy halo (eta, ell 0.1 kc,
beta 2), compare the normalised model images (band 0.3-3.5, window x mask) of reduced polar mode sets with the 147-mode
reference: relative difference per image, and the per-image misfit.  Object/pupil: BLIS-FPM rings 1+3 hold-out solution.
usage: python d036/pc/modes_convergence.py [eta] -> d036/pc/modes_convergence.json"""
import json, sys, time, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import pc_common as pcm
sys.path.insert(0, pcm.W + "fpt_pipeline")
from fptrecon import coherence as coh
torch.set_num_threads(6)
eta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
sol = np.load(pcm.W + "d036/blis_df13/s-1_ho.npz"); kc = pcm.kc; idx = np.arange(44)
sets = {"R147": ([0, .03, .07, .12, .2, .3, .45, .65, .9, 1.2], [1, 6, 8, 12, 16, 20, 24, 28, 32]),
        "R75": ([0, .03, .07, .12, .2, .3, .45, .65, .9, 1.2], [1, 6, 6, 8, 8, 10, 12, 12, 12]),
        "R42": ([0, .03, .07, .12, .2, .3, .45, .65, .9, 1.2], [1, 3, 4, 4, 6, 6, 6, 6, 6]),
        "R33": ([0, .05, .12, .25, .45, .8, 1.2], [1, 4, 6, 6, 8, 8]),
        "R23": ([0, .05, .15, .35, .7, 1.2], [1, 4, 6, 6, 6]),
        "R13": ([0, .06, .2, .5, 1.2], [1, 4, 4, 4])}
models, out = {}, dict(eta=eta, sets={})
t0 = time.time()
for name, (e, na) in sets.items():
    e = np.array(e); frac = coh.cauchy_core_fractions(e, eta, 0.02, 0.1, 2.0)
    dk, w = coh.modes_from_radial(e * kc, frac, na)
    fb = pcm.build(sol, idx, modes_k=dk, mode_w=w, dark_gain="profile")
    with torch.no_grad():
        F = torch.fft.fft2(fb.normalised() * fb.winn)[:, fb.band].numpy()
    models[name] = F; per = fb.per_image_loss()
    out["sets"][name] = dict(J=len(w), per_image=per.tolist(), seconds=time.time() - t0)
    print(f"[{time.time() - t0:5.0f}s] {name}: J={len(w)}", flush=True); del fb
ref = models["R147"]
for name in sets:
    d = np.sum(np.abs(models[name] - ref) ** 2, 1) / np.sum(np.abs(ref) ** 2, 1)
    out["sets"][name]["rel_diff_vs_R147"] = d.tolist()
    per = np.array(out["sets"][name]["per_image"])
    print(name, "J", out["sets"][name]["J"], " rel. model difference median/max  ring1 %.4f/%.4f  ring2 %.4f/%.4f  ring3 %.4f/%.4f" %
          tuple(v for R in (1, 2, 3) for v in (np.median(d[pcm.ring == R]), np.max(d[pcm.ring == R]))),
          " misfit r1/r2/r3 %.3f/%.3f/%.3f" % tuple(per[pcm.ring == R].mean() for R in (1, 2, 3)))
json.dump(out, open(pcm.W + "d036/pc/modes_convergence.json", "w"), indent=1)
