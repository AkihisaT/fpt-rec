# -*- coding: utf-8 -*-
"""036 partial coherence, step 2: forward prediction of all 44 images with the object and pupil of a coherent BLIS-FPM
solution (default: rings 1 + 3, fitted without images 0, 25, 34, 43 -> ring 2 and those four are unseen), for a set of
candidate illumination angular distributions S (mixed state, polar mode grid up to 1.2 kc).
Per-image band misfit (0.3-3.5 um^-1, window x dark-field mask, window-weighted mean) as in blis_df.py; 1 = no prediction.
usage: python d036/pc/predict_pc.py [solution.npz] -> d036/pc/predict_pc_gain.json (dark images with a profiled contrast factor)"""
import json, sys, time, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import pc_common as pcm
sys.path.insert(0, pcm.W + "fpt_pipeline")
from fptrecon import coherence as coh
torch.set_num_threads(6)
GAIN = "profile"   # per dark image least-squares contrast factor (as DIP dark_gain profile)
solf = sys.argv[1] if len(sys.argv) > 1 else pcm.W + "d036/blis_df13/s-1_ho.npz"
sol = np.load(solf); kc = pcm.kc
idx = np.arange(44); fitted = set(np.asarray(sol["keep"]).tolist())
edges = np.array([0, 0.03, 0.07, 0.12, 0.2, 0.3, 0.45, 0.65, 0.9, 1.2])       # kc
naz = [1, 6, 8, 12, 16, 20, 24, 28, 32]
Sf = json.load(open(pcm.W + "d036/pc/S_fit.json"))
cands = {"coherent": None,
         "halo eta 0.1": coh.cauchy_core_fractions(edges, 0.1, 0.02, 0.1, 2.0),
         "halo eta 0.3": coh.cauchy_core_fractions(edges, 0.3, 0.02, 0.1, 2.0),
         "halo eta 0.6": coh.cauchy_core_fractions(edges, 0.6, 0.02, 0.1, 2.0),
         "direct-image S (NNLS)": coh.annulus_fractions(np.array(Sf["rho"][1:]) , np.array(Sf["S_annulus_fraction"][1:]) , edges)}
# the NNLS S has its first annulus at rho=0 (fraction S[0]); add it to the central disc
if "direct-image S (NNLS)" in cands:
    f = cands["direct-image S (NNLS)"]; f[0] += Sf["S_annulus_fraction"][0]; cands["direct-image S (NNLS)"] = f / f.sum()
out = dict(solution=solf, fitted_images=sorted(fitted), edges_kc=edges.tolist(), n_az=naz, results={})
t0 = time.time()
for name, frac in cands.items():
    if frac is None:
        fb = pcm.build(sol, idx, modes_k=np.zeros((1, 2)), mode_w=np.ones(1), dark_gain=GAIN)
    else:
        dk, w = coh.modes_from_radial(edges * kc, frac, naz)
        fb = pcm.build(sol, idx, modes_k=dk, mode_w=w, dark_gain=GAIN)
    per = fb.per_image_loss()
    r = pcm.ring
    grp = {"ring1 fitted": [i for i in idx if r[i] == 1 and i in fitted], "ring1 unseen": [i for i in idx if r[i] == 1 and i not in fitted],
           "ring2 unseen": [i for i in idx if r[i] == 2], "ring3 fitted": [i for i in idx if r[i] == 3 and i in fitted],
           "ring3 unseen": [i for i in idx if r[i] == 3 and i not in fitted]}
    res = {g: float(np.mean(per[v])) for g, v in grp.items() if v}
    out["results"][name] = dict(per_ring=res, per_image=per.tolist(), dark_gains=fb.dark_gains().tolist(), fractions=None if frac is None else list(map(float, frac)))
    print(f"[{time.time() - t0:6.0f}s] {name:24s} " + "  ".join(f"{g} {v:.3f}" for g, v in res.items()), flush=True)
    del fb
json.dump(out, open(pcm.W + "d036/pc/predict_pc_gain.json", "w"), indent=1)
