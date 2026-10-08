# -*- coding: utf-8 -*-
"""Accuracy of per-image adaptive mode sets (fptrecon.coherence.adaptive_modes) against the 147-mode reference
(modes_convergence.json models are recomputed here for the reference).  S = core 0.02 kc + Cauchy halo (eta, 0.1 kc, 2)."""
import json, sys, time, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import pc_common as pcm
sys.path.insert(0, pcm.W + "fpt_pipeline")
from fptrecon import coherence as coh
torch.set_num_threads(6)
eta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
sol = np.load(pcm.W + "d036/blis_df13/s-1_ho.npz"); kc = pcm.kc; idx = np.arange(44); kn = pcm.kn_all[idx]
e = np.array([0, .03, .07, .12, .2, .3, .45, .65, .9, 1.2]); na = [1, 6, 8, 12, 16, 20, 24, 28, 32]
dref, wref = coh.modes_from_radial(e * kc, coh.cauchy_core_fractions(e, eta, 0.02, 0.1, 2.0), na)
def model(dk, w):
    fb = pcm.build(sol, idx, modes_k=dk if dk.ndim == 2 else dk, mode_w=w, dark_gain="profile",
                   No=int(sol["No"]) + 2 * (int(np.ceil(1.25 * kc / float(sol["dk"]))) + 2))
    with torch.no_grad():
        F = torch.fft.fft2(fb.normalised() * fb.winn)[:, fb.band].numpy()
    return F, fb.per_image_loss()
t0 = time.time(); Fr, pr = model(dref, wref); tr = time.time() - t0
print(f"reference J=147: {tr:.0f}s, misfit r1/r2/r3 " + "/".join(f"{pr[pcm.ring == R].mean():.3f}" for R in (1, 2, 3)), flush=True)
cfgs = {"A core.13 in8 out4": dict(core=0.13, sec_in=8, sec_out=4),
        "B core.13 in12 out4": dict(core=0.13, sec_in=12, sec_out=4),
        "C core.08 in8 out4": dict(core=0.08, sec_in=8, sec_out=4),
        "D core.13 in8 out2": dict(core=0.13, sec_in=8, sec_out=2),
        "E core.13 in16 out6 fine radial": dict(core=0.13, sec_in=16, sec_out=6, rad_edges=(0.13, 0.22, 0.35, 0.5, 0.7, 1.0, 1.3))}
res = {}
for name, kw in cfgs.items():
    dk, w = coh.adaptive_modes(kn, dref, wref, kc, **kw); Jn = (w > 0).sum(1)
    t0 = time.time(); F, p = model(dk, w); t = time.time() - t0
    d = np.sum(np.abs(F - Fr) ** 2, 1) / np.sum(np.abs(Fr) ** 2, 1)
    res[name] = dict(J_per_ring={R: float(Jn[pcm.ring == R].mean()) for R in (1, 2, 3)}, seconds=t, rel_diff=d.tolist(), misfit=p.tolist())
    print(f"{name:32s} J r1/r2/r3 {Jn[pcm.ring == 1].mean():.0f}/{Jn[pcm.ring == 2].mean():.0f}/{Jn[pcm.ring == 3].mean():.0f}  {t:4.0f}s  rel.diff median/max "
          + "  ".join(f"r{R} {np.median(d[pcm.ring == R]):.4f}/{d[pcm.ring == R].max():.4f}" for R in (1, 2, 3)), flush=True)
json.dump(dict(eta=eta, reference_seconds=tr, results=res), open(pcm.W + f"d036/pc/modes_adaptive_eta{eta:g}.json", "w"), indent=1)
