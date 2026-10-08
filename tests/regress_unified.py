"""RECORD ONLY: run once (2026-09-28) in the analysis workspace that still held the two original versions; it cannot run
from this repository alone. Results: tests/regress_unified.json (T1-T6 max difference 0.0). For checks on a new computer use
tests/selftest.py and tests/check_32a.sh.

Regression checks of the unified package against the two versions it was merged from.
 T1 fptrecon.nonlinear, default arguments           : unified == 036 package (coherent) and == 32a pipeline
 T2 fptrecon.nonlinear, fix_pupil=True (focus loop) : unified == 32a pipeline (posaffine version)
 T3 fptrecon.nonlinear, free_shifts=True            : unified == 32a pipeline
 T4 FPMBand with partial-coherence modes            : unified == 036 package
Input: 32a bright-field stack (every 3rd image), short L-BFGS runs; T4 on random data. usage (workspace root): python unify/tests/regress_unified.py"""
import importlib.util, json, os, sys
import numpy as np, torch
W = os.getcwd() + "/"
U = W + "unify/FPT統合パイプライン_20260928/fpt_pipeline/"
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
sys.path.insert(0, U)
import fptrecon
from fptrecon.config import load_config
V = dict(unified=load("fptrecon.nl_unified", U + "fptrecon/nonlinear.py"),
         pc036=load("fptrecon.nl_036", W + "unify/z036/FPT036_暗視野対応パイプライン_20260926/fpt_pipeline/fptrecon/nonlinear.py"),
         a32=load("fptrecon.nl_32a", W + "unify/merge/ours.py"))
torch.set_num_threads(4)
cfg = load_config(W + "pipe/fpt_pipeline/config_32a_bf_aff2.json"); cfg["nonlinear"]["threads"] = 4
wd = W + "pipe/fpt_pipeline/fpt_output_32a_bf_aff2/work/"
cal = json.load(open(wd + "calibration.json")); kn = np.array(cal["k_solver"])[::3]
R = np.asarray(np.load(wd + "preprocessed.npy", mmap_mode="r")[::3]); z0 = np.zeros((64, 64), np.float32)
ks = cfg["kappa_nom"] * int(cal.get("twin", 1))
def run(v, **kw):
    torch.manual_seed(0); np.random.seed(0)
    return V[v].run_nonlinear(R, kn, cfg, ks, cal, z0, z0, n_pupil=3, n_joint=5, log=lambda m: None, **kw)
def same(r1, r2, keys=("a", "phi", "W", "Imod", "hist")):
    return {k: float(np.nanmax(np.abs(np.asarray(r1[k], float) - np.asarray(r2[k], float)))) for k in keys}
out = {}
ru = run("unified")
out["T1_default_vs_036"] = same(ru, run("pc036")); out["T1_default_vs_32a"] = same(ru, run("a32"))
out["T2_fix_pupil_vs_32a"] = same(run("unified", fix_pupil=True), run("a32", fix_pupil=True))
out["T3_free_shifts_vs_32a"] = same(run("unified", free_shifts=True), run("a32", free_shifts=True), keys=("a", "phi", "W", "hist", "shifts_px"))
M, N = 96, 160; n = 6; rng = np.random.default_rng(0)
I = (1 + 0.05 * rng.standard_normal((n, M, M))).astype(np.float32)
s_px = np.stack([np.round(8 * np.cos(np.arange(n))), np.round(8 * np.sin(np.arange(n)))], 1)
modes = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], float); mw = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
res = {}
for v in ("unified", "pc036"):
    torch.manual_seed(0)
    fb = V[v].FPMBand(I, s_px, N, M, 20.0, (3.0, 30.0), modes=modes, mode_w=mw)
    fb.run(4, ("obj", "pupil_phase"), log=lambda m: None)
    res[v] = dict(a=fb.a.detach().numpy(), phi=fb.phi.detach().numpy(), W=fb.W.detach().numpy(), hist=np.array(fb.hist))
out["T4_partial_coherence_vs_036"] = same(res["unified"], res["pc036"], keys=("a", "phi", "W", "hist"))
out["ok"] = all(max(v.values()) == 0.0 for k, v in out.items() if k.startswith("T"))
json.dump(out, open(W + "unify/tests/regress_unified.json", "w"), indent=1)
print(json.dumps(out))
