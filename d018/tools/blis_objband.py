"""BLIS-FPM with the OBJECT spectrum constrained to |q| <= qmax (a and phi low-passed inside the forward model),
for the plane-wave (s = 0) and FOV (solver s = -1 -> physical c = +1 under twin -1) models of 018.
Same start (tile-WOTF linear solution) and iterations as the pipeline (n_pupil 20 + n_joint 150).
usage: python blis_objband.py <s> <qmax_um^-1> [threads] [excluded image indices, e.g. 0,9,18,27,36]
output: d018/objband/s<s>_q<qmax>.json (+ .npz with a, phi, W)
Environment overrides (other datasets): FPT_CFG = config json, FPT_WORK = fptrecon work dir, FPT_OUT = output dir (defaults: 018)."""
import json, os, sys, time
import numpy as np
import torch
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon.config import load_config
from fptrecon import nonlinear as nlr, wotf

s, qmax = float(sys.argv[1]), float(sys.argv[2])
cfg = load_config(os.environ.get("FPT_CFG", W + "fpt_pipeline/config_018.json"))
cfg["nonlinear"]["threads"] = int(sys.argv[3]) if len(sys.argv) > 3 else 4
work = os.environ.get("FPT_WORK", W + "fpt_pipeline/fpt_output_018/work/").rstrip("/") + "/"
R = np.asarray(np.load(work + "preprocessed.npy", mmap_mode="r")); cal = json.load(open(work + "calibration.json"))
kn = np.array(cal["k_solver"])
excl = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else []
use = np.array([i for i in range(len(kn)) if i not in excl])
R, kn = R[use], kn[use]
out = os.environ.get("FPT_OUT", W + "d018/objband/").rstrip("/") + "/"; os.makedirs(out, exist_ok=True)

_orig_init = nlr.FPMBand.__init__
def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    N = self.N; dxo = _init.dxo
    f = np.fft.fftfreq(N, dxo); q = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    self.lp = torch.tensor((q <= qmax).astype(np.float32))
    with torch.no_grad():                                   # start inside the constraint
        self.a.copy_(self._lp(self.a)); self.phi.copy_(self._lp(self.phi))
def _lp(self, x):
    return torch.fft.ifft2(torch.fft.fft2(x) * self.lp).real
def _normalised(self):
    P = self.P()
    Oe = torch.exp(-self._lp(self.a) + 1j * self._lp(self.phi)) * self.Q
    I = self._field(torch.fft.fftshift(torch.fft.fft2(Oe, norm="ortho")), P)
    if self.direct_norm:
        Id = self._field(self.Qhat, P)
        I = I / torch.clamp(Id, min=self.dir_floor * float(Id.detach().max()))
    return I / I.mean((1, 2), keepdim=True) - 1.0
nlr.FPMBand.__init__ = _init; nlr.FPMBand._lp = _lp; nlr.FPMBand.normalised = _normalised

# object pixel of the pipeline grid (needed for the mask) -- same computation as run_nonlinear
keep = np.where(np.hypot(kn[:, 0], kn[:, 1]) / cfg["kc"] <= cfg["nonlinear"]["max_k_over_kc"])[0]
dk = 1 / (R.shape[-1] * cfg["pixel_um"]); M, No = nlr.grid_sizes(cfg, kn[keep], dk)
_init.dxo = R.shape[-1] * cfg["pixel_um"] / No

t0 = time.time(); ks = s * cfg["kappa_nom"]
a0, p0, _ = wotf.tile_linear_recon(R, kn, cfg, ks, cal["defocus_um"], cal["astig_um"], log=lambda m: None)
r = nlr.run_nonlinear(R, kn, cfg, ks, cal, a0, p0, log=lambda m: None)
tag = f"s{s:+.0f}_q{qmax:.1f}" + ("_ho" if excl else "")
res = dict(s_solver=s, kappa_solver=ks, qmax=qmax, misfit=float(r["hist"][-1]), hist=[float(v) for v in r["hist"]],
           per_image=[float(v) for v in r["per_image"]], seconds=time.time() - t0, No=int(No), dxo_um=_init.dxo, excluded=excl)
json.dump(res, open(out + tag + ".json", "w"), indent=1)
np.savez_compressed(out + tag + ".npz", a=r["a"], phi=r["phi"], W=r["W"])
print(f"{tag}: misfit {res['misfit']:.4f} ({res['seconds']:.0f} s)", flush=True)
