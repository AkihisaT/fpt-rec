
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, "fpt_pipeline")
from fptrecon.config import load_config
from fptrecon import nonlinear as nlr, wotf
idx=np.array([int(v) for v in sys.argv[1].split(",")]) if "," in sys.argv[1] else np.arange(*[int(v) for v in sys.argv[1].split(":")])
kc=float(sys.argv[2]); z=float(sys.argv[3]); c=float(sys.argv[4]); tag=sys.argv[5]; nj=int(sys.argv[6]) if len(sys.argv)>6 else 60
g=json.load(open("d32a/geom_direct.json"))
cfg=load_config("fpt_pipeline/config_32a_reg_bf03.json", dict(fzp_outer_zone_um=0.5/kc, nonlinear=dict(threads=4)))
R=np.load("fpt_pipeline/fpt_output_32a_reg_bf03/work/preprocessed.npy", mmap_mode="r")
cy,cx=520,500; h=256
Rc=np.array(R[idx][:, cy-h:cy+h, cx-h:cx+h])
xc=np.array(g["xc_um"])[idx]
kap=kc/g["R_field_um"]
kn=-kap*xc                   # solver frame, FOV kappa_solver=+c*kap  (x measured from the image centre; crop offset below)
off=np.array([(cy-500)*0.0319,(cx-500)*0.0319])
kn_crop=kn+ (c*kap)*off[None]     # local illumination at the crop centre under the FOV model
cal=dict(defocus_um=z, astig_um=[0,0])
a0=np.zeros((512,512),np.float32); p0=np.zeros((512,512),np.float32)
t0=time.time()
res=nlr.run_nonlinear(Rc, kn_crop if c else kn+kap*off[None]*0, cfg, c*kap, cal, a0, p0, n_pupil=10, n_joint=nj, log=print)
print("final", res["hist"][-1], "per-image", np.round(res["per_image"],3).tolist(), "time", time.time()-t0)
np.savez_compressed(f"d32a/caltest/nl_{tag}.npz", **res)
