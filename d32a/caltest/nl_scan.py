
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, "fpt_pipeline")
from fptrecon.config import load_config
from fptrecon import nonlinear as nlr
g=json.load(open("d32a/geom_direct.json"))
kc=2.0; z=float(sys.argv[1]); nj=int(sys.argv[2]); idx=np.arange(15,60)
cfg=load_config("fpt_pipeline/config_32a_reg_bf03.json", dict(fzp_outer_zone_um=0.25, nonlinear=dict(threads=8)))
R=np.load("fpt_pipeline/fpt_output_32a_reg_bf03/work/preprocessed.npy", mmap_mode="r")
cy,cx=520,500; h=256; Rc=np.array(R[idx][:, cy-h:cy+h, cx-h:cx+h])
kn0=-(kc/g["R_field_um"])*np.array(g["xc_um"])[idx]
out=[]
for sc in [float(v) for v in sys.argv[3].split(",")]:
    for dth in [float(v) for v in sys.argv[4].split(",")]:
        a=np.deg2rad(dth); Rm=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        kn=sc*kn0@Rm.T
        res=nlr.run_nonlinear(Rc, kn, cfg, 0.0, dict(defocus_um=z, astig_um=[0,0]), np.zeros((512,512),np.float32), np.zeros((512,512),np.float32), n_pupil=5, n_joint=nj, log=lambda m: None)
        out.append((sc,dth,float(res["hist"][-1]))); print(sc,dth,round(out[-1][2],4), flush=True)
json.dump(out, open(f"d32a/caltest/scan_z{int(z)}.json","w"))
