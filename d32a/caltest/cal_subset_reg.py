
import json, os, sys, numpy as np
sys.path.insert(0, "fpt_pipeline")
from fptrecon.config import load_config
from fptrecon import wotf
dr=float(sys.argv[1]); rings=[int(v) for v in sys.argv[2].split(",")]; crop=int(sys.argv[3]); tag=sys.argv[4]
cfg=load_config("fpt_pipeline/config_32a_reg_bf03.json", dict(fzp_outer_zone_um=dr, calib=dict(crop_px=crop, crop_center=[520,500])))
R=np.load("fpt_pipeline/fpt_output_32a_reg_bf03/work/preprocessed.npy", mmap_mode="r")
meta=dict(np.load("fpt_pipeline/fpt_output_32a_reg_bf03/work/preprocess_meta.npz"))
ring=np.repeat(np.arange(4),[15,20,25,30])
idx=np.where(np.isin(ring,rings))[0]
logf=open(f"d32a/caltest/{tag}.log","w")
def log(m): print(m, flush=True); logf.write(m+"\n"); logf.flush()
cal=wotf.calibrate(np.asarray(R[idx]), cfg, meta["positions"][idx], log)
cal.update(dr=dr, rings=rings, crop=crop, kc=cfg["kc"], idx=idx.tolist())
json.dump(cal, open(f"d32a/caltest/{tag}.json","w"), indent=1)
