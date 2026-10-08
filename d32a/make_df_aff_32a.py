# -*- coding: utf-8 -*-
"""32a BF+DF work folder with the posaffine correction: d32a/work_df_aff
bright part (rings 0-2) = fpt_pipeline/fpt_output_32a_bf_aff/work/preprocessed.npy (posaffine-corrected);
dark part (rings 6-7) = d32a/work_df_nd (stage-rounding correction only) shifted by the linear mechanical drift
B_mech (px per um of lens shift, from results/position_affine.json) extrapolated to the dark-field positions, plus the
mechanical offset of the bright set; validity masks shifted identically. Defocus / astigmatism = contrast focus.
usage: python d32a/make_df_aff_32a.py [fpt_output_dir_name] [work_dir_name] [source_dir_name]
  (defaults fpt_output_32a_bf_aff, work_df_aff, work_df_nd if present else work_df = output of prep_32a_df.py;
   work_df_nd of the analysis differs from work_df only in the bright part and the defocus fields, which are replaced here)"""
import json, os, shutil
import numpy as np
from scipy import ndimage as ndi
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
import sys
BF = W + "fpt_pipeline/" + (sys.argv[1] if len(sys.argv) > 1 else "fpt_output_32a_bf_aff") + "/"; SRC = W + "d32a/" + (sys.argv[3] if len(sys.argv) > 3 else ("work_df_nd" if os.path.exists(W + "d32a/work_df_nd") else "work_df")) + "/"
DST = W + "d32a/" + (sys.argv[2] if len(sys.argv) > 2 else "work_df_aff") + "/"
os.makedirs(DST, exist_ok=True)
rep = json.load(open(BF + "results/position_affine.json")); calb = json.load(open(BF + "work/calibration.json"))
Bm = np.array(rep["B_mech_px_per_stage"]); corr_bf = np.array(rep["correction_px"])
pos_bf = np.load(BF + "work/preprocess_meta.npz")["positions"].astype(float)
c_m = (corr_bf - pos_bf @ Bm.T).mean(0)                          # offset of the linear mechanical model on the bright set
cal = json.load(open(SRC + "calibration.json")); idx = np.array(cal["positions_index"])
P = np.loadtxt(W + "d32a/data_reg/positions_um.csv", delimiter=","); pos = P[idx, 1:3]
R = np.load(SRC + "preprocessed.npy", mmap_mode="r"); Rb = np.load(BF + "work/preprocessed.npy", mmap_mode="r")
nb = Rb.shape[0]; assert np.allclose(pos[:nb], pos_bf), "bright-set order differs"
dark = np.array(cal["dark_images"]) if cal.get("dark_images") else np.arange(nb, len(idx))
corr = np.zeros((len(idx), 2)); corr[nb:] = pos[nb:] @ Bm.T + c_m
out = np.empty(R.shape, np.float32); out[:nb] = Rb
for i in range(nb, len(idx)):
    out[i] = ndi.shift(np.asarray(R[i], np.float32), -corr[i], order=3, mode="nearest")
np.save(DST + "preprocessed.npy", out); del out
mk = np.load(SRC + "dfmask.npy", mmap_mode="r")
mko = np.stack([np.clip(ndi.shift(np.asarray(mk[i], np.float32), -corr[i], order=1, mode="nearest"), 0, 1) if i >= nb else np.asarray(mk[i], np.float32)
                for i in range(mk.shape[0])]).astype(np.float16)
np.save(DST + "dfmask.npy", mko)
for f in ("noise.json", "prep_df.json"): shutil.copy2(SRC + f, DST + f)
cal["defocus_um"], cal["astig_um"] = calb["defocus_um"], calb["astig_um"]
cal["pupil_calibration_source"] = f"posaffine contrast focus ({os.path.basename(BF.rstrip('/'))})"
cal["position_affine"] = dict(calb["position_affine"], dark_correction="linear B_mech extrapolated to rings 6-7 + bright-set offset",
                              dark_correction_px=corr[nb:].tolist())
json.dump(cal, open(DST + "calibration.json", "w"), indent=1)
print(f"work_df_aff: {len(idx)} images ({len(idx) - nb} dark); dark correction rms {np.sqrt((corr[nb:] ** 2).mean(0)).round(1).tolist()} px, "
      f"max {np.abs(corr[nb:]).max(0).round(1).tolist()} px")
