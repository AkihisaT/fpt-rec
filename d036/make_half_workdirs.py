# -*- coding: utf-8 -*-
"""Half-data work folders for the FRC of EPRY (036): work_bfA/B (ring-1 images of half A/B + Phase-B calibration) and
work_dfA/B (44-image dark-mode data of half A/B + Phase-C calibration, noise and masks).
Needs d036/work_df/ (prep_036_df.py, noise_df.py) and fpt_pipeline/fpt_output_036bf/work/calibration.json.
usage: cd d036 && python make_half_workdirs.py
(Same operations as the inline step used in the 036 analysis.)"""
import numpy as np, os, shutil
for h in "AB":
    A = np.load(f"work_df/preprocessed_{h}.npy", mmap_mode="r")
    d = f"work_bf{h}/"; os.makedirs(d, exist_ok=True)
    np.save(d + "preprocessed.npy", np.asarray(A[:9]))
    shutil.copy("../fpt_pipeline/fpt_output_036bf/work/calibration.json", d)
    d = f"work_df{h}/"; os.makedirs(d, exist_ok=True)
    for f in ("calibration.json", "noise.json"):
        shutil.copy("work_df/" + f, d)
    for f in ("dfmask.npy",):
        if not os.path.exists(d + f): os.symlink(os.path.abspath("work_df/" + f), d + f)
    if not os.path.exists(d + "preprocessed.npy"): os.symlink(os.path.abspath(f"work_df/preprocessed_{h}.npy"), d + "preprocessed.npy")
print(os.listdir("work_dfA"), os.listdir("work_bfA"))
