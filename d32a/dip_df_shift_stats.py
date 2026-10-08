# Learned per-image shifts of DIP in the BF+DF runs (512 px): rms and number of dark-field images near the +-5 px bound.
# usage: python d32a/dip_df_shift_stats.py  -> d32a/dip_df_shift_stats.json
import numpy as np, json, os
out = {}
for ds in ("32adf", "32adfaff", "32adfaff2"):
    for tag in ("DIP_planewave", "DIP_FOV_c1"):
        f = f"dip_pipeline/dip_output_{ds}/{tag}_state.npz"
        if not os.path.exists(f): continue
        z = np.load(f, allow_pickle=True)
        sh = z["best_shifts_um"] / 0.0319; mx = np.abs(sh).max(1)
        b, d = slice(0, 60), slice(60, None)
        out[f"{ds}|{tag}"] = dict(it=int(z["best_it"]), bright_rms_px=np.sqrt((sh[b] ** 2).mean(0)).tolist(), dark_rms_px=np.sqrt((sh[d] ** 2).mean(0)).tolist(),
                                  dark_near_bound=int((mx[d] > 4.5).sum()), bright_near_bound=int((mx[b] > 4.5).sum()), n_dark=int(len(mx[d])))
        print(ds, tag, out[f"{ds}|{tag}"])
json.dump(out, open("d32a/dip_df_shift_stats.json", "w"), indent=1)
