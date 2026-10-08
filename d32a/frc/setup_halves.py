# Split-half data sets for the 32a FRC: images alternate between half A and B along each scan ring (azimuthal interleave),
# the same bright-field halves in the BF-only and BF+DF sets. Writes the DIP / EPRY configs (FOV model c = 1, crop 512),
# d32a/frc/halves.json and d32a/frc/ex_{bf,df}{A,B}.txt (exclusion lists for blis_32a.py).  usage: python d32a/frc/setup_halves.py [tag]   (tag aff default; aff2 = focus-loop data sets)
import json, os, sys
import numpy as np
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
TAG = sys.argv[1] if len(sys.argv) > 1 else "aff"
ring_sizes = [15, 20, 25, 30, 35, 40, 45, 50]; start = np.cumsum([0] + ring_sizes[:-1])
orig_half = {}
for rs, st in zip(ring_sizes, start):
    for j in range(rs): orig_half[int(st + j)] = "A" if j % 2 == 0 else "B"
bf_idx = list(range(60))                                                            # fpt_output_32a_bf_aff/work = original 0..59
df_idx = json.load(open(W + f"d32a/work_df_{TAG}/calibration.json"))["positions_index"]  # work_df_aff: original indices
H = {}
for name, idx in (("bf", bf_idx), ("df", df_idx)):
    for h in "AB":
        H[f"{name}{h}"] = dict(use=[i for i, o in enumerate(idx) if orig_half[o] == h], exclude=[i for i, o in enumerate(idx) if orig_half[o] != h])
json.dump(H, open(W + ("d32a/frc/halves.json" if TAG == "aff" else f"d32a/frc/halves_{TAG}.json"), "w"), indent=0)
for name, base in (("bf", f"32abf{TAG}"), ("df", f"32adf{TAG}")):
    for h, seed in (("A", 0), ("B", 1)):
        ds = f"{base}H{h}"; ex = H[f"{name}{h}"]["exclude"]
        c = json.load(open(W + f"epry_pipeline/config_epry_{base}.json"))
        c["output_dir"] = f"epry_output_{ds}"; c["epry"]["exclude_images"] = sorted(set(c["epry"].get("exclude_images", [])) | set(ex))
        c["epry"]["fov_factors"] = [1.0]; c["_comment_half"] = f"split-half FRC data set {h} (azimuthal interleave along each ring), d32a/frc/setup_halves.py"
        json.dump(c, open(W + f"epry_pipeline/config_epry_{ds}.json", "w"), indent=1, ensure_ascii=False)
        for sfx in ("", "_it30"):
            c = json.load(open(W + f"dip_pipeline/config_dip_{base}{sfx}.json"))
            c["output_dir"] = f"dip_output_{ds}{sfx}"; c["dip"]["exclude_images"] = sorted(set(c["dip"].get("exclude_images", [])) | set(ex))
            c["dip"]["fov_factors"] = [1.0]; c["dip"]["seed"] = seed
            c["_comment_half"] = f"split-half FRC data set {h}, seed {seed} (different seeds for A and B), d32a/frc/setup_halves.py"
            json.dump(c, open(W + f"dip_pipeline/config_dip_{ds}{sfx}.json", "w"), indent=1, ensure_ascii=False)
for k, v in H.items():                                                              # exclusion lists for blis_32a.py --exclude
    open(W + f"d32a/frc/ex_{k}.txt", "w").write(",".join(str(i) for i in v["exclude"]))
print({k: (len(v["use"]), len(v["exclude"])) for k, v in H.items()})
