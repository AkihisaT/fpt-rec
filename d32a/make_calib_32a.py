"""32a: pipeline calibration.json from the direct-image vignetting geometry (the WOTF calibration of fptrecon does not
converge on these data: joint weak-object misfit ~0.8 for any rotation/scale, see report).

Geometry (d32a/geom_direct_v2.json, fitted on the registered direct frames of rings 2-5): the region of the field where the
direct beam passes the objective pupil is the disc |x - x_c,n| < R_field with x_c,n = A d_n + c0 (d_n = FZP position, um).
In the forward model of the pipelines (solver frame, O_eff = O exp(i pi kappa |x|^2), pupil P(k), psi = F^-1{P(k) O_eff^(k - k_n)})
the direct beam at x passes if |k_n + kappa x| < kc, i.e. x_c = -k_n/kappa and R_field = kc/kappa. Hence
    k_n / kc = -(A d_n + c0) / R_field ,
independent of the unmeasured p. Physical sign: kappa > 0 (object-side quadratic phase of the objective at distance p), so this
solver frame is the physical frame (twin = +1; checked by corr(a, phi) < 0 of the reconstructions).
usage: python make_calib_32a.py <work_dir> <dr_um> <defocus_um> <comma list of 0-based position indices or a:b>"""
import json, os, sys
import numpy as np
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
work, dr, z0, sel = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
idx = np.arange(*[int(v) for v in sel.split(":")]) if ":" in sel else np.array([int(v) for v in sel.split(",")])
g = json.load(open(W + "d32a/geom_direct_v2.json"))
A, c0, RF = np.array(g["A"]), np.array(g["c0_um"]), g["R_field_um"]
lam = 1.239842 / 30.0 * 1e-3; kc = 1 / (2 * dr); p_nom = 0.75e6
k_per_um_nom = 1 / (lam * p_nom)
P = np.loadtxt(W + "d32a/data_reg/positions_um.csv", delimiter=",")
d = P[idx, 1:3]
Mk = -(kc / RF) * A; k0 = -(kc / RF) * c0
kn = d @ Mk.T + k0[None]
ring = np.repeat(np.arange(8), [15, 20, 25, 30, 35, 40, 45, 50])[idx]
cal = dict(Mk=Mk.tolist(), k0=k0.tolist(), defocus_um=z0, astig_um=[0.0, 0.0], rotation_deg=float(np.degrees(np.arctan2(Mk[1, 0], Mk[1, 1]))),
           mirror=bool(np.linalg.det(Mk) < 0), scale_vs_nominal=float(np.sqrt(abs(np.linalg.det(Mk))) / k_per_um_nom),
           angle_per_pulse_urad=float(np.sqrt(abs(np.linalg.det(Mk))) * lam * 1e6), misfit=None, per_image_misfit=[None] * len(idx),
           k_solver=kn.tolist(), k_over_kc=(np.hypot(*kn.T) / kc).tolist(), twin=1, twin_source="kappa sign (physical kappa > 0) + vignetting geometry",
           kappa_eff_um2=kc / RF, c_eff=float((kc / RF) / k_per_um_nom), dr_um=dr, kc=kc, positions_index=idx.tolist(), ring=ring.tolist(),
           dark_images=[int(i) for i in np.where(np.hypot(*kn.T) > 1.02 * kc)[0]],
           source="d32a/geom_direct_v2.json (direct-image vignetting); positions: FZP X*2.5 um, FZP Y*0.2 um; defocus from WOTF z-scan (rings 1-2)")
os.makedirs(work, exist_ok=True)
json.dump(cal, open(os.path.join(work, "calibration.json"), "w"), indent=1)
print(f"{work}: {len(idx)} images, kc {kc:.3f}, |k|/kc {np.hypot(*kn.T).min()/kc:.3f}-{np.hypot(*kn.T).max()/kc:.3f}, "
      f"scale {cal['scale_vs_nominal']:.3f} x nominal (p = 0.75 m), c_eff {cal['c_eff']:.3f}, dark {len(cal['dark_images'])}")
