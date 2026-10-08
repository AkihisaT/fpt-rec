"""Why does the free-pupil check fit keep its pupil quadratic ~1.2 mm less negative than the fixed-pupil focus loop?
Re-runs the posaffine check fit (free pupil, fixed positions) on the corrected aff and aff2 stacks and analyses the
recovered pupil phase W: (a) order-4 power-basis polynomial (as in posaffine), (b) pure quadratic + tilt, (c) Zernike
(balanced) defocus / spherical, (d) the parallax predicted by the local pupil gradient at the illumination points."""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.abspath("fpt_pipeline"))
from run_pipeline import load_config
from fptrecon import posaffine as PA, nonlinear as nlr
from fptrecon.optics import kgrid
out = {}
for tag in ("aff", "aff2"):
    cfg = load_config(f"fpt_pipeline/config_32a_bf_{tag}.json"); pa = dict(PA.DEFAULTS); pa.update(cfg.get("position_affine", {}))
    wd = f"fpt_pipeline/fpt_output_32a_bf_{tag}/work/"
    R = np.load(wd + "preprocessed.npy", mmap_mode="r"); cal = json.load(open(wd + "calibration.json"))
    rep = json.load(open(f"fpt_pipeline/fpt_output_32a_bf_{tag}/results/position_affine.json"))
    kn = np.array(cal["k_solver"]); pos = np.asarray(dict(np.load(wd + "preprocess_meta.npz"))["positions"], float)
    lam, px, kc = cfg["lam_um"], cfg["pixel_um"], cfg["kc"]; ks = pa["fov_factor"] * cfg["kappa_nom"] * int(cal.get("twin", 1))
    nl_cfg = json.loads(json.dumps(cfg)); nl_cfg["nonlinear"]["threads"] = 4; z0 = np.zeros((64, 64), np.float32)
    t0 = time.time()
    res = nlr.run_nonlinear(np.asarray(R), kn, nl_cfg, ks, cal, z0, z0, n_pupil=pa["n_pupil"], n_joint=pa["n_joint"], log=lambda m: None)
    W, Pamp, dk = res["W"], res["Pamp"], float(res["dk"]); M = W.shape[0]; yy, xx = kgrid(M, dk); r = np.hypot(yy, xx)
    af = PA.autofocus(res["a"], res["phi"], float(res["dxo"]), lam, band=cfg["nonlinear"]["q_band"])
    d = dict(calib_defocus_um=cal["defocus_um"], misfit=float(res["hist"][-1]), object_refocus_um=af["z_um"], fit_s=time.time() - t0)
    for order, cex in ((4, 0.25), (2, 0.25), (4, 0.0), (2, 0.0), (6, 0.25)):
        p = PA.pupil_poly(W, Pamp, dk, kc, order, cex); z, a = PA.defocus_astig_from_poly(p, lam)
        B = PA.fit_drift(PA.pupil_shifts_px(p, kn, px), pos, kn)[0]
        d[f"poly{order}_cex{cex}"] = dict(z_um=z, astig_um=a, resid_rad=p["resid_rad"], B_px_per_stage=B.tolist())
    # Zernike-balanced defocus on the unit disc rho = r/(0.95 kc): W ~ c4 (2 rho^2 - 1) + c11 (6 rho^4 - 6 rho^2 + 1) + tilts + astig
    m = np.isfinite(W) & (Pamp > 0.5) & (r < 0.95 * kc); rho = r / (0.95 * kc); th = np.arctan2(yy, xx)
    Z = [np.ones_like(rho), rho * np.cos(th), rho * np.sin(th), 2 * rho ** 2 - 1, rho ** 2 * np.cos(2 * th), rho ** 2 * np.sin(2 * th),
         (3 * rho ** 3 - 2 * rho) * np.cos(th), (3 * rho ** 3 - 2 * rho) * np.sin(th), 6 * rho ** 4 - 6 * rho ** 2 + 1]
    c = np.linalg.lstsq(np.stack([z_[m] for z_ in Z], 1), W[m], rcond=None)[0]
    k2 = (0.95 * kc) ** 2
    d["zernike"] = dict(c_defocus_rad=float(c[3]), c_coma_rad=[float(c[6]), float(c[7])], c_spherical_rad=float(c[8]),
                        z_balanced_um=float(2 * c[3] / k2 / (np.pi * lam)),       # 2 c4 rho^2 = pi lam z r^2
                        z_rho2_total_um=float((2 * c[3] - 6 * c[8]) / k2 / (np.pi * lam)))
    # local gradient of W at the illumination points (bilinear interpolation of the numerical gradient)
    gy, gx = np.gradient(np.where(m, W, np.nan), dk)
    def interp(G, k):
        fy = k[:, 0] / dk + M // 2; fx = k[:, 1] / dk + M // 2; iy, ix = np.floor(fy).astype(int), np.floor(fx).astype(int); ty, tx = fy - iy, fx - ix
        return (G[iy, ix] * (1 - ty) * (1 - tx) + G[iy + 1, ix] * ty * (1 - tx) + G[iy, ix + 1] * (1 - ty) * tx + G[iy + 1, ix + 1] * ty * tx)
    sl = -np.stack([interp(gy, kn), interp(gx, kn)], 1) / (2 * np.pi) / px
    ok = np.isfinite(sl).all(1)
    Bl = PA.fit_drift(sl[ok], pos[ok], kn[ok])[0]
    d["local_gradient"] = dict(n_ok=int(ok.sum()), B_px_per_stage=Bl.tolist())
    d["B_retained_px_per_stage"] = rep["B_pupil_px_per_stage"]
    # equivalent defocus of a 2x2 B: isotropic part / (lam kappa) with k = Mk pos
    Mk = np.array(cal["Mk"]); Pm = PA.image_frame_perm(Mk)
    for key in ("B_retained_px_per_stage",):
        pass
    np.savez_compressed(f"d32a/af_test/pupil_check_{tag}.npz", W=W, Pamp=Pamp, dk=dk)
    out[tag] = d; print(tag, json.dumps({k: v for k, v in d.items() if k != "local_gradient"}, default=float)[:1500], flush=True)
json.dump(out, open("d32a/af_test/pupil_basis_test.json", "w"), indent=1, default=float)
