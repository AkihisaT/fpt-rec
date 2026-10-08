#!/usr/bin/env python
"""BLIS-FPM (band-limited intensity-spectrum FPM) reconstruction pipeline for X-ray Fourier ptychography.

  python run_pipeline.py --config config_003.json                 # all steps
  python run_pipeline.py --config config_003.json --steps nonlinear,export,figures
  python run_pipeline.py --config config_003.json --quick         # smoke test (few iterations)

Steps: preprocess -> calibrate -> [posaffine] -> linear -> nonlinear -> export -> figures
  posaffine (optional; runs with --steps posaffine,... or with "position_affine": {"enabled": true} in the config):
  separates the image drift linear in the illumination position (mechanical 2x2 matrix: scale / rotation / shear)
  from the pupil parallax, corrects it, determines defocus / astigmatism from the contrast, and checks that the
  drift predicted by the pupil matches the data (fptrecon/posaffine.py; report results/position_affine.json).
Work files:  <output_dir>/work/    Results: <output_dir>/results/    Figures: <output_dir>/figures/
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fptrecon.config import load_config                      # noqa: E402
from fptrecon import preprocess as pre, wotf, nonlinear as nlr, postprocess as post, figures as figs  # noqa: E402

STEPS = ["preprocess", "calibrate", "posaffine", "linear", "nonlinear", "export", "figures"]


def ctag(c):
    return f"c{c:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", default="all")
    ap.add_argument("--quick", action="store_true", help="smoke test: coarse calibration, 3+5 nonlinear iterations")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--fov-factors", default=None, help="comma list of c (kappa = c/(lambda p)), overrides config")
    ap.add_argument("--n-pupil", type=int, default=None, help="pupil-only L-BFGS iterations (override)")
    ap.add_argument("--n-joint", type=int, default=None, help="joint object+pupil iterations (override)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.threads: cfg["nonlinear"]["threads"] = args.threads
    if args.fov_factors: cfg["fov_factors"] = [float(v) for v in args.fov_factors.split(",")]
    if args.n_pupil is not None: cfg["nonlinear"]["n_pupil"] = args.n_pupil
    if args.n_joint is not None: cfg["nonlinear"]["n_joint"] = args.n_joint
    steps = list(STEPS) if args.steps == "all" else args.steps.split(",")
    if args.steps == "all" and not cfg.get("position_affine", {}).get("enabled", False):
        steps.remove("posaffine")                        # optional step: default behaviour unchanged
    out = cfg["output_dir"]
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(os.path.abspath(args.config)), out)
    work, resd, figd = (os.path.join(out, d) for d in ("work", "results", "figures"))
    for d in (work, resd, figd): os.makedirs(d, exist_ok=True)
    logf = open(os.path.join(out, "pipeline.log"), "a")
    t00 = time.time()

    def log(msg):
        line = f"[{time.time()-t00:7.1f}s] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
    log(f"config {args.config}; steps {steps}; quick={args.quick}; lambda={cfg['lam_um']*1e3:.6f} nm, NA={cfg['NA']:.4e}, "
        f"kc={cfg['kc']:.3f} um^-1, nominal k/pulse={cfg['k_per_pulse_nominal']:.5f} um^-1, kappa_nom={cfg['kappa_nom']:.5f} um^-2")
    f_pre, f_meta, f_cal = (os.path.join(work, n) for n in ("preprocessed.npy", "preprocess_meta.npz", "calibration.json"))

    # 1 ------------------------------------------------------------------------------------
    if "preprocess" in steps:
        R, meta = pre.preprocess(cfg, log)
        np.save(f_pre, R); np.savez_compressed(f_meta, **meta)
        log(f"preprocess -> {f_pre} {R.shape}")
    R = np.load(f_pre, mmap_mode="r")
    meta = dict(np.load(f_meta))
    Pst = meta["positions"]
    if set(steps) <= {"preprocess"} and not os.path.exists(f_cal):
        log("done (preprocess only; calibration.json is made by the calibrate step or an external script)"); return

    # 2 ------------------------------------------------------------------------------------
    if "calibrate" in steps:
        cal = wotf.calibrate(np.asarray(R), cfg, Pst, log, quick=args.quick)
        json.dump(cal, open(f_cal, "w"), indent=1)
    cal = json.load(open(f_cal))
    kn = np.array(cal["k_solver"])

    # 2b (optional) ------------------------------------------------------------------------
    if "posaffine" in steps:
        from fptrecon import posaffine
        f_raw = os.path.join(work, "preprocessed_before_posaffine.npy")
        if not os.path.exists(f_raw):
            del R; os.replace(f_pre, f_raw)              # keep the input stack; re-runs start from it
        R0 = np.load(f_raw, mmap_mode="r")
        log("posaffine: mechanical image drift vs pupil parallax")
        posaffine.selftest_sign(log)
        Rc, upd, rep, _ = posaffine.run(R0, kn, np.asarray(Pst, float), np.array(cal["Mk"]), cfg, cal, log)
        np.save(f_pre, Rc.astype(np.float32)); del Rc
        R = np.load(f_pre, mmap_mode="r")
        cal.update(upd); json.dump(cal, open(f_cal, "w"), indent=1)
        json.dump(rep, open(os.path.join(resd, "position_affine.json"), "w"), indent=1)
        posaffine.fig_position_affine(os.path.join(figd, "fig0_position_affine.png"), rep, np.asarray(Pst, float), cfg["pixel_um"])
        log(f"posaffine -> {f_pre}; calibration defocus {cal['defocus_um']:.0f} um, astig {np.round(cal['astig_um'], 0).tolist()} um")

    # 3 ------------------------------------------------------------------------------------
    if "linear" in steps:
        a0, p0, rs = wotf.tile_linear_recon(np.asarray(R), kn, cfg, 0.0, cal["defocus_um"], cal["astig_um"], log=log)
        tw, cc = post.twin_sign(a0, p0)
        cal["corr_a_phi_linear"] = cc; cal["twin_from_linear"] = tw
        if cfg.get("twin") in (-1, 1):                  # forced by the user (e.g. weak |corr| in the linear solution)
            tw = int(cfg["twin"]); log(f"twin forced to {tw:+d} by config (linear criterion gave {cal['twin_from_linear']:+d})")
        cal["twin"] = tw; cal["twin_source"] = "config" if cfg.get("twin") in (-1, 1) else "linear corr(a,phi)"
        json.dump(cal, open(f_cal, "w"), indent=1)
        log(f"linear c=0: mean tile residual {np.mean(list(rs.values())):.4f}; corr(a,phi)={cc:+.3f} -> twin={tw:+d}")
        for c in cfg["fov_factors"]:
            ks = c * cfg["kappa_nom"] * tw
            if c == 0:
                a_, p_, rs_ = a0, p0, rs
            else:
                a_, p_, rs_ = wotf.tile_linear_recon(np.asarray(R), kn, cfg, ks, cal["defocus_um"], cal["astig_um"], log=log)
                log(f"linear c={c:g}: mean tile residual {np.mean(list(rs_.values())):.4f}")
            np.savez_compressed(os.path.join(work, f"linear_{ctag(c)}.npz"), a=a_, phi=p_, kappa_solver=ks)
    cal = json.load(open(f_cal))
    tw = cal.get("twin", 1)

    # 4 ------------------------------------------------------------------------------------
    if "nonlinear" in steps:
        for c in cfg["fov_factors"]:
            li = np.load(os.path.join(work, f"linear_{ctag(c)}.npz"))
            ks = float(li["kappa_solver"])
            log(f"nonlinear c={c:g} (kappa_solver={ks:+.5f} um^-2)")
            res = nlr.run_nonlinear(np.asarray(R), kn, cfg, ks, cal, li["a"], li["phi"],
                                    n_pupil=3 if args.quick else None, n_joint=5 if args.quick else None, log=log)
            np.savez_compressed(os.path.join(work, f"nonlinear_{ctag(c)}.npz"), **res)

    # 5 ------------------------------------------------------------------------------------
    phys_by_c, res_by_c, misfit, tiles = {}, {}, {}, {}
    if "export" in steps or "figures" in steps:
        for c in cfg["fov_factors"]:
            fn = os.path.join(work, f"nonlinear_{ctag(c)}.npz")
            if not os.path.exists(fn):
                continue
            try:
                res = dict(np.load(fn))
            except Exception as e:                      # e.g. truncated file from an interrupted run
                log(f"skip {fn}: {e}")
                continue
            phys = post.to_physical(res, tw, cfg, R.shape[-1])
            tw_nl, cc_nl = post.twin_sign(res["a"], res["phi"])
            tag = "planewave" if c == 0 else f"FOV_{ctag(c)}"
            summ, tm, coef = post.export(resd, tag, phys, res, cfg, extra=dict(
                fov_factor=c, kappa_phys_um2=c * cfg["kappa_nom"], corr_a_phi_nonlinear=cc_nl, twin_applied=tw))
            phys_by_c[c], res_by_c[c], misfit[c], tiles[c] = phys, res, summ["final_misfit"], tm
            log(f"export {tag}: misfit {summ['final_misfit']:.4f}, defocus {summ['defocus_equiv_um']:.0f} um, "
                f"tile misfit centre {tm[2:4, 2:4].mean():.3f} / border {np.r_[tm[0], tm[-1], tm[1:-1, 0], tm[1:-1, -1]].mean():.3f}")
        # physical-frame calibration summary
        Mk = np.array(cal["Mk"])
        calp = dict(twin=tw, Mk_phys=(tw * Mk).tolist(), k0_phys=(tw * np.array(cal["k0"])).tolist(),
                    defocus_phys_um=tw * cal["defocus_um"], angle_per_pulse_urad=cal["angle_per_pulse_urad"],
                    scale_vs_nominal=cal["scale_vs_nominal"], k_phys=(tw * kn).tolist(), k_over_kc=cal["k_over_kc"],
                    misfit_by_fov_factor=misfit,
                    convention="k=(row,col) um^-1, illumination exp(+i2pi k.x); O=exp(-a+i phi), material phi<0; "
                               "pupil phase W=pi*lam*z*|k|^2 for defocus z; FOV term exp(i pi kappa |x|^2), kappa=c/(lam p)")
        json.dump(calp, open(os.path.join(resd, "calibration_physical.json"), "w"), indent=1)

    # 6 ------------------------------------------------------------------------------------
    if "figures" in steps:
        figs.fig_calibration(os.path.join(figd, "fig1_calibration.png"), np.asarray(R), meta, cal, cfg, tw)
        if phys_by_c:
            figs.fig_reconstruction(os.path.join(figd, "fig2_reconstruction.png"), phys_by_c, cfg)
            figs.fig_pupil(os.path.join(figd, "fig3_pupil.png"), phys_by_c, res_by_c, cfg)
            figs.fig_fov(os.path.join(figd, "fig4_fov_effect.png"), misfit, tiles)
        log(f"figures -> {figd}")
    log("done")


if __name__ == "__main__":
    main()
