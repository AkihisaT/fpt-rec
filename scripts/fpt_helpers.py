"""Helper numbers for FPT reconstructions (from the workflow in .claude/skills/fpt-recon-workflow/SKILL.md).

usage (repository root):
  python scripts/fpt_helpers.py basic --energy 30 --dr 0.25 --pixel 0.0319 --p 0.75
  python scripts/fpt_helpers.py kk d32a/data_reg/positions_um.csv --cols 1,2 --k-per-um 0.0327 --kc 2.0 --rings 15,20,25,30,35,40,45,50
  python scripts/fpt_helpers.py check
"""
import os
import math

HC_KEV_NM = 1.239842


def fpt_basic_numbers(energy_keV=30.0, dr_um=0.25, pixel_um=0.0319, p_m=0.75, czp_focal_m=None, stage_step_um=None,
                      fzp_diameter_um=None, grid_px=512):
    """Basic FPT numbers from the experimental conditions (units in the keys).

    lambda = 1.239842 / E [nm]; NA = lambda / (2 dr); kc = NA / lambda = 1 / (2 dr) [um^-1] (pupil radius in k);
    DOF ~ lambda / NA^2; kappa_nom = 1 / (lambda p) [um^-2] (FOV quadratic phase, also k per um of lens shift for a
    parallel beam + objective-FZP scan); for a condenser scan the nominal k per stage step is step / (f lambda).
    Returns a dict; compare kc with the pupil radius seen in the data before trusting dr.
    """
    lam_um = HC_KEV_NM / energy_keV * 1e-3
    na = lam_um / (2.0 * dr_um)
    kc = 1.0 / (2.0 * dr_um)
    out = {
        "lambda_nm": lam_um * 1e3,
        "NA": na,
        "kc_um^-1": kc,
        "pupil_radius_px_on_grid": kc * grid_px * pixel_um,
        "pixel_nyquist_um^-1": 1.0 / (2.0 * pixel_um),
        "coherent_BF_halfpitch_nm": 1e3 / (2.0 * kc),
        "DOF_lambda_over_NA2_um": lam_um / na ** 2,
        "kappa_nom_um^-2": 1.0 / (lam_um * p_m * 1e6),
        "k_per_um_lens_shift_um^-1": 1.0 / (lam_um * p_m * 1e6),
    }
    if czp_focal_m and stage_step_um:
        out["k_per_stage_step_um^-1"] = stage_step_um / (czp_focal_m * 1e6 * lam_um)
    if fzp_diameter_um:
        out["fzp_focal_m"] = fzp_diameter_um * dr_um / lam_um * 1e-6
    return out


def fpt_k_over_kc(positions, k_per_unit, kc, k0=(0.0, 0.0), rings=None):
    """|k|/kc of each illumination from scan positions (N x 2, stage units) and a nominal k per unit.

    positions: list or array of (x, y); k_per_unit: um^-1 per stage unit (fpt_basic_numbers gives the nominal value;
    the calibration scale in past condenser data was 1.15-1.32 x nominal, so use the calibrated value when available);
    k0: illumination offset (um^-1). rings: optional ring label per position -> per-ring min / median / max.
    Returns dict with 'k_over_kc' (list) and, if rings is given, 'per_ring'. Classify bright field (< 1) and dark field
    (> 1) from these values; rings just above 1 may carry the direct-beam tail and need checking.
    """
    import numpy as np
    P = np.asarray(positions, float)
    k = P * float(k_per_unit) + np.asarray(k0, float)
    r = np.hypot(k[:, 0], k[:, 1]) / float(kc)
    out = {"k_over_kc": r.round(4).tolist(), "n_bright(<1)": int((r < 1).sum()), "n_dark(>1)": int((r > 1).sum())}
    if rings is not None:
        R = np.asarray(rings)
        out["per_ring"] = {str(g): [float(r[R == g].min().round(3)), float(np.median(r[R == g]).round(3)),
                                    float(r[R == g].max().round(3))] for g in sorted(set(R.tolist()))}
    return out


def fpt_package_check(root):
    """Check that the FPT pipeline repository (root folder) is complete; returns dict with the missing items."""
    need = ["README.md", "README_統合パイプライン.md", "CLAUDE.md", "paths_local.example.sh", "build_pptx.py",
            ".claude/skills/fpt-recon-workflow/SKILL.md", "scripts/fpt_helpers.py", "tests/selftest.py", "tests/check_32a.sh",
            "fpt_pipeline/run_pipeline.py", "fpt_pipeline/fptrecon/nonlinear.py", "fpt_pipeline/fptrecon/posaffine.py",
            "fpt_pipeline/fptrecon/coherence.py", "fpt_pipeline/fptrecon/preprocess.py", "fpt_pipeline/environment.yml",
            "dip_pipeline/run_dip.py", "dip_pipeline/dipfpm/coherence.py", "dip_pipeline/environment_dip.yml",
            "epry_pipeline/run_epry.py", "epry_pipeline/eprfpm/core.py",
            "d018/compare_018.py", "d036/README_036_暗視野パイプライン.md", "d036/blis_df.py",
            "d32a/runs/run_32a_main.sh", "d32a/fit_geom_direct_32a.py", "d32a/blis_32a.py", "tests/regress_unified.py"]
    missing = [f for f in need if not os.path.exists(os.path.join(root, f))]
    n_files = sum(len(fs) for _, _, fs in os.walk(root))
    return {"root": root, "n_files": n_files, "missing": missing, "ok": not missing}


def main(argv=None):
    """Command line: python scripts/fpt_helpers.py basic|kk|check ... (run from the repository root)."""
    import argparse, json
    ap = argparse.ArgumentParser(description="FPT helper numbers (see .claude/skills/fpt-recon-workflow/SKILL.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("basic", help="lambda, NA, kc, DOF, kappa from the experimental conditions")
    b.add_argument("--energy", type=float, default=30.0, help="keV"); b.add_argument("--dr", type=float, default=0.25, help="FZP outermost zone um")
    b.add_argument("--pixel", type=float, default=0.0319, help="sample-plane pixel um"); b.add_argument("--p", type=float, default=0.75, help="sample-objective distance m")
    b.add_argument("--czp-focal", type=float, default=None, help="CZP focal length m"); b.add_argument("--step", type=float, default=None, help="stage step um")
    b.add_argument("--fzp-diameter", type=float, default=None, help="um"); b.add_argument("--grid", type=int, default=512)
    k = sub.add_parser("kk", help="|k|/kc per illumination from a CSV of positions")
    k.add_argument("csv"); k.add_argument("--cols", default="1,2", help="0-based columns of x,y (stage units)")
    k.add_argument("--scale", default="1,1", help="um per unit of x,y (e.g. 2.5,0.2)"); k.add_argument("--k-per-um", type=float, required=True)
    k.add_argument("--kc", type=float, required=True); k.add_argument("--rings", default=None, help="ring sizes, e.g. 15,20,25")
    c = sub.add_parser("check", help="check that the repository is complete"); c.add_argument("root", nargs="?", default=".")
    a = ap.parse_args(argv)
    if a.cmd == "basic":
        out = fpt_basic_numbers(a.energy, a.dr, a.pixel, a.p, a.czp_focal, a.step, a.fzp_diameter, a.grid)
    elif a.cmd == "kk":
        import numpy as np
        P = np.loadtxt(a.csv, delimiter=",", ndmin=2); cx, cy = (int(v) for v in a.cols.split(",")); sx, sy = (float(v) for v in a.scale.split(","))
        pos = np.c_[P[:, cx] * sx, P[:, cy] * sy]
        rings = np.repeat(np.arange(len(a.rings.split(","))), [int(v) for v in a.rings.split(",")]) if a.rings else None
        out = fpt_k_over_kc(pos, a.k_per_um, a.kc, rings=rings); out.pop("k_over_kc") if rings is not None else None
    else:
        out = fpt_package_check(a.root)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
