"""Configuration: measurement parameters, paths and algorithm settings (JSON file)."""
import copy
import json
import os

DEFAULTS = {
    # ---- data -----------------------------------------------------------------------------
    "data_dir": ".",
    "sample_subdir": "02sample_tif",
    "direct_subdir": "04direct_tif",
    "positions_csv": "positions.csv",       # idx, x[pulse], y[pulse], ... (no header)
    "n_repeat": 2,                          # repeated frames per scan position
    "repeat_order": "interleaved",          # "interleaved": a001,a002 = pos1 ; "blocked": a001..aN = pass 1
    "output_dir": "fpt_output",
    # ---- measurement ----------------------------------------------------------------------
    "energy_keV": 30.0,
    "pixel_um": 0.0319,                     # effective pixel in the sample plane
    "fzp_outer_zone_um": 0.100,             # NA = lambda / (2 dr)
    "sample_objective_distance_m": 0.75,    # p (FOV effect: kappa_nom = 1/(lambda p))
    "czp_focal_m": 2.4,                     # nominal angle per pulse = step / f
    "stage_step_um": 0.5,
    "dark_offset": 100.0,                   # counts (no dark frames available)
    # ---- preprocessing --------------------------------------------------------------------
    "zinger_threshold": 0.25,               # |R - median3(R)| above this -> replaced
    "flatten_sigma_px": 30.0,               # divide by Gaussian low-pass (flat-field residuals)
    "clip": [0.5, 2.0],
    # ---- calibration (weak-object transfer-function fit) ----------------------------------
    "calib": {
        "crop_px": 512, "crop_center": None,          # None -> image centre
        "q_band": [0.5, 3.2],                          # um^-1
        "pupil_edge": 0.05,                            # tanh edge width of pupil disc (um^-1)
        "phi_step_deg": 6.0,
        "z_grid_um": [-20000, -12000, -8000, -5000, -3000, -2000, -1000, -500, 0,
                      500, 1000, 2000, 3000, 5000, 8000, 12000, 20000],
        "scale_grid": [0.6, 2.0, 0.05],                # start, stop, step
        "z_grid2_um": [-3000, 3000, 250],
        "de_popsize": 12, "de_maxiter": 50, "de_seed": 3,
        "de_bounds": {"dphi_deg": 8.0, "dscale": 0.25, "dz_mm": 1.5, "astig_mm": 1.0, "k0": 1.2},
    },
    # ---- linear (weak-object) tile reconstruction: initial guess --------------------------
    "linear": {"tile_px": 256, "step_px": 128, "q_band": [0.3, 3.5], "reg_rel": 0.05},
    # ---- nonlinear FPM ----------------------------------------------------------------------
    "nonlinear": {
        "q_band": [0.3, 3.5],
        "support_scale": 1.05, "pupil_init_scale": 1.02,
        "max_k_over_kc": 0.97,                 # illuminations closer to the NA edge are excluded
        "M": None, "No": None,                 # None -> automatic (alias-free) sizes
        "tukey_alpha": 0.2,
        "n_pupil": 20, "n_joint": 150,
        "dir_floor": 0.1,
        "threads": 8,
    },
    # FOV effect: kappa_phys = c / (lambda p). c = 0 -> plane-wave model.
    "fov_factors": [0.0, 1.0],
    "export_band": [0.3, 3.5],
    "twin": None,                           # None: physical twin from corr(a, phi) of the linear solution; +1/-1: forced
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        user = {k: v for k, v in user.items() if not k.startswith("_")}
        cfg = _merge(cfg, user)
        dd = os.path.expanduser(os.path.expandvars(cfg["data_dir"]))   # e.g. "${FPT018_RAW}" (see paths_local.example.sh)
        if "$" in dd:
            raise ValueError(f"data_dir {cfg['data_dir']!r}: environment variable not set (see paths_local.example.sh)")
        cfg["data_dir"] = dd
        if not os.path.isabs(cfg["data_dir"]):
            cfg["data_dir"] = os.path.join(os.path.dirname(os.path.abspath(path)), cfg["data_dir"])
    if overrides:
        cfg = _merge(cfg, overrides)
    return derive(cfg)


def derive(cfg):
    lam = 1.239842 / cfg["energy_keV"] * 1e-3            # um
    cfg["lam_um"] = lam
    cfg["NA"] = lam / (2 * cfg["fzp_outer_zone_um"])
    cfg["kc"] = cfg["NA"] / lam                          # um^-1
    ang_per_pulse = cfg["stage_step_um"] * 1e-6 / cfg["czp_focal_m"]   # rad
    cfg["k_per_pulse_nominal"] = ang_per_pulse / lam     # um^-1 per pulse
    cfg["kappa_nom"] = 1.0 / (lam * cfg["sample_objective_distance_m"] * 1e6)   # um^-2
    return cfg
