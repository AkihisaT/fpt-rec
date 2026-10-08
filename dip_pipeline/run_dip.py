#!/usr/bin/env python
"""Deep Image Prior (untrained U-Net) FPM reconstruction — driver.

Prerequisite: fptrecon (BLIS-FPM pipeline) steps preprocess, calibrate (-> <fpt_work_dir>/preprocessed.npy, calibration.json).

  python run_dip.py --config config_dip_003.json                         # train (all fov_factors) + report
  python run_dip.py --config config_dip_003.json --n-iter 20             # smoke test
  python run_dip.py --config config_dip_003.json --steps train --fov-factors 0   # train one model only
  python run_dip.py --config config_dip_003.json --steps report          # export/compare existing states

Training of different fov_factors can run as separate processes (each writes <tag>_state.npz); the
report step then gathers whatever states exist.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dipfpm import train as T, evaluate as E                      # noqa: E402
from dipfpm.physics import FPMPhysics, ZERN                        # noqa: E402

LABELS = {"DIP_planewave": "DIP, plane-wave model", "DIP_FOV_c1": "DIP, FOV model",
          "DIP_FOV_c1_freepupil": "DIP, FOV model + free pupil", "DIP_planewave_freepupil": "DIP, plane wave + free pupil",
          "DIP_planewave_noshift": "DIP, plane wave, no shift", "DIP_FOV_c1_noshift": "DIP, FOV, no shift"}
TAG_COLORS = {"DIP_planewave": "#7570b3", "DIP_FOV_c1": "#e7298a", "DIP_FOV_c1_freepupil": "#a6761d",
              "DIP_planewave_freepupil": "#1f78b4", "DIP_planewave_noshift": "#b2abd2", "DIP_FOV_c1_noshift": "#fbb4ae"}
PREV_COLORS = ["#d95f02", "#1b9e77", "#e6ab02", "#66a61e"]


def load_cfg(path, over):
    cfg = json.load(open(path, encoding="utf-8")); base = os.path.dirname(os.path.abspath(path))
    rel = lambda p: p if (p is None or os.path.isabs(p)) else os.path.join(base, p)
    for k in ("fpt_work_dir", "previous_results", "output_dir"):
        cfg[k] = rel(cfg.get(k))
    for p in cfg.get("previous_native", []):
        p["file"] = rel(p["file"])
    for k, v in over.items():
        if v is not None: cfg["dip"][k] = v
    lam = 1.239842 / cfg["energy_keV"] * 1e-3
    cfg.update(lam_um=lam, NA=lam / (2 * cfg["fzp_outer_zone_um"]), kc=1 / (2 * cfg["fzp_outer_zone_um"]),
               kappa_nom=1.0 / (lam * cfg["sample_objective_distance_m"] * 1e6))
    return cfg


def tag_of(c, cfg=None):
    t = "DIP_planewave" if c == 0 else f"DIP_FOV_c{c:g}"
    if cfg is not None and cfg["dip"].get("tag_suffix"):
        t += cfg["dip"]["tag_suffix"]
    return t


def make_phys(cfg, data, ks):
    from dipfpm.coherence import pc_modes
    md, mw = pc_modes(cfg["dip"].get("partial_coherence"), data["kn"][data["keep"]], cfg["kc"], cfg["dip"]["pupil_radius"])
    return FPMPhysics(data["kn"][data["keep"]], data["dk"], data["M"], data["No"], cfg["kc"], cfg["lam_um"], kappa=ks,
                      axis_offset_um=data["axis_off"], pupil_radius=cfg["dip"]["pupil_radius"], modes=md, mode_w=mw)


def step_train(cfg, data, log):
    out = cfg["output_dir"]
    for c in cfg["dip"]["fov_factors"]:
        tag = tag_of(c, cfg); ks = c * cfg["kappa_nom"] * data["twin"]
        log(f"== train {tag}: kappa_solver={ks:+.5f} um^-2")
        res = T.run_dip(cfg, data, ks, log, checkpoint=os.path.join(out, f"{tag}_state.npz"))
        log(f"{tag}: selected iteration {res['best']['it']} (validation {res['best']['val']:.4f})")


def step_report(cfg, data, log):
    out = cfg["output_dir"]; d = cfg["dip"]; twin = data["twin"]; dx = cfg["pixel_um"]; band = cfg["export_band"]
    nk = len(data["keep"]); vpos = [list(data["keep"]).index(k) for k in data["val"]]
    recs, frcs, metrics, summary, res_by = {}, {}, {}, {}, {}
    common_w = None
    if d.get("eval_vignetting_mask"):      # same pixel mask (FOV model, c = 1) for every method's metric
        import torch
        from scipy.signal.windows import tukey
        w0 = torch.tensor(np.outer(tukey(data["M"], 0.2), tukey(data["M"], 0.2)).astype(np.float32))
        common_w = T.vignetting_weights(make_phys(cfg, data, cfg["kappa_nom"] * twin), torch.tensor(data["z0"], dtype=torch.float32), w0).numpy()
        log(f"evaluation mask: excluded fraction {float(1 - (common_w > 0.5 * w0.numpy()).mean()):.3f}")
    _bm = E.band_misfit
    DG = bool(cfg["dip"].get("partial_coherence"))     # partial coherence: dark-gain metric (as compare_018 --dark-gain)
    E_band_misfit = lambda *a, **k: _bm(*a, weights=common_w, dark_gain=DG, **k)
    # ---- DIP results
    import glob
    pref = ["DIP_planewave", "DIP_FOV_c1", "DIP_FOV_c1_freepupil"]
    states = sorted(glob.glob(os.path.join(out, "DIP_*_state.npz")),
                    key=lambda p: (pref.index(os.path.basename(p)[:-10]) if os.path.basename(p)[:-10] in pref else 99, p))
    for f in states:
        tag = os.path.basename(f)[:-len("_state.npz")]
        res = T.load_state(f); res_by[tag] = res; b = res["best"]; ks = res["kappa_solver"]
        c = 0.0 if tag.startswith("DIP_planewave") else float(tag.split("_c")[1].split("_")[0])
        ex = E.export_dip(out, tag, res, data, cfg)
        col = TAG_COLORS.get(tag, "#666666")
        Wmap = ex["W_solver_total"]
        recs[tag] = dict(phi=ex["phi"], T=ex["T"], zc_phys=ex["zc_phys"], color=col)
        phys = make_phys(cfg, data, ks)
        a_bp = E.band_pass(b["a"], res["dxo"], *band); p_bp = E.band_pass(b["phi"], res["dxo"], *band)
        dki = b.get("dk_illum", np.zeros((nk, 2)))
        m_sh, per = E_band_misfit(phys, a_bp, p_bp, b["zc"], b["shifts_um"], data["I_meas"], band, dk_illum=dki, W_map=Wmap)
        m_0, _ = E_band_misfit(phys, a_bp, p_bp, b["zc"], 0 * b["shifts_um"], data["I_meas"], band, W_map=Wmap)
        fin = res.get("final", {})
        m_fin = None
        if "a" in fin:
            Wf_fin = None
            if np.abs(np.asarray(fin.get("W_free", 0.0))).max() > 0:
                Wf_fin = np.nan_to_num(E.pupil_map(fin["zc"], res["M"], res["dk"], cfg["kc"])) + fin["W_free"]
            m_fin, _ = E_band_misfit(phys, E.band_pass(fin["a"], res["dxo"], *band), E.band_pass(fin["phi"], res["dxo"], *band),
                                     fin["zc"], fin["shifts_um"], data["I_meas"], band, dk_illum=fin.get("dk_illum"), W_map=Wf_fin)
        itr_pos = [i for i in range(nk) if i not in vpos]
        metrics[tag] = (float(np.mean(per[itr_pos])), float(np.mean(per[vpos])) if vpos else np.nan, col, True)
        summary[tag] = dict(
            fov_factor=c, kappa_solver_um2=ks, iterations_run=int(res["n_done"]), selected_iteration=int(b["it"]),
            validation_loss=float(b["val"]), band_misfit=m_sh, band_misfit_without_shift_correction=m_0,
            band_misfit_final_iteration=m_fin, pupil_free=bool(Wmap is not None),
            seconds_per_iteration=float(res["hist"]["t"][-1] / max(res["hist"]["it"][-1], 1)),
            band_misfit_validation_images=float(np.mean(per[vpos])) if vpos else None,
            band_misfit_training_images=float(np.mean(np.delete(per, vpos))),
            zernike_rad_rms_physical={ZERN[j][2]: round(float(ex["zc_phys"][j]), 4) for j in range(len(ZERN))},
            shift_mode=d.get("shift_mode", "illumination"),
            image_shift_px={int(k): [round(float(v), 3) for v in b["shifts_um"][i] / dx] for i, k in enumerate(data["keep"])},
            illumination_dk_um_inv={int(k): [round(float(v), 5) for v in dki[i]] for i, k in enumerate(data["keep"])},
            intensity_factor={int(k): round(float(b["c"][i]), 4) for i, k in enumerate(data["keep"])},
            images_train=[int(v) for v in data["train"]], images_validation=[int(v) for v in data["val"]])
        log(f"{tag}: it {b['it']}/{res['n_done']}, band misfit {m_sh:.4f} (no shift corr.: {m_0:.4f})")
    # ---- BLIS-FPM (fptrecon), same crop / grid / metric
    for i, p in enumerate(cfg.get("previous_native", [])):
        if not os.path.exists(p["file"]):
            log(f"previous result not found: {p['file']}"); continue
        pn = E.load_previous_native(p["file"], data["crop"], data["H"], dx, data["M"], data["dk"])
        ks = pn["kappa_solver"]; phys = make_phys(cfg, data, ks); col = PREV_COLORS[i % 4]; lb = p["label"]
        a_bp = E.band_pass(pn["a"], dx, *band); p_bp = E.band_pass(pn["phi"], dx, *band); sh0 = np.zeros((nk, 2))
        m_free, per = E_band_misfit(phys, a_bp, p_bp, np.zeros(len(ZERN)), sh0, data["I_meas"], band, W_map=pn["W"])
        zfit = T.zernike_fit_map(pn["W"], data["dk"], cfg["kc"])
        m_z, _ = E_band_misfit(phys, a_bp, p_bp, zfit, sh0, data["I_meas"], band)
        itr_pos = [i for i in range(nk) if i not in vpos]
        metrics[lb] = (float(np.mean(per[itr_pos])), float(np.mean(per[vpos])) if vpos else np.nan, col, False)
        zc_phys = E.zern_to_physical(zfit, twin)
        recs[lb] = dict(phi=twin * p_bp, T=np.exp(-a_bp), zc_phys=zc_phys, color=col)
        summary[lb] = dict(file=os.path.basename(p["file"]), kappa_solver_um2=ks, band_misfit=m_free,
                           band_misfit_12zernike_pupil=m_z,
                           band_misfit_validation_images=float(np.mean(per[vpos])) if vpos else None,
                           band_misfit_training_images=float(np.mean(np.delete(per, vpos))),
                           misfit_fullFOV_own_metric=pn["final_misfit_fullFOV"],
                           zernike_rad_rms_physical={ZERN[j][2]: round(float(zc_phys[j]), 4) for j in range(len(ZERN))})
        log(f"{lb}: band misfit {m_free:.4f} (12-Zernike pupil {m_z:.4f})")
    # ---- cross-method agreement (phase, inner region)
    m_ = 32; inner = (slice(m_, -m_), slice(m_, -m_))
    dips = [t for t in recs if t.startswith("DIP")]; prevs = [t for t in recs if not t.startswith("DIP")]
    for t in dips:
        for j, pl in enumerate(prevs):
            q, f = E.frc(recs[t]["phi"][inner], recs[pl]["phi"][inner], dx)
            summary[t][f"phase_corr_vs_{pl}"] = float(np.corrcoef(recs[t]["phi"][inner].ravel(), recs[pl]["phi"][inner].ravel())[0, 1])
            summary[t][f"transmission_corr_vs_{pl}"] = float(np.corrcoef(recs[t]["T"][inner].ravel(), recs[pl]["T"][inner].ravel())[0, 1])
            summary[t][f"phase_FRC_vs_{pl}"] = dict(q_um_inv=np.round(q, 3).tolist(), frc=np.round(f, 3).tolist())
            same_model = ("FOV" in t) == ("FOV" in pl)
            if same_model:
                frcs[LABELS.get(t, t).replace("DIP, ", "")] = (q, f, recs[t]["color"], "--" if "free" in t else "-")
    if res_by:
        E.fig_training(os.path.join(out, "fig_dip_training.png"), res_by,
                       {t: (LABELS.get(t, t).replace("DIP, ", ""), TAG_COLORS.get(t, "#666666")) for t in res_by},
                       n_fit=len(data["train"]), n_val=len(data["val"]))
    order = dips + prevs
    lab = lambda k: LABELS.get(k, k.replace("fptrecon", "BLIS-FPM").replace("FOV c1", "FOV model").replace("planewave", "plane-wave model").replace("BLIS-FPM ", "BLIS-FPM, "))
    E.fig_compare(os.path.join(out, "fig_dip_comparison.png"), {lab(k): recs[k] for k in order}, cfg, frcs,
                  {lab(k): metrics[k] for k in order if k in metrics}, n_fit=len(data["train"]), n_val=len(data["val"]))
    json.dump(summary, open(os.path.join(out, "dip_summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"report written to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", default="train,report")
    ap.add_argument("--n-iter", type=int); ap.add_argument("--threads", type=int)
    ap.add_argument("--fov-factors"); ap.add_argument("--loss", choices=["intensity", "amplitude"])
    ap.add_argument("--output-dir")
    ap.add_argument("--pupil-free", action="store_true", help="add a free pupil-phase map to the Zernike pupil")
    ap.add_argument("--no-shift", action="store_true", help="disable per-image shift correction")
    a = ap.parse_args()
    cfg = load_cfg(a.config, {"n_iter": a.n_iter, "threads": a.threads, "loss": a.loss,
                              "fov_factors": [float(v) for v in a.fov_factors.split(",")] if a.fov_factors else None})
    if a.output_dir: cfg["output_dir"] = os.path.abspath(a.output_dir)
    if a.pupil_free: cfg["dip"]["pupil_free"] = True; cfg["dip"]["tag_suffix"] = cfg["dip"].get("tag_suffix", "") + "_freepupil"
    if a.no_shift: cfg["dip"]["shift_scale_px"] = 0.0; cfg["dip"]["tag_suffix"] = cfg["dip"].get("tag_suffix", "") + "_noshift"
    out = cfg["output_dir"]; os.makedirs(out, exist_ok=True)
    steps = a.steps.split(",")
    logf = open(os.path.join(out, f"dip_{'_'.join(steps)}_{'_'.join(tag_of(c, cfg) for c in cfg['dip']['fov_factors'])}.log"), "a")
    t00 = time.time()

    def log(m):
        line = f"[{time.time()-t00:7.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    d = cfg["dip"]
    log(f"config {a.config}; steps {steps}; loss={d['loss']}, n_iter={d['n_iter']}, U-Net {d['unet_px']} px "
        f"base {d['unet_base']} depth {d['unet_depth']}; fov_factors={d['fov_factors']}")
    json.dump(cfg, open(os.path.join(out, "config_used.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    data = T.prepare(cfg, log)
    if "train" in steps: step_train(cfg, data, log)
    if "report" in steps: step_report(cfg, data, log)
    log("done")


if __name__ == "__main__":
    main()
