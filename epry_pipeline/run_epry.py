#!/usr/bin/env python
"""EPRY-FPM reconstruction pipeline (Ou, Zheng & Yang 2014), comparable to the DIP and BLIS-FPM (fptrecon) results.

Prerequisites: fptrecon preprocess + calibrate (work/preprocessed.npy, calibration.json) and the dip_pipeline
package (shared data preparation, forward-model conventions and evaluation metric).

  python run_epry.py --config config_epry_003.json                    # train (all fov_factors) + report
  python run_epry.py --config config_epry_003.json --steps train --fov-factors 0
  python run_epry.py --config config_epry_003.json --steps report     # compare with DIP and fptrecon
"""
import argparse, glob, json, os, sys, time
import numpy as np


def load_cfg(path, over):
    cfg = json.load(open(path, encoding="utf-8")); base = os.path.dirname(os.path.abspath(path))
    rel = lambda p: p if (p is None or os.path.isabs(p)) else os.path.normpath(os.path.join(base, p))
    for k in ("fpt_work_dir", "dip_pipeline_dir", "dip_results_dir", "output_dir"):
        cfg[k] = rel(cfg.get(k))
    for p in cfg.get("previous_native", []):
        p["file"] = rel(p["file"])
    for k, v in over.items():
        if v is not None: cfg["epry"][k] = v
    lam = 1.239842 / cfg["energy_keV"] * 1e-3
    cfg.update(lam_um=lam, kc=1 / (2 * cfg["fzp_outer_zone_um"]), kappa_nom=1.0 / (lam * cfg["sample_objective_distance_m"] * 1e6))
    e = cfg["epry"]
    cfg["dip"] = dict(crop_px=e["crop_px"], crop_center=e["crop_center"], max_k_over_kc=e["max_k_over_kc"],
                      val_every=e["val_every"], M=e["M"], No=e["No"], unet_px=64, pupil_radius=e["pupil_radius"],
                      dark_mode=bool(e.get("dark_mode", False)), exclude_images=e.get("exclude_images", []),
                      partial_coherence=e.get("partial_coherence"))
    return cfg


def tag_of(c):
    return "EPRY_planewave" if c == 0 else f"EPRY_FOV_c{c:g}"


def initial_pupil(z0, M, dk, kc, R, edge):
    from scipy.special import expit
    from dipfpm.physics import ZERN, zernike_nm
    yy, xx = np.mgrid[-(M // 2):M - M // 2, -(M // 2):M - M // 2] * dk
    rho = np.hypot(yy, xx) / kc; th = np.arctan2(yy, xx)
    W0 = sum(z * zernike_nm(n, m, np.minimum(rho, 1), th) for z, (n, m, _) in zip(z0, ZERN))
    return expit((R - np.hypot(yy, xx)) / edge) * np.exp(1j * W0), W0


def step_train(cfg, data, log):
    from eprfpm.core import EPRY, run_epry
    from eprfpm.report import pupil_phase_unwrapped, zernike_fit
    from dipfpm import train as T
    e = cfg["epry"]; out = cfg["output_dir"]; keep = list(data["keep"])
    tr = [keep.index(k) for k in data["train"]]; va = [keep.index(k) for k in data["val"]]
    kabs = np.hypot(*data["kn"][data["keep"]].T)
    from dipfpm.coherence import pc_modes
    md, mw = pc_modes(e.get("partial_coherence"), data["kn"][data["keep"]], cfg["kc"], e["pupil_radius"])
    for c in e["fov_factors"]:
        tag = tag_of(c); ks = c * cfg["kappa_nom"] * data["twin"]
        P0, W0 = initial_pupil(data["z0"], data["M"], data["dk"], cfg["kc"], e["pupil_radius"] * cfg["kc"], 0.25 * data["dk"])
        eng = EPRY(data["I_meas"], data["kn"][data["keep"]], data["dk"], data["M"], data["No"], cfg["kc"], P0, kappa=ks,
                   axis_offset_um=data["axis_off"], pupil_radius=e["pupil_radius"], replace_tukey=e["replace_tukey"],
                   loss_tukey=e["loss_tukey"], band=cfg["export_band"], threads=e["threads"],
                   init_noise=float(e.get("init_noise", 0.0)), seed=int(e.get("seed", 0)),
                   mask_vignetted=bool(e.get("mask_vignetted", False)), pupil_gamma=float(e.get("pupil_gamma", 0.0)),
                   wmask=data.get("wmask"), modes=md, mode_w=mw)
        if md is not None: log(f"  partial coherence {e['partial_coherence']}: modes per image mean {float((mw > 0).sum(1).mean()):.1f}")
        if eng.masked_fraction: log(f"  vignetting mask: {eng.masked_fraction:.3f} of the pixels excluded")
        log(f"== EPRY {tag}: kappa_solver={ks:+.5f} um^-2, {len(tr)} images updated, {len(va)} held out; M={data['M']}, No={data['No']}")
        t0 = time.time()
        res = run_epry(eng, tr, va, e, kabs, log=log)
        for grp in ("best", "final"):
            res[grp]["zc"] = zernike_fit(pupil_phase_unwrapped(res[grp]["P"], W0), data["dk"], cfg["kc"])
        res.update(kappa_solver=ks, M=data["M"], No=data["No"], dk=data["dk"], dxo=eng.dxo)
        res["hist"]["t"] = res["hist"]["t"]
        T.save_state(os.path.join(out, f"{tag}_state.npz"), res)
        np.save(os.path.join(out, f"{tag}_W0.npy"), W0)
        log(f"{tag}: selected sweep {res['best']['it']} of {res['n_done']} (held-out misfit {res['best']['val_band']:.4f}), "
            f"{time.time() - t0:.0f} s")


def step_report(cfg, data, log):
    import torch
    from scipy.signal.windows import tukey
    from dipfpm import train as T, evaluate as E
    from dipfpm.physics import FPMPhysics, ZERN
    from eprfpm.report import export_epry, fig_sweeps, fig_pupils
    out = cfg["output_dir"]; e = cfg["epry"]; twin = data["twin"]; dx = cfg["pixel_um"]; band = cfg["export_band"]
    keep = list(data["keep"]); nk = len(keep)
    vpos = [keep.index(k) for k in data["val"]]; tpos = [i for i in range(nk) if i not in vpos]
    common_w = None
    if e.get("eval_vignetting_mask"):
        w0 = torch.tensor(np.outer(tukey(data["M"], 0.2), tukey(data["M"], 0.2)).astype(np.float32))
        physF = FPMPhysics(data["kn"][data["keep"]], data["dk"], data["M"], data["No"], cfg["kc"], cfg["lam_um"],
                           kappa=cfg["kappa_nom"] * twin, axis_offset_um=data["axis_off"], pupil_radius=e["pupil_radius"])
        common_w = T.vignetting_weights(physF, torch.tensor(data["z0"], dtype=torch.float32), w0).numpy()
    DG = bool(e.get("partial_coherence"))
    BM = lambda *a, **k: E.band_misfit(*a, weights=common_w, dark_gain=DG, **k)
    from dipfpm.coherence import pc_modes
    md_, mw_ = pc_modes(e.get("partial_coherence"), data["kn"][data["keep"]], cfg["kc"], e["pupil_radius"])
    mk = lambda ks: FPMPhysics(data["kn"][data["keep"]], data["dk"], data["M"], data["No"], cfg["kc"], cfg["lam_um"], kappa=ks,
                               axis_offset_um=data["axis_off"], pupil_radius=e["pupil_radius"], modes=md_, mode_w=mw_)
    STY = {"EPRY_planewave": ("EPRY, plane-wave model", "#1f78b4"), "EPRY_FOV_c1": ("EPRY, FOV model", "#e6ab02"),
           "DIP_planewave": ("DIP, plane-wave model", "#7570b3"), "DIP_FOV_c1": ("DIP, FOV model", "#e7298a"),
           "prev_plane": ("BLIS-FPM, plane-wave model", "#d95f02"), "prev_fov": ("BLIS-FPM, FOV model", "#1b9e77")}
    recs, metrics, summary, res_by, pupils = {}, {}, {}, {}, {}
    sh0 = np.zeros((nk, 2)); z0 = np.zeros(len(ZERN))

    def add(key, a_bp_s, p_bp_s, zc, phys, T_img, phi_phys, zc_phys, extra, **kw):
        m_all, per = BM(phys, a_bp_s, p_bp_s, zc, kw.pop("shifts", sh0), data["I_meas"], band, **kw)
        lb, col = STY[key]
        metrics[lb] = (float(per[tpos].mean()), float(per[vpos].mean()), col, not key.startswith("prev"))
        recs[lb] = dict(phi=phi_phys, T=T_img, zc_phys=zc_phys, color=col)
        summary[lb] = dict(band_misfit_all=m_all, band_misfit_training_images=float(per[tpos].mean()),
                           band_misfit_validation_images=float(per[vpos].mean()), **extra)
        log(f"{lb}: misfit all {m_all:.4f} | fitted {per[tpos].mean():.4f} | held-out {per[vpos].mean():.4f}")
    # ---- EPRY
    for c in [0.0, 1.0]:
        tag = tag_of(c); f = os.path.join(out, f"{tag}_state.npz")
        if not os.path.exists(f): continue
        res = T.load_state(f); res_by[tag] = res; W0 = np.load(os.path.join(out, f"{tag}_W0.npy"))
        ex = export_epry(out, tag, res, twin, cfg, W0)
        b = res["best"]; phys = mk(res["kappa_solver"])
        a_bp = E.band_pass(b["a"], res["dxo"], *band); p_bp = E.band_pass(b["phi"], res["dxo"], *band)
        m_z, _ = BM(phys, a_bp, p_bp, b["zc"], sh0, data["I_meas"], band)
        fin = res["final"]
        m_f, _ = BM(phys, E.band_pass(fin["a"], res["dxo"], *band), E.band_pass(fin["phi"], res["dxo"], *band), z0, sh0,
                    data["I_meas"], band, W_map=np.asarray(fin["P"]))
        add(tag, a_bp, p_bp, z0, phys, ex["T"], ex["phi"], ex["zc_phys"],
            dict(selected_sweep=int(b["it"]), sweeps_run=int(res["n_done"]), band_misfit_12zernike_pupil=m_z,
                 band_misfit_final_sweep=m_f, seconds_total=float(res["hist"]["t"][-1]),
                 zernike_rad_rms_physical={ZERN[j][2]: round(float(ex["zc_phys"][j]), 4) for j in range(len(ZERN))},
                 intensity_factor_range=[float(np.min(b["g"])), float(np.max(b["g"]))]),
            W_map=np.asarray(b["P"]))
        pupils[STY[tag][0]] = (ex["W_phys"], ex["A_phys"], res["dk"], STY[tag][1])
    # ---- DIP (same crop / same data preparation)
    dd = cfg.get("dip_results_dir")
    for tag in ["DIP_planewave", "DIP_FOV_c1"]:
        f = os.path.join(dd, f"{tag}_state.npz") if dd else ""
        if not os.path.exists(f): continue
        r = T.load_state(f); b = r["best"]; phys = mk(r["kappa_solver"])
        a_bp = E.band_pass(b["a"], r["dxo"], *band); p_bp = E.band_pass(b["phi"], r["dxo"], *band)
        add(tag, a_bp, p_bp, b["zc"], phys, np.exp(-a_bp), twin * p_bp, E.zern_to_physical(b["zc"], twin),
            dict(selected_iteration=int(b["it"])), shifts=b["shifts_um"])
    # ---- BLIS-FPM (fptrecon full-FOV solution, cropped)
    for p, key in zip(cfg.get("previous_native", []), ["prev_plane", "prev_fov"]):
        if not os.path.exists(p["file"]): continue
        pn = E.load_previous_native(p["file"], data["crop"], data["H"], dx, data["M"], data["dk"])
        phys = mk(pn["kappa_solver"])
        a_bp = E.band_pass(pn["a"], dx, *band); p_bp = E.band_pass(pn["phi"], dx, *band)
        zfit = T.zernike_fit_map(pn["W"], data["dk"], cfg["kc"])
        add(key, a_bp, p_bp, z0, phys, np.exp(-a_bp), twin * p_bp, E.zern_to_physical(zfit, twin), dict(file=os.path.basename(p["file"])),
            W_map=pn["W"])
    # ---- agreement between methods (phase FRC, same model)
    m_ = 32; inner = (slice(m_, -m_), slice(m_, -m_)); frcs = {}
    for tag in ["EPRY_planewave", "EPRY_FOV_c1"]:
        lb = STY[tag][0]
        if lb not in recs: continue
        fov = "FOV" in tag
        for other in (["DIP_FOV_c1", "prev_fov"] if fov else ["DIP_planewave", "prev_plane"]):
            ol = STY[other][0]
            if ol not in recs: continue
            q, f_ = E.frc(recs[lb]["phi"][inner], recs[ol]["phi"][inner], dx)
            summary[lb][f"phase_corr_vs_{ol}"] = float(np.corrcoef(recs[lb]["phi"][inner].ravel(), recs[ol]["phi"][inner].ravel())[0, 1])
            summary[lb][f"phase_FRC_vs_{ol}"] = dict(q_um_inv=np.round(q, 3).tolist(), frc=np.round(f_, 3).tolist())
            frcs[f"{'FOV' if fov else 'plane'}: vs {ol.split(',')[0]}"] = (q, f_, STY[other][1], "--" if fov else "-")
    E.fig_compare(os.path.join(out, "fig_epry_comparison.png"), recs, cfg, frcs, metrics, frc_title="EPRY vs",
                  n_fit=len(tpos), n_val=len(vpos))
    if res_by:
        fig_sweeps(os.path.join(out, "fig_epry_sweeps.png"), res_by, {t: (STY[t][0].replace("EPRY, ", ""), STY[t][1]) for t in res_by},
                   n_fit=len(tpos), n_val=len(vpos))
    if pupils:
        fig_pupils(os.path.join(out, "fig_epry_pupil.png"), pupils, cfg["kc"])
    json.dump(summary, open(os.path.join(out, "epry_summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"report written to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--steps", default="train,report")
    ap.add_argument("--n-sweeps", type=int); ap.add_argument("--threads", type=int); ap.add_argument("--fov-factors")
    ap.add_argument("--output-dir"); ap.add_argument("--fpt-work-dir")
    ap.add_argument("--init-noise", type=float); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    cfg = load_cfg(a.config, {"n_sweeps": a.n_sweeps, "threads": a.threads, "init_noise": a.init_noise, "seed": a.seed,
                              "fov_factors": [float(v) for v in a.fov_factors.split(",")] if a.fov_factors else None})
    if a.output_dir: cfg["output_dir"] = os.path.abspath(a.output_dir)
    if a.fpt_work_dir: cfg["fpt_work_dir"] = os.path.abspath(a.fpt_work_dir)
    sys.path.insert(0, cfg["dip_pipeline_dir"]); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dipfpm import train as T
    out = cfg["output_dir"]; os.makedirs(out, exist_ok=True); steps = a.steps.split(",")
    logf = open(os.path.join(out, f"epry_{'_'.join(steps)}.log"), "a"); t00 = time.time()

    def log(m):
        line = f"[{time.time() - t00:7.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    log(f"config {a.config}; steps {steps}; epry {json.dumps({k: v for k, v in cfg['epry'].items() if not k.startswith('_')})}")
    data = T.prepare(cfg, log)
    if "train" in steps: step_train(cfg, data, log)
    if "report" in steps: step_report(cfg, data, log)
    log("done")


if __name__ == "__main__":
    main()
