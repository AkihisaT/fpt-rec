#!/usr/bin/env python
"""Four-method comparison for dataset 018: DIP (optimal = selected iteration), DIP (30 iterations), EPRY, BLIS-FPM,
each with the plane-wave and the FOV model, on the same crop, same images and same metric as the EPRY/DIP reports.

  python compare_018.py --grid 512      # centre 512 px crop
  python compare_018.py --grid 1024     # full field (vignetting mask for the FOV model, as in the 1024 reports)

Inputs (relative to the workspace): epry_pipeline/config_epry_018[_1024].json (data preparation = dip_pipeline),
dip_pipeline/dip_output_018[_1024] (DIP, best = selected iteration), dip_pipeline/dip_output_018[_1024]_it30
(DIP trained for 30 iterations, final state), epry_pipeline/epry_output_018[_1024], BLIS-FPM fptrecon nonlinear_*.npz.
Outputs: d018/out_<grid>/  fig_compare_planewave.png, fig_compare_FOV.png, compare_summary.json, DIP30 TIFFs, spoke data."""
import argparse, json, os, sys
import numpy as np
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, W + "dip_pipeline"); sys.path.insert(0, W + "epry_pipeline")
import torch; torch.set_num_threads(2)
from scipy.signal.windows import tukey
from scipy.ndimage import map_coordinates
from dipfpm import train as T, evaluate as E
from dipfpm.physics import FPMPhysics, ZERN
from eprfpm.report import export_epry
import run_epry as RE

METHODS = [("DIP_opt", "DIP (optimal it.)", "#7570b3"), ("DIP_30", "DIP (30 it.)", "#a6761d"),
           ("EPRY", "EPRY", "#1f78b4"), ("BLIS", "BLIS-FPM", "#1b9e77")]
MODELS = [(0.0, "planewave", "plane-wave model"), (1.0, "FOV_c1", "FOV model")]


def spoke_curve(phi, cy, cx, N=36, rmax=None):
    """36-fold spoke modulation and off-harmonic RMS level vs radius (unfiltered phase, physical frame)."""
    img = phi - phi.mean(); rmax = rmax or min(cy, cx, img.shape[0] - cy, img.shape[1] - cx) - 4
    r_all = np.arange(3, rmax, 0.5); th = np.linspace(0, 2 * np.pi, 2048, endpoint=False); m, nz = [], []
    for r in r_all:
        f = np.abs(np.fft.rfft(map_coordinates(img, [cy + r * np.sin(th), cx + r * np.cos(th)], order=1))) / 2048 * 2
        m.append(f[N]); nz.append(np.sqrt(np.mean(f[np.r_[N + 4:N + 30, max(N - 30, 1):N - 4]] ** 2)))
    return r_all, np.array(m), np.array(nz)


def spoke_limit(q, ratio, thr=3.0, qmin=2.0, min_span=0.3):
    """Resolution from the 36-fold spoke modulation: lowest q >= qmin from which the modulation / off-harmonic ratio
    stays below thr over a contiguous span of at least min_span um^-1.  Narrow dips at the radii of the star's
    concentric rings (continuous rings carry no 36-fold modulation) are therefore skipped."""
    o = np.argsort(q); q, ratio = np.asarray(q)[o], np.asarray(ratio)[o]
    below = (ratio < thr) & (q >= qmin)
    i = 0
    while i < len(q):
        if below[i]:
            j = i
            while j + 1 < len(q) and below[j + 1]: j += 1
            if q[j] - q[i] >= min_span or j == len(q) - 1: return float(q[i])
            i = j + 1
        else:
            i += 1
    return float("nan")


def find_centre(phi, cy0, cx0, span=40, N=36):
    radii = np.arange(40, 140, 10); th = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    def score(cy, cx):
        return np.mean([np.abs(np.fft.rfft(map_coordinates(phi, [cy + r * np.sin(th), cx + r * np.cos(th)], order=1)))[N] for r in radii])
    best = max(((score(y, x), y, x) for y in range(cy0 - span, cy0 + span + 1, 4) for x in range(cx0 - span, cx0 + span + 1, 4)))
    _, cy, cx = best
    for step in (1.0, 0.25):
        best = max(((score(y, x), y, x) for y in np.arange(cy - 4 * step, cy + 4.01 * step, step)
                    for x in np.arange(cx - 4 * step, cx + 4.01 * step, step)))
        _, cy, cx = best
    return float(cy), float(cx)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--blis", default=None, help="comma-separated BLIS-FPM npz for plane-wave,FOV (default: from config)")
    ap.add_argument("--tag", default="", help="suffix of the output folder d018/out_<grid><tag>/")
    ap.add_argument("--ds", default="018", help="dataset tag used in config / output names (e.g. 036bf)")
    ap.add_argument("--outroot", default="d018", help="workspace folder receiving out_<grid><tag>/")
    ap.add_argument("--star-centre", default=None, help="approximate Siemens-star centre 'y,x' in camera pixels of the full 1024 frame")
    ap.add_argument("--unfiltered-metric", action="store_true",
                    help="data misfit from the unfiltered objects (dark-field images need the object beyond 3.5 um^-1)")
    ap.add_argument("--dark-gain", action="store_true", help="metric with a profiled contrast factor for dark-field images "
                    "(automatic when the EPRY config sets partial_coherence)")
    a = ap.parse_args()
    sfx = "" if a.grid == 512 else f"_{a.grid}"
    os.chdir(W + "epry_pipeline")
    cfg = RE.load_cfg(f"config_epry_{a.ds}{sfx}.json", {})
    if a.blis:
        for j, f in enumerate(a.blis.split(",")): cfg["previous_native"][j]["file"] = os.path.abspath(os.path.join(W, f))
    out = W + f"{a.outroot}/out_{a.grid}{a.tag}/"; os.makedirs(out + "tiff", exist_ok=True)
    log = lambda m: print(m, flush=True)
    data = T.prepare(cfg, log)
    e = cfg["epry"]; twin = data["twin"]; dx = cfg["pixel_um"]; band = cfg["export_band"]
    keep = list(data["keep"]); nk = len(keep)
    vpos = [keep.index(k) for k in data["val"]]; tpos = [i for i in range(nk) if i not in vpos]
    common_w = None
    if e.get("eval_vignetting_mask"):
        w0 = torch.tensor(np.outer(tukey(data["M"], 0.2), tukey(data["M"], 0.2)).astype(np.float32))
        physF = FPMPhysics(data["kn"][data["keep"]], data["dk"], data["M"], data["No"], cfg["kc"], cfg["lam_um"],
                           kappa=cfg["kappa_nom"] * twin, axis_offset_um=data["axis_off"], pupil_radius=e["pupil_radius"])
        common_w = T.vignetting_weights(physF, torch.tensor(data["z0"], dtype=torch.float32), w0).numpy()
    from dipfpm.coherence import pc_modes
    md_, mw_ = pc_modes(e.get("partial_coherence"), data["kn"][data["keep"]], cfg["kc"], e["pupil_radius"])
    DG = bool(a.dark_gain or e.get("partial_coherence"))
    if md_ is not None: log(f"partial coherence {e['partial_coherence']}: modes per image mean {float((mw_ > 0).sum(1).mean()):.1f}; dark-gain metric")
    mk = lambda ks: FPMPhysics(data["kn"][data["keep"]], data["dk"], data["M"], data["No"], cfg["kc"], cfg["lam_um"], kappa=ks,
                               axis_offset_um=data["axis_off"], pupil_radius=e["pupil_radius"], modes=md_, mode_w=mw_)
    sh0 = np.zeros((nk, 2)); z0 = np.zeros(len(ZERN))
    dipd = W + f"dip_pipeline/dip_output_{a.ds}{sfx}/"; dip30d = W + f"dip_pipeline/dip_output_{a.ds}{sfx}_it30/"
    epd = W + f"epry_pipeline/epry_output_{a.ds}{sfx}/"
    R = {}                       # (method, modeltag) -> dict
    for c, mtag, mlab in MODELS:
        # DIP selected iteration and 30 iterations
        for key, d, which in [("DIP_opt", dipd, "best"), ("DIP_30", dip30d, "final")]:
            f = d + f"DIP_{mtag}_state.npz"
            if not os.path.exists(f): log(f"missing {f}"); continue
            r = T.load_state(f); b = r[which]; phys = mk(r["kappa_solver"])
            if key == "DIP_30": E.export_dip(out + "tiff", f"DIP30_{mtag}", r, data, cfg, which="final")
            R[key, mtag] = dict(a=b["a"], phi=b["phi"], dxo=r["dxo"], phys=phys, zc=b["zc"], shifts=b["shifts_um"], W_map=None,
                                zc_phys=E.zern_to_physical(b["zc"], twin), it=int(b["it"]))
        # EPRY
        tag = f"EPRY_{mtag}"; f = epd + f"{tag}_state.npz"
        if os.path.exists(f):
            r = T.load_state(f); b = r["best"]; W0 = np.load(epd + f"{tag}_W0.npy")
            ex = export_epry(out + "tiff", tag, r, twin, cfg, W0)
            R["EPRY", mtag] = dict(a=b["a"], phi=b["phi"], dxo=r["dxo"], phys=mk(r["kappa_solver"]), zc=z0, shifts=sh0,
                                   W_map=np.asarray(b["P"]), zc_phys=ex["zc_phys"], it=int(b["it"]))
        # BLIS-FPM (full-FOV solution cropped)
        p = cfg["previous_native"][0 if c == 0 else 1]["file"]
        if os.path.exists(p):
            pn = E.load_previous_native(p, data["crop"], data["H"], dx, data["M"], data["dk"])
            zfit = T.zernike_fit_map(pn["W"], data["dk"], cfg["kc"])
            R["BLIS", mtag] = dict(a=pn["a"], phi=pn["phi"], dxo=dx, phys=mk(pn["kappa_solver"]), zc=z0, shifts=sh0, W_map=pn["W"],
                                   zc_phys=E.zern_to_physical(zfit, twin), it=None)
    summary = dict(grid=a.grid, twin=int(twin), crop=[int(v) for v in data["crop"]], images_train=len(tpos), images_validation=len(vpos),
                   image_index_train=[int(keep[i]) for i in tpos], image_index_validation=[int(keep[i]) for i in vpos],
                   unfiltered_metric=bool(a.unfiltered_metric), dark_gain_metric=DG, partial_coherence=e.get("partial_coherence"),
                   band_um_inv=band, results={})
    recs_by_model = {}
    for c, mtag, mlab in MODELS:
        recs, metrics, frcs = {}, {}, {}
        for key, lab, col in METHODS:
            if (key, mtag) not in R: continue
            s = R[key, mtag]
            a_bp = E.band_pass(s["a"], s["dxo"], *band); p_bp = E.band_pass(s["phi"], s["dxo"], *band)
            a_m, p_m = (s["a"], s["phi"]) if a.unfiltered_metric else (a_bp, p_bp)
            m_all, per = E.band_misfit(s["phys"], a_m, p_m, s["zc"], s["shifts"], data["I_meas"], band, W_map=s["W_map"], weights=common_w,
                                       dark_gain=DG)
            lb = f"{lab}, {mlab}"
            recs[lb] = dict(phi=twin * p_bp, T=np.exp(-a_bp), zc_phys=s["zc_phys"], color=col,
                            phi_unf=twin * s["phi"], a_bp=a_bp)
            metrics[lb] = (float(per[tpos].mean()), float(per[vpos].mean()), col, key != "BLIS")
            summary["results"][lb] = dict(method=key, model=mtag, iteration=s["it"], band_misfit_all=float(m_all),
                                          band_misfit_training_images=float(per[tpos].mean()),
                                          band_misfit_validation_images=float(per[vpos].mean()),
                                          band_misfit_validation_per_image=[round(float(v), 4) for v in per[vpos]],
                                          band_misfit_training_per_image=[round(float(v), 4) for v in per[tpos]],
                                          zernike_rad_rms_physical={ZERN[j][2]: round(float(s["zc_phys"][j]), 4) for j in range(len(ZERN))},
                                          absorption_rms_band=float(np.sqrt(np.mean(a_bp ** 2))),
                                          phase_rms_band=float(np.sqrt(np.mean(p_bp ** 2))),
                                          phase_rms_0p3_1=float(np.sqrt(np.mean(E.band_pass(s["phi"], s["dxo"], 0.3, 1.0) ** 2))),
                                          phase_rms_1_3p5=float(np.sqrt(np.mean(E.band_pass(s["phi"], s["dxo"], 1.0, 3.5) ** 2))),
                                          corr_absorption_phase=float(np.corrcoef(a_bp.ravel(), (twin * p_bp).ravel())[0, 1]))
            log(f"{lb:40s} fitted {per[tpos].mean():.4f} | held-out {per[vpos].mean():.4f}")
        # phase agreement with DIP (selected iteration) of the same model
        m_ = 32; inner = (slice(m_, -m_), slice(m_, -m_)); ref = f"DIP (optimal it.), {mlab}"
        for lb in recs:
            if lb == ref or ref not in recs: continue
            q, f_ = E.frc(recs[lb]["phi"][inner], recs[ref]["phi"][inner], dx)
            summary["results"][lb]["phase_corr_vs_DIP_selected"] = float(np.corrcoef(recs[lb]["phi"][inner].ravel(), recs[ref]["phi"][inner].ravel())[0, 1])
            summary["results"][lb]["phase_FRC_vs_DIP_selected"] = dict(q_um_inv=np.round(q, 3).tolist(), frc=np.round(f_, 3).tolist())
            frcs[lb.split(",")[0]] = (q, f_, recs[lb]["color"], "-")
        figrecs = {lb: {k: v for k, v in r.items() if k in ("phi", "T", "zc_phys", "color")} for lb, r in recs.items()}
        vT = tuple(np.median([np.percentile(r["T"], [0.5, 99.5]) for r in figrecs.values()], 0))
        vP = tuple(np.median([np.percentile(r["phi"], [0.5, 99.5]) for r in figrecs.values()], 0))
        summary[f"grey_scale_{mtag}"] = dict(transmission=[float(v) for v in vT], phase_rad=[float(v) for v in vP])
        short_m = {lb.replace("DIP (optimal it.)", "DIP opt.").replace("DIP (30 it.)", "DIP 30 it."): v for lb, v in metrics.items()}
        E.fig_compare(out + f"fig_compare_{mtag}.png", figrecs, cfg, frcs, short_m, vT=vT, vP=vP, frc_title="vs DIP (optimal it.)",
                      n_fit=len(tpos), n_val=len(vpos))
        recs_by_model[mtag] = recs
    # ---- spoke modulation (Siemens star, 36 spokes)
    ref = recs_by_model["FOV_c1"].get("BLIS-FPM, FOV model") or next(iter(recs_by_model["FOV_c1"].values()))
    Nimg = ref["phi_unf"].shape[0]
    if a.star_centre:
        sy, sx = [float(v) for v in a.star_centre.split(",")]
        cy0, cx0 = int(round(sy - (data["crop"][0] - Nimg // 2))), int(round(sx - (data["crop"][1] - Nimg // 2)))
    else:
        cy0, cx0 = Nimg // 2, Nimg // 2
    cy, cx = find_centre(ref["phi_unf"] - ref["phi_unf"].mean(), cy0, cx0)
    spokes = dict(centre_px=[cy, cx], N=36)
    for mtag, recs in recs_by_model.items():
        for lb, r in recs.items():
            rr, m, nz = spoke_curve(r["phi_unf"], cy, cx)
            q = 36 / (2 * np.pi * rr * dx); ratio = m / nz
            cross = spoke_limit(q, ratio)
            spokes[lb] = dict(q=q.tolist(), mod=m.tolist(), offharm=nz.tolist(), q_snr3=cross)
            summary["results"][lb]["spoke_snr3_um_inv"] = cross
            log(f"{lb:40s} spoke SNR>3 limit {cross:.2f} um^-1 (half-pitch {500 / cross:.0f} nm)")
    json.dump(spokes, open(out + "spokes.json", "w"), indent=0)
    json.dump(summary, open(out + "compare_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    np.savez_compressed(out + "recs.npz", **{f"{lb}|{k}": v for recs in recs_by_model.values() for lb, r in recs.items()
                                           for k, v in r.items() if k in ("phi", "T", "phi_unf", "a_bp")})
    log(f"written {out}")


if __name__ == "__main__":
    main()
