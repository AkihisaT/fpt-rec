#!/usr/bin/env python
"""Four-method comparison for 32a (DIP optimal iteration, DIP 30 iterations, EPRY, BLIS-FPM) x (plane-wave, FOV c = 1),
same crop, same images, same metric; all images shown and exported WITHOUT band filtering.

  python d32a/compare_32a.py --ds 32abf --grid 512        # bright field, centre 512 px
  python d32a/compare_32a.py --ds 32abf --grid 1000       # bright field, full field 1000 px
  python d32a/compare_32a.py --ds 32adf --grid 512 ...    # rings 0-2 + dark-field rings 6-7

Inputs: epry_pipeline/config_epry_<ds>[_1000].json (data preparation = dip_pipeline), dip_output_<ds>[_1000] (best =
selected iteration), dip_output_<ds>[_1000]_it30 (final state), epry_output_<ds>[_1000], BLIS-FPM npz from the config
(previous_native). Metric: dipfpm.evaluate.band_misfit (0.3-3.5 um^-1 intensity band) with the UNFILTERED objects
(primary; the model prediction of each method as reconstructed) and with 0.3-3.5 um^-1 band-passed objects (secondary,
project convention of 003/018). Output d32a/out_<ds>_<grid>/: compare_summary.json, spokes.json, recs.npz,
tiff/<method>_<model>_{transmission,phase_rad,pupil_phase_rad}.tif (physical frame, unfiltered)."""
import argparse, json, os, sys
import numpy as np
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, W + "dip_pipeline"); sys.path.insert(0, W + "epry_pipeline")
import torch; torch.set_num_threads(4)
import tifffile
from scipy.signal.windows import tukey
from scipy.ndimage import map_coordinates
from dipfpm import train as T, evaluate as E
from dipfpm.physics import FPMPhysics, ZERN
from eprfpm.report import pupil_phase_unwrapped
sys.path.insert(0, W + "fpt_pipeline")
from fptrecon import posaffine as PA
os.chdir(W + "epry_pipeline"); sys.path.insert(0, os.getcwd())
import run_epry as RE

METHODS = [("DIP_opt", "DIP (optimal it.)", "#7570b3"), ("DIP_30", "DIP (30 it.)", "#a6761d"),
           ("EPRY", "EPRY", "#1f78b4"), ("BLIS", "BLIS-FPM", "#1b9e77")]
MODELS = [(0.0, "planewave", "plane-wave model"), (1.0, "FOV_c1", "FOV model")]

R_SPLIT = 85.0      # px: inside this radius the spoke centre of the reconstructions is offset by ~5 px from the outer one

def spoke_curve(phi, cy, cx, N=36, rmax=None, inner=None):
    """36-fold modulation and off-harmonic level vs radius. inner = (cy, cx) centre used for r < R_SPLIT (two-zone centre)."""
    img = phi - phi.mean(); rmax = rmax or min(cy, cx, img.shape[0] - cy, img.shape[1] - cx) - 4
    r_all = np.arange(3, rmax, 0.5); th = np.linspace(0, 2 * np.pi, 2048, endpoint=False); m, nz = [], []
    for r in r_all:
        yc, xc = inner if (inner is not None and r < R_SPLIT) else (cy, cx)
        f = np.abs(np.fft.rfft(map_coordinates(img, [yc + r * np.sin(th), xc + r * np.cos(th)], order=1))) / 2048 * 2
        m.append(f[N]); nz.append(np.sqrt(np.mean(f[np.r_[N + 4:N + 30, max(N - 30, 1):N - 4]] ** 2)))
    return r_all, np.array(m), np.array(nz)

def spoke_limit(q, ratio, thr=3.0, qmin=2.0, min_span=0.3):
    o = np.argsort(q); q, ratio = np.asarray(q)[o], np.asarray(ratio)[o]
    below = (ratio < thr) & (q >= qmin); i = 0
    while i < len(q):
        if below[i]:
            j = i
            while j + 1 < len(q) and below[j + 1]: j += 1
            if q[j] - q[i] >= min_span or j == len(q) - 1: return float(q[i])
            i = j + 1
        else:
            i += 1
    return float("nan")

def find_centre(phi, cy0, cx0, span=40, N=36, radii=None):
    radii = np.arange(95, 165, 10) if radii is None else radii; th = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    score = lambda cy, cx: np.mean([np.abs(np.fft.rfft(map_coordinates(phi, [cy + r * np.sin(th), cx + r * np.cos(th)], order=1)))[N] for r in radii])
    _, cy, cx = max(((score(y, x), y, x) for y in range(cy0 - span, cy0 + span + 1, 4) for x in range(cx0 - span, cx0 + span + 1, 4)))
    for step in (1.0, 0.25):
        _, cy, cx = max(((score(y, x), y, x) for y in np.arange(cy - 4 * step, cy + 4.01 * step, step) for x in np.arange(cx - 4 * step, cx + 4.01 * step, step)))
    return float(cy), float(cx)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grid", type=int, default=512); ap.add_argument("--ds", default="32abf")
    ap.add_argument("--star-centre", default="470,499", help="Siemens-star centre 'y,x' in camera pixels of the full 1000 frame")
    ap.add_argument("--blis", default=None, help="comma-separated BLIS-FPM npz (plane, FOV), relative to the workspace")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sfx = "" if a.grid == 512 else f"_{a.grid}"
    cfg = RE.load_cfg(f"config_epry_{a.ds}{sfx}.json", {})
    if a.blis:
        for j, f in enumerate(a.blis.split(",")): cfg["previous_native"][j]["file"] = os.path.abspath(os.path.join(W, f))
    out = W + f"d32a/out_{a.ds}_{a.grid}{a.tag}/"; os.makedirs(out + "tiff", exist_ok=True)
    log = lambda m: print(m, flush=True)
    data = T.prepare(cfg, log)
    e = cfg["epry"]; twin = data["twin"]; dx = cfg["pixel_um"]; band = cfg["export_band"]
    cfg_lam = 1.239842 / cfg["energy_keV"] * 1e-3; kc_ = 1.0 / (2 * cfg["fzp_outer_zone_um"])     # kc = NA / lambda = 1 / (2 dr)
    keep = list(data["keep"]); nk = len(keep)
    vpos = [keep.index(k) for k in data["val"]]; tpos = [i for i in range(nk) if i not in vpos]
    dark_k = np.asarray(data["dark"]) if data.get("dark") is not None else np.zeros(nk, bool)
    common_w = None
    if e.get("eval_vignetting_mask"):
        w0 = torch.tensor(np.outer(tukey(data["M"], 0.2), tukey(data["M"], 0.2)).astype(np.float32))
        physF = FPMPhysics(data["kn"][data["keep"]], data["dk"], data["M"], data["No"], cfg["kc"], cfg["lam_um"],
                           kappa=cfg["kappa_nom"] * twin, axis_offset_um=data["axis_off"], pupil_radius=e["pupil_radius"])
        common_w = T.vignetting_weights(physF, torch.tensor(data["z0"], dtype=torch.float32), w0).numpy()
    mk = lambda ks: FPMPhysics(data["kn"][data["keep"]], data["dk"], data["M"], data["No"], cfg["kc"], cfg["lam_um"], kappa=ks,
                               axis_offset_um=data["axis_off"], pupil_radius=e["pupil_radius"])
    sh0 = np.zeros((nk, 2)); z0 = np.zeros(len(ZERN))
    M, dk, kc = data["M"], data["dk"], cfg["kc"]
    yy, xx = np.mgrid[-(M // 2):M - M // 2, -(M // 2):M - M // 2] * dk; inside = np.hypot(yy, xx) <= kc
    dipd = W + f"dip_pipeline/dip_output_{a.ds}{sfx}/"; dip30d = W + f"dip_pipeline/dip_output_{a.ds}{sfx}_it30/"
    epd = W + f"epry_pipeline/epry_output_{a.ds}{sfx}/"
    R = {}
    for c, mtag, mlab in MODELS:
        for key, d, which in [("DIP_opt", dipd, "best"), ("DIP_30", dip30d, "final")]:
            f = d + f"DIP_{mtag}_state.npz"
            if not os.path.exists(f): log(f"missing {f}"); continue
            r = T.load_state(f); b = r[which]
            Wp = np.nan_to_num(E.pupil_map(b["zc"], M, dk, kc))
            R[key, mtag] = dict(a=b["a"], phi=b["phi"], dxo=r["dxo"], phys=mk(r["kappa_solver"]), zc=b["zc"], shifts=b["shifts_um"], W_map=None,
                                zc_phys=E.zern_to_physical(b["zc"], twin), it=int(b["it"]), W_phys=np.where(inside, Wp, np.nan))
        tag = f"EPRY_{mtag}"; f = epd + f"{tag}_state.npz"
        if os.path.exists(f):
            r = T.load_state(f); b = r["best"]; W0 = np.load(epd + f"{tag}_W0.npy")
            Wu = pupil_phase_unwrapped(np.asarray(b["P"]), W0)
            from eprfpm.report import zernike_fit
            R["EPRY", mtag] = dict(a=b["a"], phi=b["phi"], dxo=r["dxo"], phys=mk(r["kappa_solver"]), zc=z0, shifts=sh0,
                                   W_map=np.asarray(b["P"]), zc_phys=zernike_fit(Wu, dk, kc), it=int(b["it"]), W_phys=np.where(inside, Wu, np.nan))
        else: log(f"missing {f}")
        p = cfg["previous_native"][0 if c == 0 else 1]["file"]
        if os.path.exists(p):
            pn = E.load_previous_native(p, data["crop"], data["H"], dx, M, dk)
            zfit = T.zernike_fit_map(pn["W"], dk, kc)
            R["BLIS", mtag] = dict(a=pn["a"], phi=pn["phi"], dxo=dx, phys=mk(pn["kappa_solver"]), zc=z0, shifts=sh0, W_map=pn["W"],
                                   zc_phys=E.zern_to_physical(zfit, twin), it=None, W_phys=np.where(inside, pn["W"], np.nan))
        else: log(f"missing {p}")
    summary = dict(ds=a.ds, grid=a.grid, twin=int(twin), crop=[int(v) for v in data["crop"]], images_train=len(tpos), images_validation=len(vpos),
                   image_index_train=[int(keep[i]) for i in tpos], image_index_validation=[int(keep[i]) for i in vpos],
                   dark_images_train=int(dark_k[tpos].sum()), dark_images_validation=int(dark_k[vpos].sum()),
                   metric="band misfit 0.3-3.5 um^-1 of the flat-fielded intensity (dipfpm.evaluate.band_misfit); 'unfiltered' = objects as reconstructed, "
                          "'bandpassed' = objects band-passed to 0.3-3.5 um^-1 before the forward model", band_um_inv=band, results={})
    recs_by_model = {}
    # ---- per-reconstruction consistency check (posaffine): drift predicted by each pupil vs drift kept in the data
    wd = os.path.normpath(os.path.join(W + "epry_pipeline", cfg["fpt_work_dir"])) + "/"
    calw = json.load(open(wd + "calibration.json")); PAc = calw.get("position_affine")
    if PAc is not None:
        if "positions_index" in calw and os.path.exists(W + "d32a/data_reg/positions_um.csv"):
            Pall = np.loadtxt(W + "d32a/data_reg/positions_um.csv", delimiter=","); pos_w = Pall[np.array(calw["positions_index"]), 1:3]
        else:
            pos_w = np.load(wd + "preprocess_meta.npz")["positions"]
        pos_k = np.asarray(pos_w, float)[np.asarray(keep)]; kn_k = np.asarray(data["kn"])[np.asarray(keep)]
        bright_k = ~dark_k
        Pm = np.array(PAc["image_frame_permutation"]); Bret = np.array(PAc["B_pupil_px_per_stage"])
        to_E = lambda B: (np.asarray(B) @ Pm.T) * dx
    def pa_check(s):
        if PAc is None: return None
        Wm = np.asarray(s["W_phys"], float); Pamp = np.isfinite(Wm).astype(float)
        poly = PA.pupil_poly(np.nan_to_num(Wm), Pamp, data["dk"], kc_, order=4, centre_exclude=0.25)
        sel = bright_k
        Bp = PA.fit_drift(PA.pupil_shifts_px(poly, kn_k[sel], dx), pos_k[sel], kn_k[sel])[0]
        No_ = s["a"].shape[0]; h = min(No_, 512) // 2; c0 = No_ // 2
        af = PA.autofocus(np.asarray(s["a"], float)[c0 - h:c0 + h, c0 - h:c0 + h], np.asarray(s["phi"], float)[c0 - h:c0 + h, c0 - h:c0 + h],
                          float(s["dxo"]), cfg_lam, band=tuple(band), frac=0.35)
        polyT = PA.add_quadratic(poly, af["z_um"], af["astig_um"], cfg_lam, sign=-1.0)
        BT = PA.fit_drift(PA.pupil_shifts_px(polyT, kn_k[sel], dx), pos_k[sel], kn_k[sel])[0]
        zP, aP = PA.defocus_astig_from_poly(poly, cfg_lam); zT, aT = PA.defocus_astig_from_poly(polyT, cfg_lam)
        d = dict(pupil_defocus_um=zP, pupil_astig_um=aP, object_refocus=af, total_defocus_um=zT, total_astig_um=aT,
                 pupil_drift_minus_kept_pct=float(np.abs(to_E(Bp) - to_E(Bret)).max() * 100),
                 total_drift_minus_kept_pct=float(np.abs(to_E(BT) - to_E(Bret)).max() * 100))
        sh = np.asarray(s["shifts"], float)
        if np.abs(sh).max() > 0:                             # DIP learned image shifts (um): linear trend should be ~0
            Bs = PA.fit_drift(sh[sel] / dx, pos_k[sel], kn_k[sel], higher=False)[0]
            d["learned_shift_linear_trend_pct"] = float(np.abs(to_E(Bs)).max() * 100); d["learned_shift_rms_px"] = (np.sqrt((sh[sel] ** 2).mean(0)) / dx).tolist()
        # direct check (independent of how the quadratic phase is split): shift of each measured image relative to the
        # model image of this reconstruction (pupil + object + learned shifts); its linear trend should be ~0
        import torch as _t
        from scipy.signal.windows import tukey as _tk
        with _t.no_grad():
            Im = s["phys"](_t.tensor(s["a"], dtype=_t.float32), _t.tensor(s["phi"], dtype=_t.float32), _t.tensor(s["zc"], dtype=_t.float32),
                           _t.tensor(np.asarray(s["shifts"], float), dtype=_t.float32), W_map=s["W_map"]).numpy()
        Mg = Im.shape[-1]; qg = np.fft.fftfreq(Mg, 1 / (Mg * data["dk"])); QRg = np.hypot(*np.meshgrid(qg, qg, indexing="ij"))
        bmg = (QRg >= band[0]) & (QRg <= band[1]); wg = np.outer(_tk(Mg, 0.3), _tk(Mg, 0.3)); up = 4; Mu = Mg * up
        def _xs(A, B):
            X = np.conj(np.fft.fft2((A - A.mean()) * wg) * bmg) * (np.fft.fft2((B - B.mean()) * wg) * bmg)
            Cc = np.zeros((Mu, Mu), complex); c0_ = Mu // 2 - Mg // 2; Cc[c0_:c0_ + Mg, c0_:c0_ + Mg] = np.fft.fftshift(X)
            cc = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(Cc)))); iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
            par = lambda m, o, p: 0.5 * (m - p) / (m - 2 * o + p)
            return np.array([iy + par(cc[iy - 1, ix], cc[iy, ix], cc[iy + 1, ix]) - Mu // 2, ix + par(cc[iy, ix - 1], cc[iy, ix], cc[iy, ix + 1]) - Mu // 2]) / up
        crop_px = float(data["crop"][2])
        msh = np.array([_xs(Im[j] / Im[j].mean(), np.asarray(data["I_meas"][j], float)) for j in np.where(sel)[0]]) * (crop_px / Mg)
        Bm = PA.fit_drift(msh, pos_k[sel], kn_k[sel], higher=False)[0]
        d["model_residual_linear_drift_pct"] = float(np.abs(to_E(Bm)).max() * 100)
        d["model_residual_linear_drift"] = PA.decompose(to_E(Bm))
        d["model_shift_rms_px"] = np.sqrt(((msh - msh.mean(0)) ** 2).mean(0)).tolist()
        d["contrast_focus_minus_kept_pct"] = d["total_drift_minus_kept_pct"]
        d["ok_model_drift"] = bool(d["model_residual_linear_drift_pct"] < 0.1)
        d["ok"] = bool(d["total_drift_minus_kept_pct"] < 0.1)
        return d
    for c, mtag, mlab in MODELS:
        recs = {}
        for key, lab, col in METHODS:
            if (key, mtag) not in R: continue
            s = R[key, mtag]
            a_bp = E.band_pass(s["a"], s["dxo"], *band); p_bp = E.band_pass(s["phi"], s["dxo"], *band)
            res = {}
            for mname, (am, pm) in (("unfiltered", (s["a"], s["phi"])), ("bandpassed", (a_bp, p_bp))):
                m_all, per = E.band_misfit(s["phys"], am, pm, s["zc"], s["shifts"], data["I_meas"], band, W_map=s["W_map"], weights=common_w)
                res[mname] = per
            lb = f"{lab}, {mlab}"
            phi_u = twin * s["phi"]; T_u = np.exp(-s["a"])
            recs[lb] = dict(phi_unf=phi_u, T_unf=T_u, W_phys=s["W_phys"], color=col)
            pu, pb = res["unfiltered"], res["bandpassed"]
            def grp(per, idx):
                idx = np.asarray(idx, int)
                d_ = dict(all=float(per[idx].mean()) if len(idx) else None)
                if dark_k.any():
                    ib = idx[~dark_k[idx]]; id_ = idx[dark_k[idx]]
                    d_.update(bright=float(per[ib].mean()) if len(ib) else None, dark=float(per[id_].mean()) if len(id_) else None)
                return d_
            summary["results"][lb] = dict(method=key, model=mtag, iteration=s["it"],
                                          misfit_unfiltered=dict(train=grp(pu, tpos), heldout=grp(pu, vpos)),
                                          misfit_bandpassed=dict(train=grp(pb, tpos), heldout=grp(pb, vpos)),
                                          heldout_per_image_unfiltered=[round(float(v), 4) for v in pu[vpos]],
                                          zernike_rad_rms_physical={ZERN[j][2]: round(float(s["zc_phys"][j]), 4) for j in range(len(ZERN))},
                                          phase_rms_unfiltered=float(np.std(phi_u)), absorption_rms_unfiltered=float(np.std(s["a"])),
                                          phase_rms_0p3_3p5=float(np.std(p_bp)), phase_rms_above_3p5=float(np.std(phi_u - E.band_pass(phi_u, s["dxo"], 0.0, 3.5))),
                                          corr_absorption_phase_bandpassed=float(np.corrcoef(a_bp.ravel(), (twin * p_bp).ravel())[0, 1]),
                                          pupil_drift_check=pa_check(s))
            if summary["results"][lb]["pupil_drift_check"]:
                ck = summary["results"][lb]["pupil_drift_check"]
                log(f"{lb:40s} drift check: pupil z {ck['pupil_defocus_um']/1e3:+.2f} mm astig ({ck['pupil_astig_um'][0]/1e3:+.2f},{ck['pupil_astig_um'][1]/1e3:+.2f}); "
                    f"object refocus z {ck['object_refocus']['z_um']/1e3:+.2f} mm; total z {ck['total_defocus_um']/1e3:+.2f} mm astig ({ck['total_astig_um'][0]/1e3:+.2f},{ck['total_astig_um'][1]/1e3:+.2f}); "
                    f"contrast-focus drift - kept {ck['total_drift_minus_kept_pct']:.3f} % -> {'OK' if ck['ok'] else 'WARNING'}; pupil-only drift - kept {ck['pupil_drift_minus_kept_pct']:.3f} %; "
                    f"model residual linear drift {ck['model_residual_linear_drift_pct']:.3f} % -> {'OK' if ck['ok_model_drift'] else 'WARNING'}"
                    + (f"; learned-shift trend {ck['learned_shift_linear_trend_pct']:.3f} %" if 'learned_shift_linear_trend_pct' in ck else ""))
            log(f"{lb:40s} unfiltered: train {pu[tpos].mean():.4f} | held-out {pu[vpos].mean():.4f}   (band-passed objects: {pb[tpos].mean():.4f} | {pb[vpos].mean():.4f})")
            tags = dict(resolution=(1 / s["dxo"], 1 / s["dxo"]), metadata={"unit": "um"})
            base = out + f"tiff/{key}_{mtag}"
            tifffile.imwrite(base + "_phase_rad.tif", phi_u.astype(np.float32), **tags)
            tifffile.imwrite(base + "_transmission.tif", T_u.astype(np.float32), **tags)
            tifffile.imwrite(base + "_pupil_phase_rad.tif", np.nan_to_num(s["W_phys"]).astype(np.float32))
        m_ = 32; inner = (slice(m_, -m_), slice(m_, -m_)); ref = f"DIP (optimal it.), {mlab}"
        for lb in recs:
            if lb == ref or ref not in recs: continue
            q, f_ = E.frc(recs[lb]["phi_unf"][inner], recs[ref]["phi_unf"][inner], dx)
            summary["results"][lb]["phase_FRC_vs_DIP_selected_unfiltered"] = dict(q_um_inv=np.round(q, 3).tolist(), frc=np.round(f_, 3).tolist())
            summary["results"][lb]["phase_corr_vs_DIP_selected_unfiltered"] = float(np.corrcoef(recs[lb]["phi_unf"][inner].ravel(), recs[ref]["phi_unf"][inner].ravel())[0, 1])
        recs_by_model[mtag] = recs
    # spokes (unfiltered phase)
    ref = recs_by_model["FOV_c1"].get("BLIS-FPM, FOV model") or next(iter(recs_by_model["FOV_c1"].values()))
    Nimg = ref["phi_unf"].shape[0]
    sy, sx = [float(v) for v in a.star_centre.split(",")]
    cy0, cx0 = int(round(sy - (data["crop"][0] - Nimg // 2))), int(round(sx - (data["crop"][1] - Nimg // 2)))
    cy, cx = find_centre(ref["phi_unf"] - ref["phi_unf"].mean(), cy0, cx0)
    cyi, cxi = find_centre(ref["phi_unf"] - ref["phi_unf"].mean(), int(round(cy)), int(round(cx)), span=12, radii=np.arange(45, 81, 5))
    spokes = dict(centre_px=[cy, cx], centre_inner_px=[cyi, cxi], r_split_px=R_SPLIT, N=36)
    for mtag, recs in recs_by_model.items():
        for lb, r in recs.items():
            rr, m, nz = spoke_curve(r["phi_unf"], cy, cx, inner=(cyi, cxi))
            q = 36 / (2 * np.pi * rr * dx); cross = spoke_limit(q, m / nz)
            spokes[lb] = dict(q=q.tolist(), mod=m.tolist(), offharm=nz.tolist(), q_snr3=cross)
            summary["results"][lb]["spoke_snr3_um_inv"] = cross
            log(f"{lb:40s} spoke SNR>3 limit {cross:.2f} um^-1 (half-pitch {500 / cross:.0f} nm)")
    summary["star_centre_crop_px"] = [cy, cx]; summary["star_centre_inner_crop_px"] = [cyi, cxi]
    json.dump(spokes, open(out + "spokes.json", "w"))
    json.dump(summary, open(out + "compare_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    np.savez_compressed(out + "recs.npz", **{f"{lb}|{k}": v for recs in recs_by_model.values() for lb, r in recs.items()
                                           for k, v in r.items() if k in ("phi_unf", "T_unf", "W_phys")},
                        dk=dk, kc=kc, dx=dx, crop=np.array(data["crop"]), star=np.array([cy, cx]))
    log(f"written {out}")

if __name__ == "__main__":
    main()
