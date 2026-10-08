# -*- coding: utf-8 -*-
"""32a: directional (anisotropy) diagnostics.
A. reconstructions (BF 512): 2D phase power-spectrum anisotropy per frequency band, and Siemens-star spoke resolution
   per grating direction (sectors). Angles in (col, row) image coordinates = (lens X, lens Y), degrees.
B. direct (sample-out) images, rings 3-5: width of the vignetting (pupil) edge vs edge-normal direction
   -> illumination angular spread per direction (coherence), sigma_theta ~ 0.8 s / p.
Writes d32a/aniso_32a.json"""
import json, numpy as np, tifffile
from scipy import ndimage as ndi
from scipy.optimize import least_squares
W = "d32a/"; PX = 0.0319
out = {}
# ---------------- A. reconstructions
z = np.load(W + "out_32abf_512/recs.npz"); S = json.load(open(W + "out_32abf_512/compare_summary.json"))
c_out = np.array(S["star_centre_crop_px"], float); c_in = np.array(S["star_centre_inner_crop_px"], float)
N = 512; q1 = np.fft.fftfreq(N, PX); QX, QY = np.meshgrid(q1, q1)          # QX along columns, QY along rows
Q = np.hypot(QX, QY); PH = np.arctan2(QY, QX)
win = np.outer(np.hanning(N), np.hanning(N))
bands = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5)]
def psd_aniso(img):
    F = np.abs(np.fft.fft2((img - img.mean()) * win)) ** 2; res = []
    for lo, hi in bands:
        m = (Q >= lo) & (Q < hi); y = np.log(F[m]); X = np.c_[np.ones(m.sum()), np.cos(2 * PH[m]), np.sin(2 * PH[m])]
        c, *_ = np.linalg.lstsq(X, y, rcond=None); amp = np.hypot(c[1], c[2]); ax = 0.5 * np.degrees(np.arctan2(c[2], c[1]))
        res.append(dict(band=[lo, hi], max_over_min_power=float(np.exp(2 * amp)), axis_of_max_deg=float((ax + 90) % 180 - 90)))
    return res
def sector_spokes(img, width=60):
    th = np.radians(np.arange(0, 360, 0.25)); rr = np.arange(15, 230); snr = {}
    dirs = np.arange(0, 180, 30)
    prof = np.zeros((len(rr), len(th)))
    for i, r in enumerate(rr):
        c = c_in if r < 85 else c_out
        prof[i] = ndi.map_coordinates(img, [c[0] + r * np.sin(th), c[1] + r * np.cos(th)], order=1, mode="nearest")
    res = {}
    for d in dirs:                                   # grating direction d  <=> sector azimuth psi = d - 90 (and +180)
        psi = np.radians(d - 90); dang = np.angle(np.exp(1j * (th - psi)))
        sel = (np.abs(dang) < np.radians(width / 2)) | (np.abs(np.angle(np.exp(1j * (th - psi - np.pi)))) < np.radians(width / 2))
        t = th[sel]; X = np.c_[np.ones(sel.sum()), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t), np.cos(36 * t), np.sin(36 * t)]
        s = []
        for i in range(len(rr)):
            c, *_ = np.linalg.lstsq(X, prof[i, sel], rcond=None); resid = prof[i, sel] - X @ c
            s.append(np.hypot(c[5], c[6]) / (resid.std() * np.sqrt(2) + 1e-12))
        s = np.array(s); q = 36 / (2 * np.pi * rr * PX)
        o = np.argsort(q); qs, ss = q[o], s[o]
        lim = np.nan; k0 = np.searchsorted(qs, 1.2)
        for k in range(k0, len(qs) - 3):             # first q >= 1.2 where SNR stays < 3 for 3 consecutive radii
            if (ss[k:k + 3] < 3).all(): lim = qs[k]; break
        res[int(d)] = float(lim)
    return res
A = {}
for key in z.files:
    if not key.endswith("|phi_unf") or "30 it" in key: continue
    lab = key.split("|")[0]; img = z[key].astype(float)
    A[lab] = dict(psd=psd_aniso(img), spoke_limit_by_grating_direction=sector_spokes(img))
out["A_reconstructions_bf512"] = A
# ---------------- B. direct images: pupil-edge width vs direction
D = np.load(W + "raw/direct_u16.npy", mmap_mode="r"); prep = json.load(open(W + "data_reg/prep_reg.json"))
res = np.c_[prep["shift_row_px"], prep["shift_col_px"]]
ring = np.array(prep["ring"]); geo = json.load(open(W + "geom_direct_v2.json")); xc = np.array(geo["xc_um"]); R = geo["R_field_um"]
E = np.median(np.stack([D[i].astype(np.float32) - 100 for i in np.where(ring == 0)[0]]), 0)
E = ndi.gaussian_filter(E, 2); band = E > 0.3 * np.percentile(E, 99.5)
rows_b = np.where(band.any(1))[0]; cols_b = np.where(band.any(0))[0]
out["B_illumination"] = dict(band_rows=[int(rows_b.min()), int(rows_b.max())], band_cols=[int(cols_b.min()), int(cols_b.max())])
yy, xx = np.mgrid[0:1000, 0:1000]; Yu = (yy - 499.5) * PX; Xu = (xx - 499.5) * PX
fits = []
for i in np.where((ring >= 3) & (ring <= 5))[0]:
    img = ndi.shift(D[i].astype(np.float32) - 100, res[i], order=1, mode="nearest")
    Ei = ndi.shift(E, res[i], order=1, mode="nearest"); bi = ndi.shift(band.astype(float), res[i], order=0) > 0.5
    ratio = img / np.maximum(Ei, 1) * (np.median(Ei[bi]) / max(np.median(img[bi]), 1)) if False else img / np.maximum(Ei, 1)
    dy, dx = Yu - xc[i, 0], Xu - xc[i, 1]; rho = np.hypot(dy, dx) - R; phi = np.arctan2(dy, dx)
    m = bi & (np.abs(rho) < 10)
    if m.sum() < 3000: continue
    sub = np.where(m.ravel())[0][::7]; rv, yv, pv = rho.ravel()[sub], ratio.ravel()[sub], phi.ravel()[sub]
    inside = np.median(yv[rv < -5]) if (rv < -5).sum() > 200 else np.nan; outside = np.median(yv[rv > 5]) if (rv > 5).sum() > 200 else np.nan
    if not (np.isfinite(inside) and np.isfinite(outside)) or inside - outside < 0.2 * inside: continue
    f = lambda p: p[1] + p[0] * 0.5 * (1 - np.tanh((rv - p[3]) / abs(p[2]))) - yv
    r_ = least_squares(f, [inside - outside, outside, 2.0, 0.0], loss="soft_l1", f_scale=0.1 * inside)
    a, b, s, dR = r_.x; mphi = np.angle(np.mean(np.exp(1j * pv))); spread = np.degrees(np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * pv))))))
    fits.append(dict(i=int(i), ring=int(ring[i]), s_um=float(abs(s)), dR_um=float(dR), normal_deg=float(np.degrees(mphi)), arc_spread_deg=float(spread), contrast=float(a / inside)))
F = [f for f in fits if f["arc_spread_deg"] < 35 and f["contrast"] > 0.5 and abs(f["dR_um"]) < 4]
ph = np.radians([f["normal_deg"] for f in F]); s2 = np.array([f["s_um"] for f in F]) ** 2
X = np.c_[np.cos(ph) ** 2, 2 * np.cos(ph) * np.sin(ph), np.sin(ph) ** 2]; c, *_ = np.linalg.lstsq(X, s2, rcond=None)
Mq = np.array([[c[0], c[1]], [c[1], c[2]]]); w, v = np.linalg.eigh(Mq)
pv_ = 0.75e6
out["B_pupil_edge"] = dict(n_fits=len(fits), n_used=len(F), s_um_quadratic_form=Mq.tolist(),
    s_principal_um=[float(np.sqrt(max(w[0], 0))), float(np.sqrt(max(w[1], 0)))], axis_of_larger_deg=float(np.degrees(np.arctan2(v[1, 1], v[0, 1]))),
    s_along_col_um=float(np.sqrt(max(c[0], 0))), s_along_row_um=float(np.sqrt(max(c[2], 0))),
    sigma_theta_urad_col_row=[float(0.8 * np.sqrt(max(c[0], 0)) / pv_ * 1e6), float(0.8 * np.sqrt(max(c[2], 0)) / pv_ * 1e6)],
    per_frame=F)
json.dump(out, open(W + "aniso_32a.json", "w"), indent=1)
# ---- short print
for lab, v in A.items():
    print(lab[:28].ljust(28), "PSD max/min, axis:", " ".join(f"{b['band'][0]:.1f}-{b['band'][1]:.1f}:{b['max_over_min_power']:.2f}@{b['axis_of_max_deg']:+.0f}" for b in v["psd"]))
    print(" " * 28, "spoke limit by grating dir:", v["spoke_limit_by_grating_direction"])
print("illum:", out["B_illumination"]); bp = out["B_pupil_edge"]
print("edge fits used", bp["n_used"], "/", bp["n_fits"], "| s along col (normal horizontal)", round(bp["s_along_col_um"], 2), "um, along row", round(bp["s_along_row_um"], 2),
      "um | principal", np.round(bp["s_principal_um"], 2), "axis", round(bp["axis_of_larger_deg"], 1), "| sigma_theta urad (col,row)", np.round(bp["sigma_theta_urad_col_row"], 2))
print("normal angles covered:", np.round(sorted([f["normal_deg"] for f in F]), 0))
