# 32a spatial resolution, bright field (BF, rings 0-2) vs bright + dark field (BF+DF, + rings 6-7), after the posaffine correction.
# (1) Signed Siemens-star MTF of the unfiltered phase (full data sets): complex 36th angular harmonic c(r) on circles
#     (two-zone centre as in compare_32a.py), q = 36 / (2 pi r); MTF(q) = Re[c(q) c_ref*] / |c_ref|^2 with c_ref = mean c over
#     1.3-1.9 um^-1 (negative = contrast reversal); noise = rms of the off-harmonic orders / |c_ref|.
#     Also after refocusing the object O = exp(-a + i phi) by the autofocus result of compare_32a.py (pupil_drift_check.
#     object_refocus: z, astig; kernel exp(+i pi lam [z |u|^2 + a0 (ux^2 - uy^2) + a1 2 ux uy])).
#     Summary numbers: q_mtf10 = first q > 1.5 um^-1 where the signed MTF drops below 0.1; df_band = mean signed MTF over
#     3.9-4.6 um^-1 (beyond the bright-field limit kc (1 + max|k|/kc) = 3.75 um^-1, reachable only through the dark-field images).
# (2) Split-half FRC: two independent reconstructions from interleaved image halves (d32a/frc/setup_halves.py; DIP with
#     different seeds), unfiltered phase, 512 crop minus a 32 px border, Tukey 0.25, rings of 2 px; half-bit threshold.
# usage: python d32a/frc/frc_mtf_32a.py [tag]  -> d32a/frc/frc_mtf_32a.json (tag aff, default) or d32a/frc/frc_mtf_32a_<tag>.json
#        (tag aff2 = posaffine with the focus loop: data sets 32abfaff2 / 32adfaff2 and their split halves)
import json, os, sys
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal.windows import tukey
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
TAG = sys.argv[1] if len(sys.argv) > 1 else "aff"
DX = 0.0319; R_SPLIT = 85.0; REF = (1.3, 1.9); N = 36
METH = ["DIP (optimal it.)", "DIP (30 it.)", "EPRY", "BLIS-FPM"]

def star_harmonic(phi, cy, cx, inner):
    img = phi - phi.mean(); rmax = min(cy, cx, img.shape[0] - cy, img.shape[1] - cx) - 4
    r_all = np.arange(3, rmax, 0.5); th = np.linspace(0, 2 * np.pi, 2048, endpoint=False); c, nz = [], []
    for r in r_all:
        yc, xc = inner if r < R_SPLIT else (cy, cx)
        f = np.fft.rfft(map_coordinates(img, [yc + r * np.sin(th), xc + r * np.cos(th)], order=1)) / 2048 * 2
        c.append(f[N]); nz.append(np.sqrt(np.mean(np.abs(f[np.r_[N + 4:N + 30, max(N - 30, 1):N - 4]]) ** 2)))
    q = N / (2 * np.pi * r_all * DX); o = np.argsort(q)
    return q[o], np.array(c)[o], np.array(nz)[o]

def signed_mtf(phi, cy, cx, inner):
    q, c, nz = star_harmonic(phi, cy, cx, inner)
    cref = c[(q >= REF[0]) & (q <= REF[1])].mean()
    return q, np.real(c * np.conj(cref)) / abs(cref) ** 2, np.abs(c) / abs(cref), nz / abs(cref)

LAM = 1.239842 / 30 * 1e-3
def refocus_phase(phi, a, z, ast):
    No = phi.shape[0]; u = np.fft.fftfreq(No, DX); UY, UX = np.meshgrid(u, u, indexing="ij")
    Oz = np.fft.ifft2(np.fft.fft2(np.exp(-a + 1j * phi)) * np.exp(1j * np.pi * LAM * (z * (UX ** 2 + UY ** 2) + ast[0] * (UX ** 2 - UY ** 2) + ast[1] * 2 * UX * UY)))
    return np.angle(Oz * np.exp(-1j * np.angle(Oz.mean())))

def summary_numbers(q, sm, nz):
    o = (q > 1.5) & (sm < 0.1); q10 = float(q[o][0]) if o.any() else float("nan")
    b = (q >= 3.9) & (q <= 4.6)
    return dict(q_mtf10=q10, df_band=float(sm[b].mean()), df_band_noise=float(nz[b].mean()), min_signed_2_6=float(sm[(q >= 2) & (q <= 6)].min()))

def frc_rings(a, b, ring_px=2.0, apod=0.25):
    n = a.shape[-1]; w = np.outer(tukey(n, apod), tukey(n, apod))
    F1 = np.fft.fftshift(np.fft.fft2((a - a.mean()) * w)); F2 = np.fft.fftshift(np.fft.fft2((b - b.mean()) * w))
    yy, xx = np.mgrid[-(n // 2):n - n // 2, -(n // 2):n - n // 2]; rr = np.hypot(yy, xx)
    edges = np.arange(1, n // 2, ring_px); q, f, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (rr >= lo) & (rr < hi)
        f.append(np.real((F1[s] * np.conj(F2[s])).sum()) / np.sqrt((np.abs(F1[s]) ** 2).sum() * (np.abs(F2[s]) ** 2).sum()))
        q.append(0.5 * (lo + hi) / (n * DX)); cnt.append(s.sum())
    q, f, cnt = map(np.array, (q, f, cnt))
    return q, f, (0.2071 + 1.9102 / np.sqrt(cnt)) / (1.2071 + 0.9102 / np.sqrt(cnt))

def crossing(q, f, thr, qmin=0.5):
    thr = np.broadcast_to(thr, f.shape); below = np.where((q >= qmin) & (f < thr))[0]
    if not len(below): return float("nan")
    i = below[0]
    if i == 0: return float(q[0])
    d0, d1 = f[i - 1] - thr[i - 1], f[i] - thr[i]
    return float(q[i - 1] + (q[i] - q[i - 1]) * d0 / (d0 - d1))

out = dict(method="see header", ref_band_um_inv=REF, mtf={}, frc={})
for g in ("512", "1000"):
    for s, key in (("BF", f"32abf{TAG}_{g}"), ("BF+DF", f"32adf{TAG}_{g}")):
        d = W + f"d32a/out_{key}/"
        if not os.path.exists(d + "recs.npz"): continue
        z = np.load(d + "recs.npz"); sp = json.load(open(d + "spokes.json"))
        cy, cx = sp["centre_px"]; inner = tuple(sp["centre_inner_px"])
        CS = json.load(open(d + "compare_summary.json"))["results"]
        for lb in [k.split("|")[0] for k in z.files if k.endswith("|phi_unf")]:
            phi = z[f"{lb}|phi_unf"].astype(float); a = -np.log(np.clip(z[f"{lb}|T_unf"].astype(float), 1e-6, None))
            q, sm, am, nz = signed_mtf(phi, cy, cx, inner)
            rf = CS[lb]["pupil_drift_check"]["object_refocus"]
            q2, sm2, am2, nz2 = signed_mtf(refocus_phase(phi, a, rf["z_um"], rf["astig_um"]), cy, cx, inner)
            out["mtf"][f"{g}|{s}|{lb}"] = dict(q=q.tolist(), signed=sm.tolist(), abs=am.tolist(), noise=nz.tolist(), q_snr3=sp[lb]["q_snr3"],
                                               signed_refocused=sm2.tolist(), noise_refocused=nz2.tolist(), refocus_um=[rf["z_um"]] + list(rf["astig_um"]),
                                               numbers=summary_numbers(q, sm, nz), numbers_refocused=summary_numbers(q2, sm2, nz2))
for s, (ka, kb) in (("BF", (f"32abf{TAG}HA_512", f"32abf{TAG}HB_512")), ("BF+DF", (f"32adf{TAG}HA_512", f"32adf{TAG}HB_512"))):
    fa, fb = W + f"d32a/out_{ka}/recs.npz", W + f"d32a/out_{kb}/recs.npz"
    if not (os.path.exists(fa) and os.path.exists(fb)): continue
    za, zb = np.load(fa), np.load(fb); b = 32
    for m in METH:
        lb = f"{m}, FOV model|phi_unf"
        if lb not in za.files or lb not in zb.files: continue
        q, f, hb = frc_rings(za[lb][b:-b, b:-b].astype(float), zb[lb][b:-b, b:-b].astype(float))
        out["frc"][f"{s}|{m}"] = dict(q=q.tolist(), frc=f.tolist(), halfbit=hb.tolist(), q_halfbit=crossing(q, f, hb), q_1_7=crossing(q, f, 1 / 7))
        print(f"FRC {s:6s} {m:18s} half-bit {out['frc'][f'{s}|{m}']['q_halfbit']:.2f} um^-1, 1/7 {out['frc'][f'{s}|{m}']['q_1_7']:.2f}")
# refocused split-half FRC (each half refocused by its own autofocus), crossings capped at the support limit
geo = json.load(open(W + "d32a/geom_direct_v2.json")); kk = np.array(geo["k_over_kc"]); rings = np.repeat(np.arange(8), [15, 20, 25, 30, 35, 40, 45, 50])
QB, QD = 2.0 * (1 + kk[rings <= 2].max()), 2.0 * (1 + kk[np.isin(rings, [6, 7])].max()); out["support_limits_um_inv"] = dict(BF=QB, BF_DF=QD)
def half(key, m):
    d = W + f"d32a/out_{key}/"; z = np.load(d + "recs.npz"); rf = json.load(open(d + "compare_summary.json"))["results"][f"{m}, FOV model"]["pupil_drift_check"]["object_refocus"]
    phi = z[f"{m}, FOV model|phi_unf"].astype(float); a = -np.log(np.clip(z[f"{m}, FOV model|T_unf"].astype(float), 1e-6, None))
    return refocus_phase(phi, a, rf["z_um"], rf["astig_um"]), rf["z_um"]
out["frc_refocused"], out["frc_limits_capped"] = {}, {}
for s, (ka, kb), qcap in (("BF", (f"32abf{TAG}HA_512", f"32abf{TAG}HB_512"), QB), ("BF+DF", (f"32adf{TAG}HA_512", f"32adf{TAG}HB_512"), QD)):
    if not os.path.exists(W + f"d32a/out_{kb}/recs.npz"): continue
    for m in METH:
        (A_, za_), (B_, zb_) = half(ka, m), half(kb, m)
        q, f, hb = frc_rings(A_[32:-32, 32:-32], B_[32:-32, 32:-32])
        out["frc_refocused"][f"{s}|{m}"] = dict(q=q.tolist(), frc=f.tolist(), halfbit=hb.tolist(), refocus_um=[za_, zb_])
        for row, d in (("as reconstructed", out["frc"][f"{s}|{m}"]), ("after object refocus", out["frc_refocused"][f"{s}|{m}"])):
            qq, ff, hh = map(np.asarray, (d["q"], d["frc"], d["halfbit"])); sel = qq <= qcap; c = crossing(qq[sel], ff[sel], hh[sel])
            out["frc_limits_capped"][f"{row}|{s}|{m}"] = dict(q=float(c) if np.isfinite(c) else float(qcap), no_crossing_below_support_limit=not np.isfinite(c))
# star-based focus check: band means of the signed MTF (512, FOV) after propagating the object by z
zs = np.arange(-1500, 3001, 250.0); out["star_focus_scan"] = dict(z_um=zs.tolist(), band_2_7_3_6={}, band_3_9_4_6={})
for s, key in (("BF", f"32abf{TAG}_512"), ("BF+DF", f"32adf{TAG}_512")):
    d = W + f"d32a/out_{key}/"; z = np.load(d + "recs.npz"); sp = json.load(open(d + "spokes.json")); cy, cx = sp["centre_px"]; inner = tuple(sp["centre_inner_px"])
    for m in METH:
        phi = z[f"{m}, FOV model|phi_unf"].astype(float); a = -np.log(np.clip(z[f"{m}, FOV model|T_unf"].astype(float), 1e-6, None)); v1, v2 = [], []
        for zz in zs:
            q, sm, am, nz = signed_mtf(refocus_phase(phi, a, zz, (0.0, 0.0)), cy, cx, inner)
            v1.append(float(sm[(q >= 2.7) & (q <= 3.6)].mean())); v2.append(float(sm[(q >= 3.9) & (q <= 4.6)].mean()))
        out["star_focus_scan"]["band_2_7_3_6"][f"{s}|{m}"] = v1; out["star_focus_scan"]["band_3_9_4_6"][f"{s}|{m}"] = v2
json.dump(out, open(W + ("d32a/frc/frc_mtf_32a.json" if TAG == "aff" else f"d32a/frc/frc_mtf_32a_{TAG}.json"), "w"), indent=0)
print("mtf entries", len(out["mtf"]), "frc entries", len(out["frc"]))
