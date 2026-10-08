"""Weak-object transfer-function (WOTF) model: illumination/pupil calibration and linear
(first-order) FPM reconstruction on overlapping tiles.

Convention ("solver" frame): illumination exp(+i 2 pi k_n . x), object O = exp(-a + i phi),
pupil P(k) = disc * exp(i pi lam z |k|^2 + astig). Linearised flat-fielded intensity spectrum
    I_n(q) = -A(q) [T1 + T2] + i Phi(q) [T1 - T2],   T1 = P*(k_n) P(k_n+q),  T2 = P(k_n) P*(k_n-q).
The solution is determined up to the twin (O, P(k), k) <-> (O*, P*(-k), -k); see postprocess.
"""
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .optics import pupil_fn


def crop_spectra(R, center, N, dx, qband, window="hann"):
    """Windowed FFT of (I-1) for a square crop; returns data (n, nq) and q vectors (nq, 2)."""
    cy, cx = center
    sub = R[:, cy - N // 2:cy + N // 2, cx - N // 2:cx + N // 2]
    sub = sub / sub.mean((1, 2), keepdims=True)
    w = np.outer(np.hanning(N), np.hanning(N))
    F = np.fft.fftshift(np.fft.fft2((sub - 1) * w, axes=(-2, -1)), axes=(-2, -1))
    dk = 1 / (N * dx)
    qy, qx = np.mgrid[-N // 2:N // 2, -N // 2:N // 2] * dk
    sel = (np.hypot(qy, qx) > qband[0]) & (np.hypot(qy, qx) < qband[1])
    return F[:, sel], np.stack([qy[sel], qx[sel]], 1), sel


class WOTF:
    def __init__(self, Iq, qv, kc, lam, z=0.0, astig=(0.0, 0.0), edge=0.05):
        self.Iq, self.qv, self.kc, self.lam, self.z, self.astig, self.edge = Iq, qv, kc, lam, z, astig, edge

    def P(self, k):
        return pupil_fn(k, self.kc, self.z, self.astig, self.lam, self.edge)

    def transfer(self, kn):
        k = np.atleast_2d(kn)[:, None, :]
        P0 = self.P(k); Pp = self.P(k + self.qv[None]); Pm = self.P(k - self.qv[None])
        T1 = np.conj(P0) * Pp; T2 = P0 * np.conj(Pm)
        return -(T1 + T2), 1j * (T1 - T2)

    def solve(self, kn, reg=0.0):
        Ha, Hp = self.transfer(kn)
        Iq = self.Iq
        a11 = (np.abs(Ha) ** 2).sum(0) + reg; a22 = (np.abs(Hp) ** 2).sum(0) + reg
        a12 = (np.conj(Ha) * Hp).sum(0)
        b1 = (np.conj(Ha) * Iq).sum(0); b2 = (np.conj(Hp) * Iq).sum(0)
        det = a11 * a22 - np.abs(a12) ** 2 + 1e-12
        self.Aq = (a22 * b1 - a12 * b2) / det
        self.Pq = (a11 * b2 - np.conj(a12) * b1) / det
        res = Iq - Ha * self.Aq - Hp * self.Pq
        self.per_image = (np.abs(res) ** 2).sum(1) / (np.abs(Iq) ** 2).sum(1)
        return float((np.abs(res) ** 2).sum() / (np.abs(Iq) ** 2).sum())


# ---- illumination geometry ----------------------------------------------------------------
def kvec_rot(Pst, phi_deg, scale, k_per_pulse, k0=(0.0, 0.0), flip=False):
    """Stage (x, y) [pulse] -> k (row, col) [um^-1]: rotation phi, isotropic scale, optional mirror."""
    x, y = Pst[:, 0], Pst[:, 1] * (-1 if flip else 1)
    ang = np.arctan2(y, x) + np.deg2rad(phi_deg)
    r = np.hypot(x, y) * scale * k_per_pulse
    return np.stack([r * np.sin(ang) + k0[0], r * np.cos(ang) + k0[1]], 1)


def kvec_affine(Pst, Mk, k0):
    return Pst @ np.asarray(Mk).T + np.asarray(k0)[None]


def calibrate(R, cfg, Pst, log=print, quick=False):
    """Fit stage->k mapping, defocus and astigmatism with the WOTF model on a central crop."""
    c = cfg["calib"]; lam, dx, kc, kpp = cfg["lam_um"], cfg["pixel_um"], cfg["kc"], cfg["k_per_pulse_nominal"]
    N = c["crop_px"]; H = R.shape[-1]
    center = c["crop_center"] or (H // 2, H // 2)
    Iq, qv, _ = crop_spectra(R, center, N, dx, c["q_band"])
    W = WOTF(Iq, qv, kc, lam, edge=c["pupil_edge"])
    t0 = time.time()
    # A) rotation / mirror / defocus grid at nominal scale
    phis = np.arange(0, 180, c["phi_step_deg"])
    zs = np.array(c["z_grid_um"], float)
    best = (np.inf, None)
    for fl in (False, True):
        for ph in phis:
            kn = kvec_rot(Pst, ph, 1.0, kpp, flip=fl)
            for z in zs:
                W.z = z; W.astig = (0, 0)
                f = W.solve(kn)
                if f < best[0]:
                    best = (f, (ph, fl, z))
    phi0, flip, z0 = best[1]
    log(f"[calib A] rotation {phi0:.1f} deg, mirror={flip}, defocus {z0:.0f} um, misfit {best[0]:.4f} ({time.time()-t0:.0f} s)")
    # A2) scale / defocus grid
    s0, s1, ds = c["scale_grid"]; z0g, z1g, dzg = c["z_grid2_um"]
    scs = np.arange(s0, s1 + 1e-9, ds * (2 if quick else 1)); zs2 = np.arange(z0g, z1g + 1e-9, dzg * (2 if quick else 1))
    prof = []                                             # best misfit per scale
    for sc in scs:
        kn = kvec_rot(Pst, phi0, sc, kpp, flip=flip)
        fz = []
        for z in zs2:
            W.z = z
            fz.append(W.solve(kn))
        prof.append((min(fz), sc, zs2[int(np.argmin(fz))]))
    prof = np.array(prof)
    ib = int(np.argmin(prof[:, 0])); best2 = (prof[ib, 0], (prof[ib, 1], prof[ib, 2]))
    # local minima of the scale profile -> multi-start seeds
    seeds = [(prof[i, 1], prof[i, 2]) for i in range(len(prof))
             if prof[i, 0] <= prof[max(i - 1, 0), 0] and prof[i, 0] <= prof[min(i + 1, len(prof) - 1), 0]]
    sc0, z0 = best2[1]
    log(f"[calib A2] scale {sc0:.3f} x nominal, defocus {z0:.0f} um, misfit {best2[0]:.4f} ({time.time()-t0:.0f} s)")

    # B) global optimisation (rotation, scale, defocus, astigmatism, pupil-centre offset)
    def obj(p):
        ph, sc, z, a0, a1, k0y, k0x = p
        W.z = z * 1000; W.astig = (a0 * 1000, a1 * 1000)
        return W.solve(kvec_rot(Pst, ph, sc, kpp, (k0y, k0x), flip))
    b = c["de_bounds"]
    bounds = [(phi0 - b["dphi_deg"], phi0 + b["dphi_deg"]), (max(0.05, sc0 - b["dscale"]), sc0 + b["dscale"]),
              (z0 / 1000 - b["dz_mm"], z0 / 1000 + b["dz_mm"]), (-b["astig_mm"], b["astig_mm"]),
              (-b["astig_mm"], b["astig_mm"]), (-b["k0"], b["k0"]), (-b["k0"], b["k0"])]
    de = differential_evolution(obj, bounds, popsize=8 if quick else c["de_popsize"],
                                maxiter=20 if quick else c["de_maxiter"], tol=1e-7, seed=c["de_seed"], polish=False)
    starts = [de.x] + [np.array([phi0, sc_, z_ / 1000, 0, 0, 0, 0]) for sc_, z_ in seeds]
    cands = []
    for x0 in starts:
        r_ = minimize(obj, x0, method="Nelder-Mead", options=dict(maxiter=400 if quick else 1500, xatol=1e-4, fatol=1e-6))
        cands.append((r_.fun, r_.x))
        log(f"   start scale {x0[1]:.3f}: -> scale {r_.x[1]:.4f}, rot {r_.x[0]:.2f}, z {r_.x[2]*1000:.0f} um, "
            f"k0 ({r_.x[5]:.2f},{r_.x[6]:.2f}), misfit {r_.fun:.4f}")
    cands.sort(key=lambda t: t[0])
    nm_fun, nm_x = cands[0]
    ph, sc, z, a0, a1, k0y, k0x = nm_x
    class _R: pass
    nm = _R(); nm.fun = nm_fun
    log(f"[calib B] rot {ph:.2f} deg, scale {sc:.4f}, z {z*1000:.0f} um, astig ({a0*1000:.0f},{a1*1000:.0f}) um, "
        f"k0 ({k0y:.3f},{k0x:.3f}) um^-1, misfit {nm.fun:.4f} ({time.time()-t0:.0f} s)")
    # C) affine refinement of the stage->k map
    e = np.eye(2)
    Mk0 = np.stack([kvec_rot(e[i:i + 1], ph, sc, kpp, flip=flip)[0] for i in range(2)], 1)  # columns = images of unit x, y

    def obj_aff(p):
        Mk = np.array([[p[0], p[1]], [p[2], p[3]]]) * 1e-3
        W.z = p[4] * 1000; W.astig = (p[5] * 1000, p[6] * 1000)
        return W.solve(kvec_affine(Pst, Mk, (p[7], p[8])))
    p0 = np.r_[Mk0.ravel() * 1e3, z, a0, a1, k0y, k0x]
    ra = minimize(obj_aff, p0, method="Nelder-Mead", options=dict(maxiter=500 if quick else 3000, xatol=1e-5, fatol=1e-7, adaptive=True))
    Mk = np.array([[ra.x[0], ra.x[1]], [ra.x[2], ra.x[3]]]) * 1e-3
    k0 = (float(ra.x[7]), float(ra.x[8]))
    kn = kvec_affine(Pst, Mk, k0)
    W.z = ra.x[4] * 1000; W.astig = (ra.x[5] * 1000, ra.x[6] * 1000)
    fin = W.solve(kn)
    sv = np.linalg.svd(Mk, compute_uv=False)
    log(f"[calib C] affine: singular values {sv / kpp} x nominal, misfit {fin:.4f} ({time.time()-t0:.0f} s)")
    return dict(Mk=Mk.tolist(), k0=list(k0), defocus_um=float(W.z), astig_um=[float(W.astig[0]), float(W.astig[1])],
                rotation_deg=float(ph), mirror=bool(flip), scale_vs_nominal=float(np.sqrt(abs(np.linalg.det(Mk))) / kpp),
                angle_per_pulse_urad=float(np.sqrt(abs(np.linalg.det(Mk))) * lam * 1e6),
                misfit=fin, per_image_misfit=W.per_image.tolist(),
                k_solver=kn.tolist(), k_over_kc=(np.hypot(kn[:, 0], kn[:, 1]) / kc).tolist())


# ---- linear reconstruction on overlapping tiles -------------------------------------------
def tile_linear_recon(R, kn, cfg, kappa, z, astig, edge=0.05, log=print):
    """Tikhonov-regularised WOTF solution on 50%-overlap tiles; local illumination
    k(x) = k_n + kappa (x - x_centre) (FOV effect). Returns absorption a, phase phi (camera grid)."""
    L = cfg["linear"]; dx, kc, lam = cfg["pixel_um"], cfg["kc"], cfg["lam_um"]
    Nt, step, (qlo, qhi), reg_rel = L["tile_px"], L["step_px"], L["q_band"], L["reg_rel"]
    H = R.shape[-1]
    wt = np.hanning(Nt + 1)[:-1]; wt2 = np.outer(wt, wt)       # periodic Hann: partition of unity
    dkt = 1 / (Nt * dx)
    qy, qx = np.mgrid[-Nt // 2:Nt // 2, -Nt // 2:Nt // 2] * dkt
    sel = (np.hypot(qy, qx) > qlo) & (np.hypot(qy, qx) < qhi)
    W = WOTF(None, np.stack([qy[sel], qx[sel]], 1), kc, lam, z, astig, edge)
    a_map = np.zeros((H, H)); p_map = np.zeros((H, H)); wsum = np.zeros((H, H))
    starts = list(range(0, H - Nt + 1, step))
    if starts[-1] != H - Nt:
        starts.append(H - Nt)
    resid = {}
    for y0 in starts:
        for x0 in starts:
            sub = R[:, y0:y0 + Nt, x0:x0 + Nt]
            sub = sub / sub.mean((1, 2), keepdims=True)
            F = np.fft.fftshift(np.fft.fft2((sub - 1) * wt2, axes=(-2, -1)), axes=(-2, -1))[:, sel]
            dxy = np.array([(y0 + Nt / 2 - H / 2) * dx, (x0 + Nt / 2 - H / 2) * dx])
            Ha, Hp = W.transfer(kn + kappa * dxy[None])
            a11 = (np.abs(Ha) ** 2).sum(0); a22 = (np.abs(Hp) ** 2).sum(0); a12 = (np.conj(Ha) * Hp).sum(0)
            reg = reg_rel * np.median(a22)
            b1 = (np.conj(Ha) * F).sum(0); b2 = (np.conj(Hp) * F).sum(0)
            A11, A22 = a11 + reg, a22 + reg
            det = A11 * A22 - np.abs(a12) ** 2
            Aq = (A22 * b1 - a12 * b2) / det; Pq = (A11 * b2 - np.conj(a12) * b1) / det
            res = F - Ha * Aq - Hp * Pq
            resid[(y0, x0)] = float((np.abs(res) ** 2).sum() / (np.abs(F) ** 2).sum())
            Am = np.zeros((Nt, Nt), complex); Pm = np.zeros((Nt, Nt), complex)
            Am[sel] = Aq; Pm[sel] = Pq
            a_map[y0:y0 + Nt, x0:x0 + Nt] += np.fft.ifft2(np.fft.ifftshift(Am)).real
            p_map[y0:y0 + Nt, x0:x0 + Nt] += np.fft.ifft2(np.fft.ifftshift(Pm)).real
            wsum[y0:y0 + Nt, x0:x0 + Nt] += wt2
    ok = wsum > 0.2
    a_map[ok] /= wsum[ok]; p_map[ok] /= wsum[ok]
    return a_map.astype(np.float32), p_map.astype(np.float32), resid
