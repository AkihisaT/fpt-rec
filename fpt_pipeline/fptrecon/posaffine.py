"""Optional step 'posaffine': separate the image drift that is linear in the lens / illumination position from the
pupil-phase parallax, and correct it as a mechanical (stage / detector) error.

Why: an image shift that grows linearly with the illumination (lens) position can come from two sources,
  (a) the pupil phase W(k): the direct beam passes the pupil at k_n, so the image of illumination n is shifted by
      s_n = -grad W(k_n) / (2 pi)   (defocus / astigmatism -> linear in k_n, coma -> quadratic, ...),
  (b) mechanics: the detector does not follow the image exactly (scale error of the follow, rotation or shear between
      the lens stage and the detector axes).
The reconstruction can only express (b) as pupil phase, so an uncorrected mechanical drift turns into a spurious
astigmatism / defocus of the pupil, and the object then has to cancel it (elliptical ringing). This step
  1. measures absolute image shifts s_n with a pairwise cross-correlation network (neighbours in k; model-free),
  2. fits s_n = B pos_n + c + (curl-free higher orders in k) + random, B = full 2x2 matrix (px per stage unit),
  3. removes all measured shifts, fits pupil + object (BLIS-FPM), and brings the object into focus. The image contrast
     fixes only the SUM of the pupil and object quadratic phases (pupil defocus + object propagation give the same
     contrast; only the image drift tells them apart), so the contrast focus = pupil quadratic - object refocus is
     assigned to the pupil (object in focus); the parallax this pupil predicts is kept, every other measured shift
     (the 2x2 matrix B_mech, higher orders, random) is corrected as mechanical,
  3b. (optional, focus_iter > 0; added 2026-09-28) focus loop: the object is solved with the pupil FIXED at the
     contrast focus (pure defocus + astigmatism, analytic_W) on the stack that keeps this parallax; a residual object
     defocus is moved into the pupil (and the kept parallax is recomputed), until |object refocus| < focus_tol_um.
     Note: an object defocus z is equivalent to a pupil defocus z plus an image shift lambda z k_n, and a shift linear
     in k_n cannot be told from the isotropic scale of the mechanical drift; the data misfit is therefore (nearly)
     unchanged by the loop, which selects the solution with the object in focus (32a: -1.53 -> -2.72 mm, isotropic
     mechanical scale -0.16 %),
  4. checks with fixed positions (the consistency check) that (pupil - object refocus) of the new fit predicts the
     kept drift and that the model-based residual shifts have no linear trend; the split of the quadratic phase
     between pupil and object (object refocus) is reported; then the random part of the residual shifts is applied,
  5. writes the corrected stack, the contrast-based defocus / astigmatism into calibration.json, and a report.
Conventions: solver frame of nonlinear.py (psi_n = F^-1{P(k) O^(k - k_n)}: the direct beam passes the pupil at +k_n);
shifts in camera px, (row, col); an image shifted by s means I_n(x) = I_ref(x - s).
The sign of s_n = -grad W / 2pi is verified numerically by selftest_sign()."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from scipy.signal.windows import tukey

from . import nonlinear as nlr
from .optics import fcrop_intensity, kgrid

DEFAULTS = dict(fov_factor=1.0, n_neighbours=4, n_pupil=20, n_joint=150,
                model_refine=True, warn_percent=0.1, poly_order=4, centre_exclude=0.25, threads=None,
                focus_iter=0, focus_tol_um=100.0)


# ---------------------------------------------------------------- shift measurement
def xcorr_shift(A, B):
    """shift s (row, col) such that B(x) ~ A(x - s); parabolic sub-pixel peak."""
    c = np.fft.fftshift(np.real(np.fft.ifft2(np.conj(np.fft.fft2(A)) * np.fft.fft2(B))))
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    par = lambda m, o, p: 0.5 * (m - p) / (m - 2 * o + p)
    return np.array([iy + par(c[iy - 1, ix], c[iy, ix], c[iy + 1, ix]) - c.shape[0] // 2,
                     ix + par(c[iy, ix - 1], c[iy, ix], c[iy, ix + 1]) - c.shape[1] // 2])


def neighbour_pairs(kn, n_nb=4):
    """Pairs (i, j) of nearest neighbours in k (each image with its n_nb nearest); graph made connected."""
    n = len(kn); d = np.hypot(*(kn[:, None, :] - kn[None, :, :]).transpose(2, 0, 1)); np.fill_diagonal(d, np.inf)
    P = set()
    for i in range(n):
        for j in np.argsort(d[i])[:n_nb]: P.add((min(i, j), max(i, j)))
    par = list(range(n))
    def find(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in P: par[find(a)] = find(b)
    while len({find(i) for i in range(n)}) > 1:              # join components by their closest pair
        comp = np.array([find(i) for i in range(n)]); c0 = comp == comp[0]
        sub = np.where(c0[:, None] & ~c0[None, :], d, np.inf); i, j = np.unravel_index(np.argmin(sub), sub.shape)
        P.add((min(i, j), max(i, j))); par[find(i)] = find(j)
    return sorted(P)


def network_shifts(R, kn, n_nb=4, threads=8, log=print):
    """Absolute shifts s_n (zero mean) from pairwise cross-correlations of band-passed images (1.5-12 px)."""
    N = R.shape[-1]; win = np.outer(np.hanning(N), np.hanning(N))
    hp = lambda im: ((ndi.gaussian_filter(im, 1.5) - ndi.gaussian_filter(im, 12)) * win).astype(np.float32)
    with ThreadPoolExecutor(threads) as ex:
        H = list(ex.map(lambda i: hp(np.asarray(R[i], np.float32)), range(len(R))))
    pairs = neighbour_pairs(kn, n_nb)
    with ThreadPoolExecutor(threads) as ex:
        meas = np.array(list(ex.map(lambda p: xcorr_shift(H[p[0]], H[p[1]]), pairs)))
    n = len(R); A = np.zeros((len(pairs) + 1, n))
    for k, (a, b) in enumerate(pairs): A[k, a] = -1; A[k, b] = 1
    A[-1] = 1; w = np.ones(len(pairs))
    for _ in range(5):
        Aw = A.copy(); Aw[:-1] *= w[:, None]
        s = np.stack([np.linalg.lstsq(Aw, np.r_[meas[:, d] * w, 0], rcond=None)[0] for d in range(2)], 1)
        res = np.array([s[b] - s[a] for a, b in pairs]) - meas; rn = np.hypot(*res.T)
        w = (rn < 3 * np.median(rn) + 0.5).astype(float)
    info = dict(n_pairs=len(pairs), outliers=int((w == 0).sum()), closure_rms_px=float(np.sqrt((rn[w > 0] ** 2).mean())))
    log(f"  shift network: {n} images, {len(pairs)} pairs, {info['outliers']} outliers, closure rms {info['closure_rms_px']:.2f} px")
    return s, info


# ---------------------------------------------------------------- linear / curl-free decomposition
def _grad_basis(k, degrees=(3, 4)):
    """gradients of monomials of the given degrees in k = (ky, kx): curl-free fields quadratic / cubic in k."""
    ky, kx = k[:, 0], k[:, 1]; cols = []
    for deg in degrees:
        for p in range(deg + 1):
            q = deg - p
            gy = p * ky ** (p - 1) * kx ** q if p > 0 else np.zeros_like(ky)
            gx = q * ky ** p * kx ** (q - 1) if q > 0 else np.zeros_like(kx)
            cols.append(np.r_[gy, gx])
    return np.stack(cols, 1)


def fit_drift(s, pos, kn, higher=True):
    """s_n = B pos_n + c + curl-free(quadratic, cubic in k) + random.  Returns B (2x2: (row, col) <- pos), c, higher, random."""
    n = len(s); X = np.c_[pos, np.ones(n)]
    Z = np.zeros((2 * n, 6)); Z[:n, 0:2] = pos; Z[:n, 4] = 1; Z[n:, 2:4] = pos; Z[n:, 5] = 1
    G = np.c_[Z, _grad_basis(kn / np.abs(kn).max())] if higher else Z
    coef = np.linalg.lstsq(G, np.r_[s[:, 0], s[:, 1]], rcond=None)[0]
    B = np.array([[coef[0], coef[1]], [coef[2], coef[3]]]); c = np.array([coef[4], coef[5]])
    hi = (G[:, 6:] @ coef[6:]) if higher else np.zeros(2 * n); hi = np.stack([hi[:n], hi[n:]], 1)
    lin = pos @ B.T + c
    return B, c, hi, s - lin - hi


def image_frame_perm(Mk):
    """Signed permutation Pm with k ~ scale * Pm pos (maps stage axes onto image (row, col) axes)."""
    best = None
    for P in (np.eye(2), np.array([[0., 1.], [1., 0.]])):
        for sg in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            Pm = np.diag(sg) @ P; sc = np.trace(np.asarray(Mk) @ Pm.T)
            if best is None or sc > best[0]: best = (sc, Pm)
    return best[1]


def decompose(E):
    """E: dimensionless 2x2 (image shift / lens shift) in the image frame, rows/cols = (row, col).
    Returns percent values: scale_row, scale_col, isotropic, rotation (deg), shear, principal values and axis."""
    S = (E + E.T) / 2; w, v = np.linalg.eigh(S)
    ax = np.degrees(np.arctan2(v[0, 1], v[1, 1]))                 # angle of the larger principal axis, (col, row) frame
    return dict(scale_row_pct=100 * E[0, 0], scale_col_pct=100 * E[1, 1], isotropic_pct=100 * (E[0, 0] + E[1, 1]) / 2,
                rotation_deg=float(np.degrees((E[1, 0] - E[0, 1]) / 2)), shear_pct=100 * (E[0, 1] + E[1, 0]) / 2,
                row_from_col_axis_pct=100 * E[0, 1], col_from_row_axis_pct=100 * E[1, 0],
                principal_pct=[100 * float(w[0]), 100 * float(w[1])], principal_axis_deg=float((ax + 90) % 180 - 90),
                matrix_pct=(100 * E).tolist())


# ---------------------------------------------------------------- pupil-predicted drift
def pupil_poly(W, Pamp, dk, kc, order=4, centre_exclude=0.25):
    """Least-squares polynomial (degree 1..order in ky, kx) of the pupil phase inside the NA (centre excluded)."""
    M = W.shape[0]; yy, xx = kgrid(M, dk); r = np.hypot(yy, xx)
    m = np.isfinite(W) & (Pamp > 0.5) & (r < 0.95 * kc) & (r > centre_exclude * kc)
    terms = [(p, q) for deg in range(1, order + 1) for p in range(deg + 1) for q in [deg - p]]
    X = np.stack([yy[m] ** p * xx[m] ** q for p, q in terms] + [np.ones(m.sum())], 1)
    coef = np.linalg.lstsq(X, W[m], rcond=None)[0]
    resid = float(np.std(W[m] - X @ coef))
    return dict(terms=terms, coef=coef[:-1], resid_rad=resid)


def pupil_grad(poly, k):
    ky, kx = k[:, 0], k[:, 1]; gy = np.zeros(len(k)); gx = np.zeros(len(k))
    for (p, q), c in zip(poly["terms"], poly["coef"]):
        if p > 0: gy += c * p * ky ** (p - 1) * kx ** q
        if q > 0: gx += c * q * ky ** p * kx ** (q - 1)
    return np.stack([gy, gx], 1)


def pupil_shifts_px(poly, kn, pixel_um):
    """Image shift of illumination n caused by the pupil phase: -grad W(k_n) / (2 pi), in camera px (row, col)."""
    return -pupil_grad(poly, kn) / (2 * np.pi) / pixel_um


def defocus_astig_from_poly(poly, lam):
    """W = pi lam [ z (ky^2 + kx^2) + a0 (kx^2 - ky^2) + a1 2 kx ky ]  ->  (z, [a0, a1]) in um."""
    c = dict(zip(poly["terms"], poly["coef"])); cyy, cxx, cxy = c.get((2, 0), 0.0), c.get((0, 2), 0.0), c.get((1, 1), 0.0)
    z = (cyy + cxx) / (2 * np.pi * lam); a0 = (cxx - cyy) / (2 * np.pi * lam); a1 = cxy / (2 * np.pi * lam)
    return float(z), [float(a0), float(a1)]


# ---------------------------------------------------------------- model-based residual shifts
def model_shifts(Imod, R, keep, M, dk, N, band=(0.3, 3.5)):
    """Shift of each measured image relative to its model image (camera px); common mean removed."""
    q = np.fft.fftfreq(M, 1 / (M * dk)); QR = np.hypot(*np.meshgrid(q, q, indexing="ij")); bm = (QR >= band[0]) & (QR <= band[1])
    w = np.outer(tukey(M, 0.3), tukey(M, 0.3)); up = 4; Mu = M * up
    def xs(A, B):
        X = np.conj(np.fft.fft2((A - A.mean()) * w) * bm) * (np.fft.fft2((B - B.mean()) * w) * bm)
        C = np.zeros((Mu, Mu), complex); c0 = Mu // 2 - M // 2; C[c0:c0 + M, c0:c0 + M] = np.fft.fftshift(X)
        c = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(C)))); iy, ix = np.unravel_index(np.argmax(c), c.shape)
        par = lambda m, o, p: 0.5 * (m - p) / (m - 2 * o + p)
        return np.array([iy + par(c[iy - 1, ix], c[iy, ix], c[iy + 1, ix]) - Mu // 2, ix + par(c[iy, ix - 1], c[iy, ix], c[iy, ix + 1]) - Mu // 2]) / up
    sh = np.zeros((len(keep), 2))
    for j, i in enumerate(keep):
        Im = fcrop_intensity(np.asarray(R[i], np.float32), M); Im /= Im.mean()
        sh[j] = xs(Imod[j], Im) * (N / M)
    return sh - sh.mean(0)


def shift_stack(R, corr, order=3):
    return np.stack([ndi.shift(np.asarray(R[i], np.float32), -corr[i], order=order, mode="nearest") if np.any(corr[i])
                     else np.asarray(R[i], np.float32) for i in range(len(R))])


# ---------------------------------------------------------------- self test of the sign convention
def selftest_sign(log=print, seed=0):
    """Forward-model check of s_n = -grad W(k_n)/2pi with a pure astigmatic + defocus pupil on a random phase object."""
    import torch
    rng = np.random.default_rng(seed); M, No, dk, lam, kc = 96, 160, 0.1, 4.13e-5, 1.5
    am = ndi.gaussian_filter(rng.standard_normal((No, No)), 6.0); am = 0.2 * am / am.std()      # smooth amplitude object
    kn = np.array([[0.8, 0.0], [0.0, 0.8], [-0.6, 0.5], [0.4, -0.9]])
    yy, xx = kgrid(M, dk); Pamp = (np.hypot(yy, xx) <= kc).astype(float)
    z, a0, a1 = 20000.0, 8000.0, -5000.0
    W = np.pi * lam * (z * (yy ** 2 + xx ** 2) + a0 * (xx ** 2 - yy ** 2) + a1 * 2 * xx * yy)
    imgs = {}
    for tag, Wt in (("flat", np.zeros_like(W)), ("aber", W)):
        fb = nlr.FPMBand(np.ones((len(kn), M, M), np.float32), -kn / dk, No, M, kc / dk, (0.0, M), P0=Pamp * np.exp(1j * Wt),
                         a0=am.astype(np.float32), phi0=np.zeros((No, No), np.float32), direct_norm=False)
        with torch.no_grad():
            fb.W.copy_(torch.tensor(np.fft.ifftshift(Wt), dtype=torch.float32)); imgs[tag] = (fb.normalised() + 1).numpy()
    px = 1 / (M * dk)                                           # image pixel (um) on the M grid
    meas = np.array([xcorr_shift(imgs["flat"][n], imgs["aber"][n]) for n in range(len(kn))]) * px
    poly = pupil_poly(np.where(Pamp > 0, W, np.nan), Pamp, dk, kc, order=2, centre_exclude=0.0)
    pred = pupil_shifts_px(poly, kn, 1.0)                       # um
    ok = bool(np.abs(meas - pred).max() < 0.1 * np.abs(pred).max() + 0.2 * px)
    log(f"  sign self-test: measured {np.round(meas, 3).tolist()} um vs predicted {np.round(pred, 3).tolist()} um -> {'OK' if ok else 'FAILED'}")
    return ok, meas, pred



# ---------------------------------------------------------------- object focus (total defocus seen by the contrast)
def quad_poly(z, astig, lam):
    """polynomial dict of W = pi lam [ z (ky^2+kx^2) + a0 (kx^2-ky^2) + a1 2 kx ky ]"""
    return dict(terms=[(2, 0), (0, 2), (1, 1)], coef=np.array([np.pi * lam * (z - astig[0]), np.pi * lam * (z + astig[0]), 2 * np.pi * lam * astig[1]]), resid_rad=0.0)


def add_quadratic(poly, z, astig, lam, sign=1.0):
    """poly + sign * quadratic(z, astig)"""
    q = quad_poly(z, astig, lam); d = dict(zip(poly["terms"], poly["coef"]))
    for t_, c_ in zip(q["terms"], q["coef"]): d[t_] = d.get(t_, 0.0) + sign * c_
    return dict(terms=list(d.keys()), coef=np.array(list(d.values())), resid_rad=poly["resid_rad"])


def autofocus(a, phi, dxo, lam, band=(0.3, 3.5), frac=0.3, z_span=12000.0, z_step=1500.0):
    """Quadratic refocus (z, a0, a1) of the reconstructed object O = exp(-a + i phi) that makes it 'in focus':
    minimum of var(A - g P) / var(P) for the band-passed log-amplitude A and phase P (g fitted: a single-material
    object has A = g P in focus; a defocused object mixes phase into amplitude). Returns the refocus parameters
    z_H (um; kernel exp(+i pi lam [z |u|^2 + a0 (ux^2-uy^2) + a1 2 ux uy])) and the metric before / after.
    Pupil and object quadratic phases are degenerate in FPM (image contrast depends on their sum, the image drift on
    the pupil part only), so the total quadratic phase seen by the contrast is  W_total = W_pupil - refocus."""
    from scipy.optimize import minimize
    No = a.shape[0]; u = np.fft.fftfreq(No, dxo); UY, UX = np.meshgrid(u, u, indexing="ij"); Uq = np.hypot(UY, UX)
    bp = ((Uq > band[0]) & (Uq < band[1])).astype(float); Of = np.fft.fft2(np.exp(-a + 1j * phi))
    h = int(frac * No); c = No // 2; sl = (slice(c - h, c + h), slice(c - h, c + h))
    def metric(p):
        Oz = np.fft.ifft2(Of * np.exp(1j * np.pi * lam * (p[0] * Uq ** 2 + p[1] * (UX ** 2 - UY ** 2) + p[2] * 2 * UX * UY)))
        A = np.real(np.fft.ifft2(np.fft.fft2(-np.log(np.abs(Oz) + 1e-12)) * bp))[sl]
        P = np.real(np.fft.ifft2(np.fft.fft2(np.angle(Oz * np.exp(-1j * np.angle(Oz.mean())))) * bp))[sl]
        g = float((A * P).sum() / (P * P).sum()); return float(((A - g * P) ** 2).mean() / P.var())
    zs = np.arange(-z_span, z_span + 1, z_step); z0 = min(zs, key=lambda z_: metric([z_, 0.0, 0.0]))
    r = minimize(metric, [z0, 0.0, 0.0], method="Nelder-Mead", options=dict(xatol=30, fatol=1e-6, maxiter=500,
                 initial_simplex=[[z0, 0, 0], [z0 + z_step, 0, 0], [z0, z_step, 0], [z0, 0, z_step]]))
    return dict(z_um=float(r.x[0]), astig_um=[float(r.x[1]), float(r.x[2])], metric_before=metric([0.0, 0.0, 0.0]), metric_after=float(r.fun))

# ---------------------------------------------------------------- driver
def run(R, kn, pos, Mk, cfg, cal, log=print, stage_step_um=None):
    """R: preprocessed stack (n, N, N); kn (n, 2) solver-frame k; pos (n, 2) stage positions; Mk: k = Mk pos + k0.
    1. measured shifts s_n (pairwise network) -> linear 2x2 matrix B, higher orders, random part
    2. all measured shifts removed -> BLIS-FPM (fixed positions) -> pupil W_p and object O
    3. contrast: the image contrast fixes only the SUM of the pupil and object quadratic phases (the drift fixes the
       pupil part). The object is brought into focus (autofocus()); the total defocus / astigmatism
       Z_t = Z_p - refocus is assigned to the pupil (object in focus) together with the higher orders of W_p
    4. predicted parallax s_pred = -grad W_t(k_n)/2pi; every other measured shift is mechanical:
       the images are corrected by -(s_n - s_pred)
    4b. (focus_iter > 0) focus loop: object-only fit with the pupil fixed at Z_t; residual object defocus -> pupil, redo 4
    5. check: BLIS-FPM with fixed positions from Z_t: the new pupil must predict s_pred again, the new object must be
       in focus, and the model-based residual shifts must have no linear trend; their random part is applied last.
       (With the focus loop the free pupil of this check may keep a smaller polynomial defocus than Z_t -- 32a: the
       free pupil has a central phase step and a non-quadratic radial profile -- so check.ok can be False although the
       fixed-pupil model is self-consistent; see rep['focus_loop'].)"""
    pa = dict(DEFAULTS); pa.update(cfg.get("position_affine", {}))
    lam, px, kc = cfg["lam_um"], cfg["pixel_um"], cfg["kc"]
    step_um = cfg.get("stage_step_um", 1.0) if stage_step_um is None else stage_step_um
    thr = int(pa["threads"] or cfg["nonlinear"]["threads"])
    N = R.shape[-1]; n = len(R); t0 = time.time()
    ks = pa["fov_factor"] * cfg["kappa_nom"] * int(cal.get("twin", 1))
    Pm = image_frame_perm(Mk)
    to_E = lambda B: (np.asarray(B) @ Pm.T) * px / step_um
    maxE = lambda E: float(np.abs(E).max() * 100)
    pk = np.abs(pos - pos.mean(0)).max()
    s, net = network_shifts(R, kn, pa["n_neighbours"], threads=max(thr, 4), log=log)
    B, c, hi, rnd = fit_drift(s, pos, kn)
    dm = decompose(to_E(B))
    log(f"  measured linear drift: col<-col {dm['scale_col_pct']:+.3f} %, row<-row {dm['scale_row_pct']:+.3f} %, "
        f"col<-row {dm['col_from_row_axis_pct']:+.3f} %, row<-col {dm['row_from_col_axis_pct']:+.3f} %; "
        f"higher-order rms {np.sqrt((hi ** 2).mean(0)).round(2).tolist()} px, random rms {np.sqrt((rnd ** 2).mean(0)).round(2).tolist()} px")
    zeros = np.zeros((64, 64), np.float32)
    nl_cfg = json.loads(json.dumps(cfg)); nl_cfg["nonlinear"]["threads"] = thr
    fit = lambda Rin, cal_in: nlr.run_nonlinear(Rin, kn, nl_cfg, ks, cal_in, zeros, zeros, n_pupil=pa["n_pupil"], n_joint=pa["n_joint"], log=lambda m: None)
    drift_of = lambda poly: fit_drift(pupil_shifts_px(poly, kn, px), pos, kn)[0]
    zstr = lambda z, a: f"z {z/1e3:+.2f} mm, astig ({a[0]/1e3:+.2f}, {a[1]/1e3:+.2f}) mm"
    # 2 -- everything aligned
    cal0 = dict(cal); cal0["defocus_um"], cal0["astig_um"] = 0.0, [0.0, 0.0]
    res1 = fit(shift_stack(R, s), cal0)
    polyP = pupil_poly(res1["W"], res1["Pamp"], res1["dk"], kc, pa["poly_order"], pa["centre_exclude"]); zP, aP = defocus_astig_from_poly(polyP, lam)
    af1 = autofocus(res1["a"], res1["phi"], float(res1["dxo"]), lam, band=cfg["nonlinear"]["q_band"])
    # 3 -- total (contrast) focus assigned to the pupil
    polyT = add_quadratic(polyP, af1["z_um"], af1["astig_um"], lam, sign=-1.0); zT, aT = defocus_astig_from_poly(polyT, lam)
    log(f"  aligned stack: pupil {zstr(zP, aP)}; object refocus {zstr(af1['z_um'], af1['astig_um'])} "
        f"(metric {af1['metric_before']:.3f} -> {af1['metric_after']:.3f}) => contrast focus {zstr(zT, aT)} ({time.time() - t0:.0f} s)")
    # 4 -- mechanical correction
    def correct(polyT_):
        spT_ = pupil_shifts_px(polyT_, kn, px); spT_ = spT_ - spT_.mean(0); corr_ = s - spT_
        return spT_, drift_of(polyT_), corr_, shift_stack(R, corr_)
    spT, B_pup, corr, Rc = correct(polyT)
    # 4b (optional) -- focus loop: object solved with the pupil FIXED at the contrast focus (and the data keeping its parallax);
    #    a residual object defocus means the contrast wants a different pupil focus: move it into the pupil, redo 4, repeat
    loop = []
    if pa["focus_iter"]:
        fitF = lambda Rin, cal_in: nlr.run_nonlinear(Rin, kn, nl_cfg, ks, cal_in, zeros, zeros, n_pupil=0, n_joint=pa["n_joint"], log=lambda m: None, fix_pupil=True)
        for it_ in range(int(pa["focus_iter"]) + 1):
            calL = dict(cal); calL["defocus_um"], calL["astig_um"] = zT, aT
            resL = fitF(Rc, calL); afL = autofocus(resL["a"], resL["phi"], float(resL["dxo"]), lam, band=cfg["nonlinear"]["q_band"])
            loop.append(dict(pupil_defocus_um=zT, pupil_astig_um=list(aT), object_refocus=afL, misfit=float(resL["hist"][-1])))
            log(f"  focus loop {it_}: fixed pupil {zstr(zT, aT)} -> object refocus {zstr(afL['z_um'], afL['astig_um'])}, misfit {resL['hist'][-1]:.4f} ({time.time() - t0:.0f} s)")
            if max(abs(afL["z_um"]), *map(abs, afL["astig_um"])) < pa["focus_tol_um"] or it_ == int(pa["focus_iter"]):
                break
            polyT = add_quadratic(polyT, afL["z_um"], afL["astig_um"], lam, sign=-1.0); zT, aT = defocus_astig_from_poly(polyT, lam)
            spT, B_pup, corr, Rc = correct(polyT)
    Bm, _, him, rndm = fit_drift(corr, pos, kn)
    # 5 -- check with fixed positions
    calT = dict(cal); calT["defocus_um"], calT["astig_um"] = zT, aT
    def check(Rin):
        res = fit(Rin, calT)
        polyF = pupil_poly(res["W"], res["Pamp"], res["dk"], kc, pa["poly_order"], pa["centre_exclude"]); zF, aF = defocus_astig_from_poly(polyF, lam)
        af = autofocus(res["a"], res["phi"], float(res["dxo"]), lam, band=cfg["nonlinear"]["q_band"])
        keep = np.asarray(res["keep"]); ms = model_shifts(res["Imod"], Rin, keep, int(res["M"]), float(res["dk"]), N)
        Br, _, _, rr = fit_drift(ms, pos[keep], kn[keep], higher=False)
        polyTF = add_quadratic(polyF, af["z_um"], af["astig_um"], lam, sign=-1.0)
        d = dict(pupil=dict(defocus_um=zF, astig_um=aF), object_refocus=af, misfit=float(res["hist"][-1]),
                 pupil_drift_minus_retained_pct=maxE(to_E(drift_of(polyF)) - to_E(B_pup)),
                 object_focus_drift_pct=maxE(to_E(drift_of(quad_poly(af["z_um"], af["astig_um"], lam)))),
                 total_focus_drift_minus_retained_pct=maxE(to_E(drift_of(polyTF)) - to_E(B_pup)),
                 residual_linear_drift_pct=maxE(to_E(Br)), residual_linear_drift=decompose(to_E(Br)),
                 model_shift_rms_px=np.sqrt((ms ** 2).mean(0)).tolist(), random_rms_px=np.sqrt((rr ** 2).mean(0)).tolist())
        d["residual_linear_drift_max_px"] = d["residual_linear_drift_pct"] / 100 * pk * step_um / px
        log(f"  check: pupil {zstr(zF, aF)}, object refocus {zstr(af['z_um'], af['astig_um'])}, misfit {res['hist'][-1]:.4f}; "
            f"pupil drift - retained {d['pupil_drift_minus_retained_pct']:.3f} %, (pupil - refocus) drift - retained "
            f"{d['total_focus_drift_minus_retained_pct']:.3f} %, model residual linear drift {d['residual_linear_drift_pct']:.3f} % "
            f"({d['residual_linear_drift_max_px']:.2f} px), random rms {np.round(d['random_rms_px'], 2).tolist()} px ({time.time() - t0:.0f} s)")
        return res, d, keep, rr
    resF, chk, keep, rr = check(Rc)
    chk_first = chk
    if pa["model_refine"]:
        corr2 = np.zeros((n, 2)); corr2[keep] = rr; corr = corr + corr2; Rc = shift_stack(R, corr)
        resF, chk, keep, rr = check(Rc)
    w = pa["warn_percent"]
    # the solver may split the quadratic phase between pupil and object (degenerate); the observable is their sum
    ok = bool(chk["total_focus_drift_minus_retained_pct"] < w and chk["residual_linear_drift_pct"] < w)
    Emeas, Epup = to_E(B), to_E(B_pup); Emech = Emeas - Epup
    rep = dict(settings=pa, network=net, image_frame_permutation=Pm.tolist(), stage_step_um=step_um,
               measured=decompose(Emeas), pupil_predicted=decompose(Epup), mechanical=decompose(Emech),
               B_measured_px_per_stage=B.tolist(), B_pupil_px_per_stage=B_pup.tolist(), B_mech_px_per_stage=(B - B_pup).tolist(),
               higher_order_rms_px=np.sqrt((hi ** 2).mean(0)).tolist(), random_rms_px=np.sqrt((rnd ** 2).mean(0)).tolist(),
               mechanical_higher_order_rms_px=np.sqrt((him ** 2).mean(0)).tolist(), mechanical_random_rms_px=np.sqrt((rndm ** 2).mean(0)).tolist(),
               pupil_parallax_rms_px=np.sqrt((spT ** 2).mean(0)).tolist(),
               aligned_stack=dict(pupil=dict(defocus_um=zP, astig_um=aP), object_refocus=af1, misfit=float(res1["hist"][-1])),
               contrast_focus=dict(defocus_um=zT, astig_um=aT, note="pupil quadratic of the aligned-stack fit minus the object refocus "
                                   "(image contrast fixes pupil + object quadratic phase; the object is taken in focus)"
                                   + ("; then corrected by the focus loop (object solved with the pupil fixed, residual object defocus moved into the pupil)" if loop else "")),
               focus_loop=loop, check_first=chk_first, check=dict(chk, warn_percent=w, ok=ok),
               correction_px=corr.tolist(), measured_shifts_px=s.tolist(), pupil_shifts_px=spT.tolist(),
               pupil_defocus_um=zT, pupil_astig_um=aT, pupil_poly_resid_rad=polyP["resid_rad"], consistent=ok,
               fraction_of_drift_explained_by_pupil=float(np.linalg.norm(Epup) / max(np.linalg.norm(Emeas), 1e-12)))
    log(f"  mechanical drift: col<-col {rep['mechanical']['scale_col_pct']:+.3f} %, row<-row {rep['mechanical']['scale_row_pct']:+.3f} %, "
        f"col<-row {rep['mechanical']['col_from_row_axis_pct']:+.3f} %, row<-col {rep['mechanical']['row_from_col_axis_pct']:+.3f} % "
        f"(rotation {rep['mechanical']['rotation_deg']:+.3f} deg, shear {rep['mechanical']['shear_pct']:+.3f} %); "
        f"pupil part {100 * rep['fraction_of_drift_explained_by_pupil']:.0f} % of the measured drift; consistency {'OK' if ok else 'WARNING'}")
    cal_upd = dict(defocus_um=zT, astig_um=aT, pupil_calibration_source="posaffine: contrast focus (object in focus) after mechanical drift correction",
                   position_affine=dict(B_mech_px_per_stage=(B - B_pup).tolist(), B_pupil_px_per_stage=B_pup.tolist(),
                                        B_measured_px_per_stage=B.tolist(), image_frame_permutation=Pm.tolist(), consistent=ok))
    return Rc, cal_upd, rep, resF


def check_pupil_vs_data(W, Pamp, dk, kc, kn, pos, B_retained, Pm, pixel_um, stage_step_um=1.0, order=4, centre_exclude=0.25, warn_percent=0.1):
    """Per-reconstruction check: linear drift predicted by this pupil vs the drift retained in the corrected data."""
    poly = pupil_poly(W, Pamp, dk, kc, order, centre_exclude)
    Bp, *_ = fit_drift(pupil_shifts_px(poly, kn, pixel_um), pos, kn)
    to_E = lambda B: (np.asarray(B) @ np.asarray(Pm).T) * pixel_um / stage_step_um
    Ep, Er = to_E(Bp), to_E(B_retained); d = Ep - Er; pk = np.abs(pos - pos.mean(0)).max()
    return dict(predicted=decompose(Ep), retained=decompose(Er), max_diff_pct=float(np.abs(d).max() * 100),
                max_diff_px_over_scan=float(np.abs(d).max() * pk * stage_step_um / pixel_um), ok=bool(np.abs(d).max() * 100 < warn_percent))


def fig_position_affine(path, rep, pos, pixel_um):
    """(a) per-image shifts at the lens positions: measured (grey) and predicted by the pupil (green);
    (b) the 2x2 drift matrix (image shift / lens shift, %): measured, pupil-predicted and mechanical (= corrected)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    step = float(rep.get("stage_step_um") or 1.0)
    P = np.asarray(pos, float) * step; Pm = np.asarray(rep["image_frame_permutation"], float)
    ms = np.asarray(rep["measured_shifts_px"], float); ps = np.asarray(rep["pupil_shifts_px"], float)
    # arrows drawn in the lens frame: the stage axis mapped to image columns is the horizontal axis
    ic = int(np.argmax(np.abs(Pm[1])))                  # stage axis that drives image columns
    ir = 1 - ic
    x, y = P[:, ic], P[:, ir]
    ext = max(np.ptp(x), np.ptp(y)); smax = np.abs(ms).max()
    sc = 0.2 * ext / max(smax, 1e-9)                    # longest arrow = 20 % of the scan extent
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.5))
    ax = axs[0]
    ax.scatter(x, y, s=3, color="0.25", zorder=3, lw=0)
    ax.quiver(x, y, ms[:, 1], ms[:, 0], color="0.6", angles="xy", scale_units="xy", scale=1 / sc, width=0.0045, headwidth=3.5, label="measured shift", zorder=2)
    ax.quiver(x, y, ps[:, 1], ps[:, 0], color="#1b7837", angles="xy", scale_units="xy", scale=1 / sc, width=0.0065, headwidth=3.0, label="pupil parallax (kept)", zorder=4)
    L = 10 ** np.floor(np.log10(smax)); L = L * (5 if smax / L > 5 else 2 if smax / L > 2 else 1)
    x0, y0 = x.max() - L * sc, y.max() + 0.13 * ext
    ax.annotate("", xy=(x0 + L * sc, y0), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="0.2", lw=0.8))
    ax.text(x0 + L * sc / 2, y0 + 0.03 * ext, f"{L:g} px", ha="center", va="bottom", fontsize=7)
    ax.set_xlim(x.min() - 0.12 * ext, x.max() + 0.25 * ext); ax.set_ylim(y.min() - 0.12 * ext, y.max() + 0.25 * ext)
    ax.set_aspect("equal"); ax.set_xlabel("lens position, column axis (µm)"); ax.set_ylabel("lens position, row axis (µm)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, handlelength=1.2, columnspacing=1.0)
    
    ax = axs[1]
    keys = [("scale_col_pct", "col ← col axis"), ("scale_row_pct", "row ← row axis"), ("col_from_row_axis_pct", "col ← row axis"), ("row_from_col_axis_pct", "row ← col axis")]
    grp = [("measured", "measured", "0.55"), ("pupil_predicted", "pupil (kept)", "#1b7837"), ("mechanical", "mechanical (corrected)", "#d95f02")]
    yy = np.arange(len(keys))[::-1]; h = 0.26
    for j, (g_, lab_, col_) in enumerate(grp):
        v = np.array([rep[g_][k] for k, _ in keys])
        ax.barh(yy + (1 - j) * h, v, height=h, color=col_, label=lab_)
        for yv, vv in zip(yy + (1 - j) * h, v):
            ax.text(vv + (0.04 if vv >= 0 else -0.04), yv, "0.00" if abs(vv) < 0.005 else f"{vv:+.2f}", va="center", ha="left" if vv >= 0 else "right", fontsize=6.5)
    ax.axvline(0, color="0.2", lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels([l for _, l in keys])
    lim = max(abs(rep[g_][k]) for g_, _, _ in grp for k, _ in keys) * 1.45
    ax.set_xlim(-lim, lim); ax.set_xlabel("image shift / lens shift (%)")
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, handlelength=1.0, columnspacing=0.8)
    
    ck = rep["check"]; mm = rep["mechanical"]
    txt = (f"mechanical: rotation {mm['rotation_deg']:+.2f}°, shear {mm['shear_pct']:+.2f} %\n"
           f"pupil (contrast focus): z {rep['pupil_defocus_um'] / 1e3:+.2f} mm, astig ({rep['pupil_astig_um'][0] / 1e3:+.2f}, {rep['pupil_astig_um'][1] / 1e3:+.2f}) mm\n"
           f"check: (pupil − object refocus) drift − kept {ck['total_focus_drift_minus_retained_pct']:.3f} %, "
           f"residual linear {ck['residual_linear_drift_pct']:.3f} % → {'consistent' if rep['consistent'] else 'NOT consistent'}")
    fig.canvas.draw()
    for a_, lt in zip(axs, "ab"):
        bb = a_.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        fig.text(bb.x0, axs[0].get_position().y1 + 0.1, lt, fontsize=10, fontweight="bold", va="bottom", ha="left")
    fig.text(axs[0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).x0, -0.02, txt.replace("-0.00", "0.00").replace(", 0.00)", ", 0.00)"), fontsize=6.8, va="top", linespacing=1.35)
    fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)
