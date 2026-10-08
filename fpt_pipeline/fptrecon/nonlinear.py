"""Step 4: nonlinear Fourier ptychographic reconstruction.

Forward model (solver frame, same as wotf.py):
    O_eff(x)  = O(x) * Q(x),   O = exp(-a + i phi),   Q = exp(i pi kappa |x|^2)   (FOV effect; kappa=0 -> plane wave)
    psi_n     = F^-1{ P(k) * Ohat_eff(k - k_n) }                 on an M x M spectral patch
    psi_n^dir = F^-1{ P(k) * Qhat(k - k_n) }                     (modelled flat field, O = 1)
    m_n       = (|psi_n|^2 / |psi_n^dir|^2) / mean - 1           (flat-fielded, normalised model)
Loss = sum_n sum_{q in band} | F{w m_n}(q) - F{w (I_n - 1)}(q) |^2 / sum |F{w (I_n-1)}|^2
minimised with L-BFGS over (a, phi) and the pupil phase W(k) (|P| fixed to the NA disc).
"""
import time

import numpy as np
import torch
from scipy.signal.windows import tukey

from .optics import analytic_W, fast_even, fcrop_intensity, fresample, kgrid

CT = torch.complex64


class FPMBand:
    def __init__(self, I_meas, s_px, N, M, kc_px, band_px, Q=None, a0=None, phi0=None, P0=None,
                 support_scale=1.05, window=None, direct_norm=True, dir_floor=0.1, dark=None, wmask=None, img_weights=None,
                 modes=None, mode_w=None, dark_gain=None, free_shifts=False):
        """Optional (036 Phase C, dark-field images; all None = original behaviour):
        dark: (n,) bool, images whose direct beam lies outside the pupil -> no direct-beam normalisation;
        wmask: (n, M, M) validity weights (multiply the window; contrast taken about the masked mean);
        img_weights: (n,) weights lambda_n of the images in the loss (e.g. inverse noise power).
        free_shifts: (posaffine step) one free translation per image (px of the M grid), applied as a phase ramp on the
        model spectrum; the pupil is then determined by the image contrast only (its parallax is absorbed by the shifts).
        Partial coherence (036; None = coherent, original behaviour):
        modes: (J, 2) illumination-mode offsets in the units of s_px (spectral pixels, same sign convention);
        mode_w: (J,) or (n, J) mode weights (normalised to sum 1 per image).  The image is the weighted sum of the
        mode intensities; the modelled direct beam uses the same modes.  Dark images (dark=True) are then modelled
        as (I - I_direct), matching data from which the sample-out image was subtracted."""
        self.N, self.M = N, M
        self.ext = dark is not None or wmask is not None or img_weights is not None
        self.direct_norm, self.dir_floor = direct_norm, dir_floor
        self.s = np.round(np.asarray(s_px)).astype(int)
        self.nimg = len(self.s)
        c, h = N // 2, M // 2
        self.y0 = c - h + self.s[:, 0]
        self.x0 = c - h + self.s[:, 1]
        self.pc = modes is not None
        self.dark_gain = dark_gain       # "profile": least-squares contrast factor per dark image (additive background, level)
        if self.pc:
            md = np.round(np.asarray(modes, float)).astype(int)          # (J, 2) common or (n, J, 2) per image
            if md.ndim == 2: md = np.broadcast_to(md, (self.nimg,) + md.shape)
            wj = np.asarray(mode_w, float); wj = np.broadcast_to(wj, (self.nimg, md.shape[1])).copy()
            wj /= wj.sum(1, keepdims=True)
            self.mw = torch.tensor(wj, dtype=torch.float32)
            self.my0 = self.y0[:, None] + md[:, :, 0]; self.mx0 = self.x0[:, None] + md[:, :, 1]
            self.nmode = md.shape[1]
            self.mact = [np.where(wj[n] > 0)[0] for n in range(self.nimg)]   # skip zero-weight padding
            nn, jj = np.nonzero(wj > 0)                                       # flattened active (image, mode) pairs
            ar = np.arange(M)
            self.g_iy = torch.tensor((self.my0[nn, jj][:, None] + ar[None]).astype(np.int64))
            self.g_ix = torch.tensor((self.mx0[nn, jj][:, None] + ar[None]).astype(np.int64))
            self.g_n = torch.tensor(nn.astype(np.int64)); self.g_w = torch.tensor(wj[nn, jj], dtype=torch.float32)
            self.Id_cache = None
            assert self.my0.min() >= 0 and self.my0.max() + M <= N and self.mx0.min() >= 0 and self.mx0.max() + M <= N, \
                "object grid too small for the illumination modes (increase No)"
        assert self.y0.min() >= 0 and self.y0.max() + M <= N and self.x0.min() >= 0 and self.x0.max() + M <= N, \
            "object grid too small for the illumination shifts (increase No)"
        win = np.outer(np.hanning(M), np.hanning(M)) if window is None else window
        self.win = torch.tensor(win.astype(np.float32))
        yy, xx = np.mgrid[-(M // 2):M - M // 2, -(M // 2):M - M // 2]
        kr = np.hypot(yy, xx)
        bmask = np.fft.ifftshift((kr >= band_px[0]) & (kr <= band_px[1]))
        self.band = torch.tensor(bmask)
        if self.ext:
            n_ = len(self.s)
            self.dark = torch.tensor(np.zeros(n_, bool) if dark is None else np.asarray(dark, bool))
            wm = np.ones((n_, M, M), np.float32) if wmask is None else np.asarray(wmask, np.float32)
            self.wm = torch.tensor(wm)
            self.winn = self.win[None] * self.wm
            lam = np.ones(n_) if img_weights is None else np.asarray(img_weights, float)
            self.lam = torch.tensor(lam, dtype=torch.float64)
            Im = np.asarray(I_meas, np.float32)
            if modes is not None:         # partial coherence: window-weighted mean (grid-edge diffraction of Q ignored)
                wwm = wm * win[None]
                Im = Im / ((Im * wwm).sum((1, 2), keepdims=True) / wwm.sum((1, 2), keepdims=True))
            else:
                Im = Im / ((Im * wm).sum((1, 2), keepdims=True) / wm.sum((1, 2), keepdims=True))
            D = np.fft.fft2((Im - 1.0) * self.winn.numpy())
            self.D = torch.tensor(D[:, bmask], dtype=CT)
            self.Dnorm = float((self.lam * (self.D.abs() ** 2).sum(1).double()).sum())
        else:
            D = np.fft.fft2((np.asarray(I_meas, np.float32) - 1.0) * win)
            self.D = torch.tensor(D[:, bmask], dtype=CT)
            self.Dnorm = float((self.D.abs() ** 2).sum())
        self.sup = torch.tensor(np.fft.ifftshift(kr <= kc_px * support_scale).astype(np.float32))
        self.Q = torch.tensor(np.ones((N, N), complex) if Q is None else Q, dtype=CT)
        self.Qhat = torch.fft.fftshift(torch.fft.fft2(self.Q, norm="ortho"))
        self.a = torch.tensor(np.zeros((N, N)) if a0 is None else a0, dtype=torch.float32, requires_grad=True)
        self.phi = torch.tensor(np.zeros((N, N)) if phi0 is None else phi0, dtype=torch.float32, requires_grad=True)
        if P0 is None:
            P0 = (kr <= kc_px).astype(complex)
        P0u = np.fft.ifftshift(P0)
        self.Pamp = torch.tensor(np.abs(P0u), dtype=torch.float32)                 # fixed amplitude
        self.W = torch.tensor(np.angle(P0u) * (np.abs(P0u) > 0), dtype=torch.float32, requires_grad=True)
        self.scale = M / N
        self.hist = []
        self.free_shifts = bool(free_shifts)
        self.t = torch.zeros((self.nimg, 2), dtype=torch.float32, requires_grad=True)
        fq = np.fft.fftfreq(M)
        self.fy = torch.tensor(np.broadcast_to(fq[:, None], (M, M))[bmask], dtype=torch.float32)
        self.fx = torch.tensor(np.broadcast_to(fq[None, :], (M, M))[bmask], dtype=torch.float32)

    # -- model ------------------------------------------------------------------------------
    def P(self):
        return self.sup * self.Pamp * torch.exp(1j * self.W)

    def _field(self, spec, P):
        if self.pc:                       # mixed state: weighted sum of the mode intensities (batched patch gather)
            out = torch.zeros((self.nimg, self.M, self.M), dtype=torch.float32)
            G = self.g_iy.shape[0]; step = 256
            for g0 in range(0, G, step):
                sl = slice(g0, min(G, g0 + step))
                pt = spec[self.g_iy[sl][:, :, None], self.g_ix[sl][:, None, :]]
                psi = torch.fft.ifft2(P[None] * torch.fft.ifftshift(pt, dim=(-2, -1)), norm="ortho") * self.scale
                out = out.index_add(0, self.g_n[sl], self.g_w[sl][:, None, None] * (psi.real ** 2 + psi.imag ** 2))
            return out
        patches = torch.stack([spec[self.y0[n]:self.y0[n] + self.M, self.x0[n]:self.x0[n] + self.M]
                               for n in range(self.nimg)])
        patches = torch.fft.ifftshift(patches, dim=(-2, -1))
        psi = torch.fft.ifft2(P[None] * patches, norm="ortho") * self.scale
        return psi.real ** 2 + psi.imag ** 2

    def normalised(self):
        P = self.P()
        Oe = torch.exp(-self.a + 1j * self.phi) * self.Q
        I = self._field(torch.fft.fftshift(torch.fft.fft2(Oe, norm="ortho")), P)
        if self.ext:
            if self.pc:                   # partial coherence: bright images / direct beam, dark images - direct beam
                if getattr(self, "cache_direct", False):
                    if self.Id_cache is None: self.refresh_direct()
                    Id = self.Id_cache
                else:
                    Id = self._field(self.Qhat, P)
                if self.direct_norm and not bool(self.dark.all()):
                    den = torch.clamp(Id, min=self.dir_floor * float(Id.detach()[~self.dark].max()))
                else:
                    den = torch.ones_like(Id)
                I = torch.where(self.dark[:, None, None], I - Id, I / den)
                mu = (I * self.winn).sum((1, 2), keepdim=True) / self.winn.sum((1, 2), keepdim=True)
                return I / torch.clamp(mu, min=1e-12) - 1.0
            if self.direct_norm and not bool(self.dark.all()):
                Id = self._field(self.Qhat, P)
                den = torch.clamp(Id, min=self.dir_floor * float(Id.detach()[~self.dark].max()))
                I = I / torch.where(self.dark[:, None, None], torch.ones_like(den), den)
            return I / ((I * self.wm).sum((1, 2), keepdim=True) / self.wm.sum((1, 2), keepdim=True)) - 1.0
        if self.direct_norm:
            Id = self._field(self.Qhat, P)
            I = I / torch.clamp(Id, min=self.dir_floor * float(Id.detach().max()))
        return I / I.mean((1, 2), keepdim=True) - 1.0

    def refresh_direct(self):
        """Recompute the cached mixed-state direct beam with the current pupil (no gradient through it)."""
        with torch.no_grad():
            self.Id_cache = self._field(self.Qhat, self.P())

    def _residual(self):
        F = torch.fft.fft2(self.normalised() * (self.winn if self.ext else self.win[None]))[:, self.band]
        if self.free_shifts:                                  # model image shifted by t_n (M-grid px)
            ph = -2 * np.pi * (self.t[:, :1] * self.fy[None] + self.t[:, 1:] * self.fx[None])
            F = F * torch.exp(1j * ph)
        if self.ext and self.dark_gain == "profile" and bool(self.dark.any()):
            cg = torch.clamp((F.conj() * self.D).real.sum(1) / ((F.real ** 2 + F.imag ** 2).sum(1) + 1e-30), min=0.0)
            F = F * torch.where(self.dark, cg, torch.ones_like(cg))[:, None]
        return F - self.D

    def dark_gains(self):
        with torch.no_grad():
            F = torch.fft.fft2(self.normalised() * self.winn)[:, self.band]
            return torch.clamp((F.conj() * self.D).real.sum(1) / ((F.real ** 2 + F.imag ** 2).sum(1) + 1e-30), min=0.0).numpy()

    def loss(self):
        r = self._residual()
        if self.ext:
            return (self.lam * (r.real.double() ** 2 + r.imag.double() ** 2).sum(1)).sum() / self.Dnorm
        return (r.real.double() ** 2 + r.imag.double() ** 2).sum() / self.Dnorm

    def per_image_loss(self):
        with torch.no_grad():
            r = self._residual()
            return ((r.abs() ** 2).sum(1) / (self.D.abs() ** 2).sum(1)).numpy()

    # -- optimisation -------------------------------------------------------------------------
    def run(self, n_iter, params=("obj",), log=print, every=10, history_size=20):
        plist = []
        if "obj" in params: plist += [self.a, self.phi]
        if "pupil_phase" in params: plist += [self.W]
        if "shifts" in params: plist += [self.t]

        def make():
            return torch.optim.LBFGS(plist, lr=1, max_iter=1, history_size=history_size, line_search_fn="strong_wolfe",
                                     tolerance_grad=1e-14, tolerance_change=1e-16)
        opt = make()

        def closure():
            opt.zero_grad()
            L = self.loss()
            L.backward()
            return L
        stall = 0
        for it in range(n_iter):
            if getattr(self, "cache_direct", False) and self.pc and "pupil_phase" in params and it % self.refresh == 0:
                self.refresh_direct(); opt = make(); stall = 0
            L = float(opt.step(closure).detach())
            self.hist.append(L)
            if len(self.hist) > 1 and abs(self.hist[-2] - L) < 1e-9 * abs(L):
                stall += 1
                if stall >= 2:                      # line search stalled -> reset quasi-Newton memory
                    opt = make(); stall = 0
            else:
                stall = 0
            if every and (it % every == 0 or it == n_iter - 1):
                log(f"    it {it:4d}  misfit {L:.5f}")
        return self


def grid_sizes(cfg, kn, dk):
    nl = cfg["nonlinear"]; kc = cfg["kc"]
    M = nl["M"] or fast_even((2 * nl["support_scale"] * kc + nl["q_band"][1]) / dk)
    smax = np.abs(kn / dk).max()
    No = nl["No"] or fast_even(M + 2 * np.ceil(smax) + 8)
    return int(M), int(No)


def run_nonlinear(R, kn, cfg, kappa, calib, a_init, phi_init, n_pupil=None, n_joint=None, log=print, free_shifts=False, fix_pupil=False):
    """R: preprocessed stack (camera grid). kn: (n,2) solver-frame illumination k. kappa: solver-frame FOV coefficient.
    fix_pupil: (posaffine focus loop) keep the pupil at the calibration defocus / astigmatism, fit the object only."""
    nl = cfg["nonlinear"]; lam, dx, kc = cfg["lam_um"], cfg["pixel_um"], cfg["kc"]
    torch.set_num_threads(int(nl["threads"]))
    n_pupil = nl["n_pupil"] if n_pupil is None else n_pupil
    n_joint = nl["n_joint"] if n_joint is None else n_joint
    keep = np.where(np.hypot(kn[:, 0], kn[:, 1]) / kc <= nl["max_k_over_kc"])[0]
    Ncam = R.shape[-1]; dk = 1 / (Ncam * dx)
    M, No = grid_sizes(cfg, kn[keep], dk)
    dxo = Ncam * dx / No
    I = fcrop_intensity(np.clip(R[keep], *cfg["clip"]), M)
    I = I / I.mean((1, 2), keepdims=True)
    y = (np.arange(No) - No / 2) * dxo
    Y, X = np.meshgrid(y, y, indexing="ij")
    Q = np.exp(1j * np.pi * kappa * (X ** 2 + Y ** 2)) if kappa != 0 else None
    z, astig = calib["defocus_um"], calib["astig_um"]
    W0 = analytic_W(M, dk, lam, z, astig)
    yy, xx = kgrid(M, dk)
    P0 = (np.hypot(yy, xx) <= nl["pupil_init_scale"] * kc) * np.exp(1j * W0)
    win = np.outer(tukey(M, nl["tukey_alpha"]), tukey(M, nl["tukey_alpha"]))
    band = (nl["q_band"][0] / dk, nl["q_band"][1] / dk)
    fb = FPMBand(I, -kn[keep] / dk, No, M, kc / dk, band, Q=Q, P0=P0, window=win,
                 support_scale=nl["support_scale"], a0=fresample(a_init, No).astype(np.float32),
                 phi0=fresample(phi_init, No).astype(np.float32), direct_norm=True, dir_floor=nl["dir_floor"], free_shifts=free_shifts)
    with torch.no_grad():                                    # unwrapped initial pupil phase
        fb.W.copy_(torch.tensor(np.fft.ifftshift(W0), dtype=torch.float32))
    t0 = time.time()
    L0 = float(fb.loss().detach())
    log(f"  images {len(keep)}/{len(kn)}, M={M}, No={No} (object pixel {dxo*1e3:.1f} nm), initial misfit {L0:.4f}")
    if n_pupil and not fix_pupil:
        fb.run(n_pupil, ("pupil_phase",) + (("shifts",) if free_shifts else ()), log=log); log(f"  pupil stage done: {fb.hist[-1]:.4f} ({time.time()-t0:.0f} s)")
    if n_joint:
        grp = ("obj",) if fix_pupil else ("obj", "pupil_phase")
        fb.run(n_joint, grp + (("shifts",) if free_shifts else ()), log=log); log(f"  {'object' if fix_pupil else 'joint'} stage done: {fb.hist[-1]:.4f} ({time.time()-t0:.0f} s)")
    with torch.no_grad():
        Imod = (fb.normalised() + 1.0).numpy()
    return dict(a=fb.a.detach().numpy(), phi=fb.phi.detach().numpy(),
                W=np.fft.fftshift(fb.W.detach().numpy()), W0=W0, Pamp=np.fft.fftshift(fb.Pamp.numpy() * fb.sup.numpy()),
                hist=np.array(fb.hist), L0=L0, per_image=fb.per_image_loss(), Imod=Imod.astype(np.float32),
                Imeas=I.astype(np.float32), keep=keep, kappa=kappa, dxo=dxo, M=M, No=No, dk=dk, band=np.array(band),
                shifts_px=fb.t.detach().numpy() * (Ncam / M))
