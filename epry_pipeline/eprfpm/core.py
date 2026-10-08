"""EPRY-FPM solver (sequential, per-image projection updates of the object spectrum and the pupil).

Algorithm (Ou, Zheng & Yang, Opt. Express 22, 4960 (2014)), for each image n in order of increasing |k_n|:
    Phi_n  = P_n . S_n                       S_n = object-spectrum patch at -k_n,  P_n = pupil at q + e_n
    psi_n  = F^-1{Phi_n}
    psi'_n = psi_n + w . (A_n psi_n/|psi_n| - psi_n)      amplitude replacement, A_n = sqrt(g_n I_Q,n I_meas,n)
    dPhi   = F{psi'_n} - Phi_n
    S_n   += alpha conj(P_n)/max|P_n|^2 dPhi                 (object spectrum)
    P     += beta  B_n^T[ conj(S_n)/max|S_n|^2 dPhi ]         (pupil; B_n = sub-pixel shift by e_n)
Conventions identical to dipfpm.physics.FPMPhysics (solver frame, patch offsets, sub-pixel remainder e_n
applied to the pupil, FOV quadratic phase Q with direct-beam normalisation I_Q).
Additions to the published algorithm (all switchable in the config):
  * sub-pixel illumination positions (bilinear pupil shift and its adjoint),
  * per-image intensity factor g_n (least squares after each sweep, mean 1),
  * adaptive step size (Zuo, Sun & Chen, Opt. Express 24, 20724 (2016)): alpha, beta halved when the
    amplitude error decreases by less than step_eps in a sweep,
  * Tukey-weighted amplitude replacement w (suppresses the crop-edge discontinuity),
  * held-out images (not used for updates) for choosing the sweep, same rule as dipfpm.
"""
import time

import numpy as np
from scipy import fft as sfft
from scipy.signal.windows import tukey


def shift_bilinear(P, fy, fx):
    """P(i + fy, j + fx) for |f| <= 1 (bilinear)."""
    sy, sx = (1 if fy >= 0 else -1), (1 if fx >= 0 else -1); ay, ax = abs(fy), abs(fx)
    Py = (1 - ay) * P + ay * np.roll(P, -sy, axis=0)
    return (1 - ax) * Py + ax * np.roll(Py, -sx, axis=1)


def shift_bilinear_adj(G, fy, fx):
    sy, sx = (1 if fy >= 0 else -1), (1 if fx >= 0 else -1); ay, ax = abs(fy), abs(fx)
    Gx = (1 - ax) * G + ax * np.roll(G, sx, axis=1)
    return (1 - ay) * Gx + ay * np.roll(Gx, sy, axis=0)


class EPRY:
    def __init__(self, I_meas, k_solver, dk, M, No, kc, P0, kappa=0.0, axis_offset_um=(0.0, 0.0),
                 pupil_radius=1.02, replace_tukey=0.2, loss_tukey=0.2, band=(0.3, 3.5), threads=8,
                 dir_floor=0.1, init_noise=0.0, seed=0, mask_vignetted=False, pupil_gamma=0.0, wmask=None, modes=None, mode_w=None):
        """wmask: optional (n, M, M) validity weights of dark-mode data (036 Phase C); images with
        |k_n| > pupil_radius kc are dark field: no direct-beam normalisation (I_Q := 1), contrast about the masked mean."""
        self.I = np.clip(np.asarray(I_meas, dtype=np.float64), 0, None)
        k = np.asarray(k_solver, dtype=np.float64)
        self.dark = np.hypot(k[:, 0], k[:, 1]) > pupil_radius * kc
        self.wmask = None if wmask is None else np.asarray(wmask, dtype=np.float64)
        # dark-field intensity factors: the unknown illumination level of a dark-field position is estimated once, at the
        # first visit of the image (least squares of the current model against the data), and then kept fixed.
        # (Re-estimating it every sweep has a degenerate fixed point g -> 0 for dark-field images.)
        self.dark_gain_first_visit = bool(wmask is not None)
        self.visited = np.zeros(len(k), bool)
        s = np.round(-k / dk).astype(int)
        c, h = No // 2, M // 2
        self.y0, self.x0 = c - h + s[:, 0], c - h + s[:, 1]
        assert self.y0.min() >= 0 and self.y0.max() + M <= No and self.x0.min() >= 0 and self.x0.max() + M <= No
        self.frac = (k + s * dk) / dk                           # sub-pixel remainder e_n in pixels
        # partial coherence (036): illumination modes k_n + dk_nj with weights w_nj (mixed state).  Each mode's exit wave
        # is scaled by the common factor sqrt(T_n / I_n) (I_n = sum_j w_nj |psi_nj|^2), and the object spectrum / pupil are
        # updated from every mode with weight w_nj (mixed-state ptychography, Thibault & Menzel 2013).  Bright images:
        # T = g I_dir I_meas with the mixed-state direct beam I_dir; dark images (sample-out image subtracted in the data):
        # T = alpha I_meas + beta + I_dir, with gain alpha and offset beta estimated once at the first visit.
        self.pc = modes is not None
        if self.pc:
            md = np.asarray(modes, float); wj = np.asarray(mode_w, float); wj = wj / wj.sum(1, keepdims=True)
            kk = k[:, None, :] + md; sm = np.round(-kk / dk).astype(int)
            self.m_y0, self.m_x0 = No // 2 - M // 2 + sm[..., 0], No // 2 - M // 2 + sm[..., 1]
            self.m_frac = (kk + sm * dk) / dk; self.m_w = wj
            self.m_js = [np.where(wj[i] > 0)[0] for i in range(len(k))]
            act = wj > 0
            assert self.m_y0[act].min() >= 0 and self.m_y0[act].max() + M <= No and self.m_x0[act].min() >= 0 and self.m_x0[act].max() + M <= No
            self.dark_ab = np.tile([1.0, 0.0], (len(k), 1))
        self.M, self.No, self.dk, self.n, self.kc, self.w = M, No, dk, len(s), kc, threads
        self.sc = M / No
        yy, xx = np.mgrid[-h:M - h, -h:M - h] * dk
        self.support = (np.hypot(yy, xx) <= pupil_radius * kc)
        self.P = (np.asarray(P0, dtype=np.complex128) * self.support)
        dxo = 1.0 / (No * dk); y = (np.arange(No) - No / 2) * dxo
        Y, X = np.meshgrid(y + axis_offset_um[0], y + axis_offset_um[1], indexing="ij")
        self.Q = np.exp(1j * np.pi * kappa * (X ** 2 + Y ** 2)) if kappa != 0 else np.ones((No, No), complex)
        self.kappa, self.dxo, self.dir_floor, self.pupil_gamma = kappa, dxo, dir_floor, float(pupil_gamma)
        self.Qh = self._fft_c(self.Q)
        if init_noise > 0:                                       # optional random initial phase (independence tests)
            rng = np.random.default_rng(seed)
            self.Oh = self._fft_c(self.Q * np.exp(1j * init_noise * rng.standard_normal((No, No))))
        else:
            self.Oh = self.Qh.copy()                             # flat object O = 1 (effective object O.Q)
        self.g = np.ones(self.n)
        self.wrep = np.outer(tukey(M, replace_tukey), tukey(M, replace_tukey)) if replace_tukey else np.ones((M, M))
        self.wl = np.outer(tukey(M, loss_tukey), tukey(M, loss_tukey))
        q = np.fft.fftfreq(M, d=1.0 / (M * dk)); QR = np.hypot(*np.meshgrid(q, q, indexing="ij"))
        self.bsel = (QR >= band[0]) & (QR <= band[1])
        self.update_direct()
        if mask_vignetted and kappa != 0:
            # pixels where the model direct beam leaves the pupil (I_Q < floor) are neither enforced nor scored
            # (same rule as dipfpm.train.vignetting_weights; mask fixed from the initial pupil)
            from scipy.ndimage import gaussian_filter
            m = (self.Id_raw >= dir_floor * self.Id_raw[~self.dark].max()).astype(float) if (~self.dark).any() \
                else np.ones_like(self.Id_raw)
            m[self.dark] = 1.0
            m = np.stack([np.clip(gaussian_filter(mi, 3.0), 0, 1) * mi for mi in m])
            self.wrep = self.wrep[None] * m; self.wl = self.wl[None] * m
            self.masked_fraction = float(1 - (m > 0.5).mean())
        else:
            self.wrep = np.broadcast_to(self.wrep, (self.n, M, M)); self.wl = np.broadcast_to(self.wl, (self.n, M, M))
            self.masked_fraction = 0.0
        if self.wmask is not None:
            self.wrep = self.wrep * self.wmask; self.wl = self.wl * self.wmask
            if self.pc:
                self.Dm = sfft.fft2((self.I / self._wmean_w(self.I) - 1.0) * self.wl, workers=threads)[:, self.bsel]
            else:
                self.Dm = sfft.fft2((self.I / self._wmean(self.I) - 1.0) * self.wl, workers=threads)[:, self.bsel]
        else:
            self.Dm = sfft.fft2((self.I - 1.0) * self.wl, workers=threads)[:, self.bsel]

    def _wmean_w(self, I, idx=None):
        m = self.wl if idx is None else self.wl[list(idx)]
        return (I * m).sum((1, 2), keepdims=True) / m.sum((1, 2), keepdims=True)

    # -- partial coherence helpers --------------------------------------------------------------------
    def _pc_fields(self, n, spec=None, P=None):
        spec = self.Oh if spec is None else spec; P = self.P if P is None else P
        out = []
        for j in self.m_js[n]:
            Pj = shift_bilinear(P, *self.m_frac[n, j])
            Sj = spec[self.m_y0[n, j]:self.m_y0[n, j] + self.M, self.m_x0[n, j]:self.m_x0[n, j] + self.M]
            Phi = Pj * Sj
            out.append((j, self._ifft_c(Phi) * self.sc, Phi, Pj, Sj))
        return out

    def _pc_I(self, fl, n):
        return sum(self.m_w[n, j] * np.abs(psi) ** 2 for j, psi, _, _, _ in fl)

    def _wmean(self, I, idx=None):
        if self.wmask is None:
            return I.mean((1, 2), keepdims=True)
        m = self.wmask if idx is None else self.wmask[list(idx)]
        return (I * m).sum((1, 2), keepdims=True) / m.sum((1, 2), keepdims=True)

    # -- FFT helpers (centred spectra, ortho) ------------------------------------------------------
    def _fft_c(self, x):
        return sfft.fftshift(sfft.fft2(x, norm="ortho", workers=self.w), axes=(-2, -1))

    def _ifft_c(self, X):
        return sfft.ifft2(sfft.ifftshift(X, axes=(-2, -1)), norm="ortho", workers=self.w)

    def patch(self, spec, n):
        return spec[self.y0[n]:self.y0[n] + self.M, self.x0[n]:self.x0[n] + self.M]

    def Pn(self, n, P=None):
        return shift_bilinear(self.P if P is None else P, *self.frac[n])

    def field(self, n, P=None, spec=None):
        Phi = self.Pn(n, P) * self.patch(self.Oh if spec is None else spec, n)
        return self._ifft_c(Phi) * self.sc, Phi

    def update_direct(self):
        """Model image of Q alone (flat-field reference) for every image, with the current pupil."""
        if self.pc:
            Id = np.stack([self._pc_I(self._pc_fields(n, spec=self.Qh), n) for n in range(self.n)])
        else:
            Id = np.stack([np.abs(self._ifft_c(self.Pn(n) * self.patch(self.Qh, n)) * self.sc) ** 2 for n in range(self.n)])
        self.Id_raw = Id
        if self.dark.all():
            self.Id = np.ones_like(Id)
        else:
            self.Id = np.maximum(Id, self.dir_floor * Id[~self.dark].max())
            self.Id[self.dark] = 1.0       # dark field: intensity relative to the unvignetted bright-field level

    def model_intensity(self, idx=None):
        idx = range(self.n) if idx is None else idx
        if self.pc:
            return np.stack([self._pc_I(self._pc_fields(n), n) - self.Id_raw[n] if self.dark[n]
                             else self._pc_I(self._pc_fields(n), n) / self.Id[n] for n in idx])
        return np.stack([np.abs(self.field(n)[0]) ** 2 / self.Id[n] for n in idx])

    def band_per_image(self, idx=None):
        idx = list(range(self.n)) if idx is None else list(idx)
        I = self.model_intensity(idx)
        D = self.Dm[idx]
        if self.pc:           # window-weighted means; dark images: profiled contrast factor (additive background)
            mu = self._wmean_w(I, idx); dk_ = self.dark[idx][:, None, None]
            F = sfft.fft2(np.where(dk_, I - mu, I / mu - 1.0) * self.wl[idx], workers=self.w)[:, self.bsel]
            cg = np.clip((np.conj(F) * D).real.sum(1) / ((np.abs(F) ** 2).sum(1) + 1e-30), 0, None)
            F = F * np.where(self.dark[idx], cg, 1.0)[:, None]
            return (np.abs(F - D) ** 2).sum(1) / (np.abs(D) ** 2).sum(1)
        F = sfft.fft2((I / self._wmean(I, idx) - 1.0) * self.wl[idx], workers=self.w)[:, self.bsel]
        return (np.abs(F - D) ** 2).sum(1) / (np.abs(D) ** 2).sum(1)

    def update_gain(self, idx):
        """Per-image intensity factor g_n (|psi|^2 ~ g_n I_Q I_meas), normalised to mean 1 over idx."""
        for n in idx:
            if self.pc:
                if self.dark[n]:
                    if not self.visited[n]: self._pc_dark_ab(n)
                    continue
                Im = self._pc_I(self._pc_fields(n), n) / self.Id[n]
                self.g[n] = (self.wl[n] * Im).sum() / (self.wl[n] * self.I[n]).sum()
                continue
            if self.dark_gain_first_visit and self.dark[n]:
                if not self.visited[n]:             # held-out dark image: estimate once from the current model
                    self.g[n] = (self.wl[n] * np.abs(self.field(n)[0]) ** 2).sum() / max((self.wl[n] * self.I[n]).sum(), 1e-30)
                    self.visited[n] = True
                continue
            Im = np.abs(self.field(n)[0]) ** 2 / self.Id[n]
            self.g[n] = (self.wl[n] * Im).sum() / (self.wl[n] * self.I[n]).sum()
        br = [n for n in idx if not self.dark[n]]
        # scale fixed by the bright-field images (dark-field factors relative to them)
        self.g /= self.g[br].mean() if br else self.g[list(idx)].mean()

    def _pc_dark_ab(self, n, fl=None):
        """Gain alpha and offset beta of a dark image: model (I - I_dir) ~ alpha * data + beta (window-weighted LS)."""
        fl = self._pc_fields(n) if fl is None else fl
        m = self._pc_I(fl, n) - self.Id_raw[n]; d = self.I[n]; w = self.wl[n]
        s0, s1, s2 = w.sum(), (w * d).sum(), (w * d * d).sum(); t0, t1 = (w * m).sum(), (w * d * m).sum()
        a = max((s0 * t1 - s1 * t0) / max(s0 * s2 - s1 * s1, 1e-30), 1e-6); b = (t0 - a * s1) / s0
        self.dark_ab[n] = (a, b); self.visited[n] = True

    def _sweep_pc(self, order, alpha, beta, update_pupil=True):
        num = den = 0.0
        for n in order:
            fl = self._pc_fields(n)
            I = self._pc_I(fl, n)
            if self.dark[n]:
                if not self.visited[n]: self._pc_dark_ab(n, fl)
                a_, b_ = self.dark_ab[n]
                T = np.maximum(a_ * self.I[n] + b_ + self.Id_raw[n], 0.0)
            else:
                T = self.g[n] * self.Id[n] * self.I[n]
            num += (self.wl[n] * (np.sqrt(I) - np.sqrt(T)) ** 2).sum(); den += (self.wl[n] * T).sum()
            r = self.wrep[n] * (np.sqrt(T / (I + 1e-30)) - 1.0)
            for j, psi, Phi, Pj, Sj in fl:
                dPhi = self._fft_c(psi * r) / self.sc               # psi' - psi = wrep (r psi - psi)
                wj = self.m_w[n, j]; S_old = Sj.copy()
                Sj += alpha * wj * np.conj(Pj) / (np.abs(Pj) ** 2).max() * dPhi   # in place (view of self.Oh)
                if update_pupil:
                    dP = beta * wj * np.conj(S_old) / max((np.abs(S_old) ** 2).max(), 1e-30) * dPhi
                    self.P += shift_bilinear_adj(dP, *self.m_frac[n, j])
            if update_pupil: self.P *= self.support
        return num / den

    # -- one EPRY sweep ---------------------------------------------------------------------------
    def sweep(self, order, alpha, beta, update_pupil=True):
        if self.pc:
            return self._sweep_pc(order, alpha, beta, update_pupil)
        num = den = 0.0
        for n in order:
            psi, Phi = self.field(n)
            if self.dark_gain_first_visit and self.dark[n] and not self.visited[n]:
                self.g[n] = (self.wl[n] * np.abs(psi) ** 2 / self.Id[n]).sum() / max((self.wl[n] * self.I[n]).sum(), 1e-30)
                self.visited[n] = True
            A = np.sqrt(self.g[n] * self.Id[n] * self.I[n])
            amp = np.abs(psi)
            num += (self.wl[n] * (amp - A) ** 2).sum(); den += (self.wl[n] * A ** 2).sum()
            psi_new = psi + self.wrep[n] * (A * psi / (amp + 1e-12) - psi)
            dPhi = self._fft_c(psi_new) / self.sc - Phi
            Pn = self.Pn(n); S = self.patch(self.Oh, n)
            S_old = S.copy()
            S += alpha * np.conj(Pn) / (np.abs(Pn) ** 2).max() * dPhi          # in-place into self.Oh
            if update_pupil:
                if self.pupil_gamma > 0:      # rPIE-type normalisation (Maiden, Johnson & Li 2017)
                    a2 = np.abs(S_old) ** 2
                    dP = beta * np.conj(S_old) * dPhi / ((1 - self.pupil_gamma) * a2 + self.pupil_gamma * a2.max())
                else:                         # EPRY (Ou et al. 2014): conj(S)/max|S|^2
                    dP = beta * np.conj(S_old) / (np.abs(S_old) ** 2).max() * dPhi
                self.P += shift_bilinear_adj(dP, *self.frac[n])
                self.P *= self.support
        return num / den

    def state(self):
        O = self._ifft_c(self.Oh) / self.Q
        return dict(a=-np.log(np.maximum(np.abs(O), 1e-6)), phi=np.angle(O), P=self.P.copy(), g=self.g.copy())


def run_epry(eng, train_pos, val_pos, cfg_e, kabs, log=print):
    """Sweeps with adaptive step size; returns dict(best, final, hist) with the dipfpm selection rule.
    kabs: |k_n| of the images (sets the centre-out update order)."""
    tp = set(int(v) for v in train_pos)
    order = [int(n) for n in np.argsort(kabs) if int(n) in tp]              # centre-out (increasing |k|)
    train_pos, val_pos = list(train_pos), list(val_pos)
    alpha, beta = float(cfg_e["alpha"]), float(cfg_e["beta"])
    hist = dict(it=[], err=[], train_band=[], val_band=[], alpha=[], t=[])
    snaps, best_min, tol = [], np.inf, float(cfg_e.get("select_tol", 0.01))
    t0 = time.time(); e_prev = None; n_done = 0
    for it in range(int(cfg_e["n_sweeps"])):
        up = cfg_e["pupil_update"] and it >= int(cfg_e["pupil_update_from"])
        err = eng.sweep(order, alpha, beta, update_pupil=up)
        eng.update_direct()
        if cfg_e["intensity_correction"]:
            eng.update_gain(train_pos)
            if len(val_pos): eng.update_gain(list(train_pos) + list(val_pos))
        bp = eng.band_per_image()
        trb = float(bp[train_pos].mean()); vab = float(bp[val_pos].mean()) if len(val_pos) else trb
        for k_, v_ in dict(it=it, err=err, train_band=trb, val_band=vab, alpha=alpha, t=time.time() - t0).items():
            hist[k_].append(v_)
        score = vab if cfg_e.get("select_by", "val_band") == "val_band" else trb
        best_min = min(best_min, score)
        snaps.append(dict(it=it, score=score, val_band=vab, train_band=trb, **eng.state()))
        snaps = [s for s in snaps if s["score"] <= best_min * (1 + tol)]    # candidates only
        if it % 5 == 0 or it == int(cfg_e["n_sweeps"]) - 1:
            log(f"  sweep {it:4d}  amplitude error {err:.5f} | band misfit train {trb:.4f} val {vab:.4f} | "
                f"step {alpha:.3g} | best@{snaps[0]['it']}  ({time.time() - t0:.0f} s)")
        if cfg_e.get("adaptive_step", True) and e_prev is not None and (e_prev - err) / e_prev < float(cfg_e["step_eps"]):
            alpha *= 0.5; beta *= 0.5
        e_prev = err; n_done = it + 1
        if alpha < float(cfg_e["alpha_min"]):
            log(f"  step size below {cfg_e['alpha_min']} -> stop after sweep {it}"); break
    best = snaps[0]
    final = dict(it=n_done - 1, **eng.state())
    return dict(best=best, final=final, hist={k: np.array(v) for k, v in hist.items()}, n_done=n_done)
