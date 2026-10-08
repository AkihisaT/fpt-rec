"""Differentiable FPM forward model (slide p.3):
    I_pred,n(x - dx_n) = | FT^-1[ P(u,v) . FT[ O(x,y) exp(i k_n . r) ] ] |^2 ,   loss uses c_n I_pred,n.
Implementation (solver frame of fptrecon): object spectrum on an N_o grid; image n is computed on an
M x M spectral patch cut at the integer offset nearest to -k_n (alias-free for |q| <= M dk / 2).
The sub-pixel remainder of k_n and the learned illumination correction dk_n are applied exactly by
evaluating the (continuous) pupil at q + e_n, e_n = k_n + dk_n - k_n(rounded), since
|F^-1{P(q) O^(q - k)}|^2 = |F^-1{P(q + e) O^(q - k + e)}|^2.
P = soft-edged NA disc * exp(i sum_j z_j Z_j) (Zernike, Noll-normalised, Cartesian polynomials);
an optional lateral image shift dx_n is a phase ramp on the field spectrum;
optional FOV quadratic phase Q = exp(i pi kappa |x - x_axis|^2), with direct-beam normalisation."""
import numpy as np
import torch

from math import factorial

ZERN = [(2, 0, "defocus"), (2, 2, "astig 0deg"), (2, -2, "astig 45deg"), (3, 1, "coma x"), (3, -1, "coma y"),
        (3, 3, "trefoil x"), (3, -3, "trefoil y"), (4, 0, "spherical"), (4, 2, "2nd astig 0deg"),
        (4, -2, "2nd astig 45deg"), (4, 4, "quadrafoil 0deg"), (4, -4, "quadrafoil 45deg")]


def zernike_nm(n, m, rho, th):
    R = np.zeros_like(rho)
    for s in range((n - abs(m)) // 2 + 1):
        R += ((-1) ** s * factorial(n - s) /
              (factorial(s) * factorial((n + abs(m)) // 2 - s) * factorial((n - abs(m)) // 2 - s)) * rho ** (n - 2 * s))
    norm = np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)
    return norm * R * (np.cos(m * th) if m >= 0 else np.sin(-m * th))


def fourier_resample(x, Nout):
    """Real (…, N, N) -> (…, Nout, Nout) by zero-padding / cropping the centred spectrum (values preserved)."""
    N = x.shape[-1]
    F = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    if Nout >= N:
        c, h = Nout // 2, N // 2
        Fo = torch.zeros(*x.shape[:-2], Nout, Nout, dtype=F.dtype)
        Fo[..., c - h:c - h + N, c - h:c - h + N] = F
    else:
        c, h = N // 2, Nout // 2
        Fo = F[..., c - h:c - h + Nout, c - h:c - h + Nout]
    return torch.fft.ifft2(torch.fft.ifftshift(Fo, dim=(-2, -1))).real * (Nout / N) ** 2


def _zernike_cartesian_terms():
    """For each (n, m): list of (coef, power of rho^2) for R_n^|m|(rho)/rho^|m| and the Noll norm."""
    out = []
    for n, m, _ in ZERN:
        am = abs(m); terms = []
        for s in range((n - am) // 2 + 1):
            c = (-1) ** s * factorial(n - s) / (factorial(s) * factorial((n + am) // 2 - s) * factorial((n - am) // 2 - s))
            terms.append((c, (n - 2 * s - am) // 2))
        out.append((terms, am, m, np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)))
    return out


_ZT = _zernike_cartesian_terms()


def zernike_torch(yn, xn):
    """Zernike basis at normalised pupil coordinates (tensors of any shape) -> (..., len(ZERN)).
    Smooth polynomial form (no atan2), valid also slightly outside the unit disc."""
    r2 = yn * yn + xn * xn
    zc = torch.complex(xn, yn)
    cols = []
    for terms, am, m, norm in _ZT:
        R = sum(c * r2 ** p for c, p in terms)
        ang = torch.ones_like(r2) if am == 0 else (zc ** am).real if m > 0 else (zc ** am).imag
        cols.append(norm * R * ang)
    return torch.stack(cols, -1)


class FPMPhysics(torch.nn.Module):
    def __init__(self, k_solver, dk, M, No, kc, lam, kappa=0.0, axis_offset_um=(0.0, 0.0), pupil_radius=1.02,
                 direct_norm=True, dir_floor=0.1, edge_px=0.25, modes=None, mode_w=None):
        """k_solver: (n, 2) illumination wave-vectors (row, col), um^-1, solver frame.
        direct_norm: divide by the image of Q alone (the flat-field reference already contains the
        FOV vignetting of the illumination), I -> I / max(I_Q, dir_floor * max I_Q). Only active if kappa != 0.
        edge_px: width (in dk) of the logistic pupil edge (keeps the edge differentiable w.r.t. k).
        modes / mode_w: partial coherence (036): (n, J, 2) illumination-mode offsets (um^-1, added to k_n) and (n, J)
        weights; the image is the weighted sum of the mode intensities, bright images are divided by the mixed-state
        direct beam and dark images are (I - I_direct) (data with the sample-out image subtracted).  None = coherent."""
        super().__init__()
        self.use_dn, self.dir_floor = bool(direct_norm and kappa != 0), dir_floor
        k_solver = np.asarray(k_solver, dtype=np.float64)
        s = np.round(-k_solver / dk).astype(int)
        c, h = No // 2, M // 2
        self.y0 = c - h + s[:, 0]; self.x0 = c - h + s[:, 1]
        assert self.y0.min() >= 0 and self.y0.max() + M <= No and self.x0.min() >= 0 and self.x0.max() + M <= No, \
            "object grid too small for the illumination shifts"
        self.M, self.No, self.dk, self.n, self.kc = M, No, dk, len(s), kc
        self.R, self.edge = pupil_radius * kc, edge_px * dk
        # dark-field images (illumination outside the pupil, |k_n| > pupil_radius kc): no direct-beam normalisation;
        # their intensities are relative to the unvignetted bright-field level (|psi|^2 = 1 for O = 1 inside the pupil)
        self.dark = np.hypot(k_solver[:, 0], k_solver[:, 1]) > self.R
        self.register_buffer("kres", torch.tensor(k_solver + s * dk, dtype=torch.float32))   # k - k_rounded
        self.pc = modes is not None
        if self.pc:
            md = np.asarray(modes, np.float64); wj = np.asarray(mode_w, np.float64)
            wj = wj / wj.sum(1, keepdims=True)
            kk = k_solver[:, None, :] + md                                       # (n, J, 2) mode wave-vectors
            sm = np.round(-kk / dk).astype(int)
            self.pairs = [np.where(wj[i] > 0)[0] for i in range(len(s))]
            self.m_y0 = c - h + sm[..., 0]; self.m_x0 = c - h + sm[..., 1]
            act = wj > 0
            assert self.m_y0[act].min() >= 0 and self.m_y0[act].max() + M <= No and self.m_x0[act].min() >= 0 and self.m_x0[act].max() + M <= No, \
                "object grid too small for the illumination modes"
            self.m_res = torch.tensor(kk + sm * dk, dtype=torch.float32)            # sub-pixel remainders per mode
            self.m_w = torch.tensor(wj, dtype=torch.float32)
            self.Id_cache = None; self.cache_on = False
        yy, xx = np.mgrid[-(M // 2):M - M // 2, -(M // 2):M - M // 2] * dk
        self.qc = (yy, xx)                                                   # centred grids (numpy)
        self.register_buffer("ky", torch.tensor(np.fft.ifftshift(yy).astype(np.float32)))
        self.register_buffer("kx", torch.tensor(np.fft.ifftshift(xx).astype(np.float32)))
        # Zernike basis and its k-gradient (per um^-1) on the grid: W(q + e) = W(q) + e . grad W(q) + O(e^2)
        # (measured: max error 3.3e-3 rad at |e| = 0.14 um^-1 for the aberrations here; the error scales as |e|^2,
        #  i.e. ~3e-4 rad for the sub-pixel remainder |e| <= dk/sqrt(2) = 0.043 um^-1 used in image-shift mode)
        with torch.no_grad():
            Y = torch.tensor(np.fft.ifftshift(yy) / kc, dtype=torch.float64); X = torch.tensor(np.fft.ifftshift(xx) / kc, dtype=torch.float64)
            hh = 1e-5
            Z0 = zernike_torch(Y, X)
            Zy = (zernike_torch(Y + hh, X) - zernike_torch(Y - hh, X)) / (2 * hh * kc)
            Zx = (zernike_torch(Y, X + hh) - zernike_torch(Y, X - hh)) / (2 * hh * kc)
        self.register_buffer("Z", Z0.permute(2, 0, 1).float().contiguous())
        self.register_buffer("Zy", Zy.permute(2, 0, 1).float().contiguous())
        self.register_buffer("Zx", Zx.permute(2, 0, 1).float().contiguous())
        dxo = 1.0 / (No * dk)
        y = (np.arange(No) - No / 2) * dxo
        Y, X = np.meshgrid(y + axis_offset_um[0], y + axis_offset_um[1], indexing="ij")
        Q = np.exp(1j * np.pi * kappa * (X ** 2 + Y ** 2)) if kappa != 0 else np.ones((No, No))
        self.register_buffer("Q", torch.tensor(Q, dtype=torch.complex64))
        self.register_buffer("Qh", torch.fft.fftshift(torch.fft.fft2(torch.tensor(Q, dtype=torch.complex64), norm="ortho")))
        self.dxo = dxo

    def pupil(self, zc, e=None, W_free=None):
        """Pupil(s) on the ifftshifted M-grid evaluated at q + e. e: None or (m, 2) um^-1 -> (m, M, M).
        W_free: optional free pupil-phase map (ifftshifted (M, M)) added to the Zernike expansion."""
        W0 = torch.einsum("j,jyx->yx", zc, self.Z)
        if W_free is not None:
            W0 = W0 + W_free
        if e is None:
            amp = torch.sigmoid((self.R - torch.sqrt(self.ky ** 2 + self.kx ** 2)) / self.edge)
            return amp * torch.exp(1j * W0), W0
        Wy = torch.einsum("j,jyx->yx", zc, self.Zy); Wx = torch.einsum("j,jyx->yx", zc, self.Zx)
        if W_free is not None:     # central differences (the ifftshifted grid is periodic; edges lie outside the pupil)
            Wy = Wy + (torch.roll(W_free, -1, 0) - torch.roll(W_free, 1, 0)) / (2 * self.dk)
            Wx = Wx + (torch.roll(W_free, -1, 1) - torch.roll(W_free, 1, 1)) / (2 * self.dk)
        ey, ex = e[:, 0, None, None], e[:, 1, None, None]
        W = W0[None] + ey * Wy[None] + ex * Wx[None]
        qy, qx = self.ky[None] + ey, self.kx[None] + ex
        amp = torch.sigmoid((self.R - torch.sqrt(qy * qy + qx * qx + 1e-12)) / self.edge)
        return amp * torch.polar(torch.ones_like(W), W), W

    def pupil_from_map(self, W_map, e):
        """Pupil map (centred numpy (M, M), for evaluation of other methods) at q + e (numpy (m, 2)).
        Real map: pupil phase (amplitude = soft NA disc). Complex map: full complex pupil (e.g. EPRY),
        bilinearly interpolated (real and imaginary parts), zero outside the grid."""
        from scipy.ndimage import map_coordinates
        from scipy.special import expit
        yy, xx = self.qc; h = self.M // 2
        out = []
        for ey, ex in e:
            qy, qx = yy + ey, xx + ex
            crd = [qy / self.dk + h, qx / self.dk + h]
            if np.iscomplexobj(W_map):
                P = (map_coordinates(W_map.real, crd, order=1, mode="constant")
                     + 1j * map_coordinates(W_map.imag, crd, order=1, mode="constant"))
                out.append(np.fft.ifftshift(P)); continue
            W = map_coordinates(W_map, crd, order=1, mode="nearest")
            amp = expit((self.R - np.hypot(qy, qx)) / self.edge)
            out.append(np.fft.ifftshift(amp * np.exp(1j * W)))
        return torch.tensor(np.array(out), dtype=torch.complex64)

    def forward(self, a, phi, zc, shifts_um, idx=None, W_map=None, dk_illum=None, W_free=None):
        """a, phi: (No, No) real maps (O = exp(-a + i phi)). zc: Zernike coefficients (rad).
        shifts_um: (n, 2) lateral image shifts (row, col), um.  dk_illum: None or (n, 2) illumination
        wave-vector corrections (row, col), um^-1.  W_map: optional centred numpy (M, M) pupil-phase map
        replacing the Zernike pupil (evaluation only).  Returns intensities (len(idx), M, M)."""
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        if self.pc:
            return self._forward_pc(a, phi, zc, shifts_um, idx, W_map, dk_illum, W_free)
        e = self.kres[idx] if dk_illum is None else self.kres[idx] + dk_illum[idx]
        P = self.pupil(zc, e, W_free)[0] if W_map is None else self.pupil_from_map(W_map, e.detach().numpy())
        O = torch.exp(torch.complex(-a, phi)) * self.Q
        sh = shifts_um[idx]
        ramp = torch.exp(-2j * np.pi * (self.ky[None] * sh[:, 0, None, None] + self.kx[None] * sh[:, 1, None, None]))
        I = self._image(torch.fft.fftshift(torch.fft.fft2(O, norm="ortho")), P, idx, ramp)
        if self.use_dn:
            br = np.where(~self.dark[idx])[0]
            if len(br) == len(idx):
                Id = self._image(self.Qh, P, idx, None)
                I = I / torch.clamp(Id, min=self.dir_floor * float(Id.detach().max()))
            elif len(br):
                Pb = P[br] if P.shape[0] == len(idx) else P
                Id = self._image(self.Qh, Pb, idx[br], None)
                den = torch.ones_like(I)
                den[br] = torch.clamp(Id, min=self.dir_floor * float(Id.detach().max()))
                I = I / den
        return I

    # -- partial coherence (mixed state) ------------------------------------------------------------
    def _pc_sum(self, spec, zc, idx, dk_illum, W_free, W_map, shifts_um=None, chunk=192):
        """sum_j w_nj |psi_nj|^2 for the images idx (modes gathered in chunks)."""
        img = np.concatenate([np.full(len(self.pairs[i]), k) for k, i in enumerate(idx)])
        ii = np.concatenate([np.full(len(self.pairs[i]), i) for i in idx]); jj = np.concatenate([self.pairs[i] for i in idx])
        ar = np.arange(self.M)
        out = torch.zeros((len(idx), self.M, self.M))
        for g0 in range(0, len(ii), chunk):
            sl = slice(g0, g0 + chunk); i_, j_ = ii[sl], jj[sl]
            e = self.m_res[i_, j_] if dk_illum is None else self.m_res[i_, j_] + dk_illum[i_]
            P = self.pupil(zc, e, W_free)[0] if W_map is None else self.pupil_from_map(W_map, e.detach().numpy())
            iy = torch.tensor(self.m_y0[i_, j_][:, None] + ar[None]); ix = torch.tensor(self.m_x0[i_, j_][:, None] + ar[None])
            pt = torch.fft.ifftshift(spec[iy[:, :, None], ix[:, None, :]], dim=(-2, -1))
            F = P * pt
            if shifts_um is not None:
                sh = shifts_um[i_]
                F = F * torch.exp(-2j * np.pi * (self.ky[None] * sh[:, 0, None, None] + self.kx[None] * sh[:, 1, None, None]))
            psi = torch.fft.ifft2(F, norm="ortho") * (self.M / self.No)
            out = out.index_add(0, torch.tensor(img[sl]), self.m_w[i_, j_][:, None, None] * (psi.real ** 2 + psi.imag ** 2))
        return out

    def refresh_direct(self, zc, dk_illum=None, W_free=None):
        """Cache the mixed-state direct beam of all images (no gradient); used while cache_on is True."""
        with torch.no_grad():
            self.Id_cache = self._pc_sum(self.Qh, zc, np.arange(self.n), dk_illum, W_free, None)

    def direct(self, zc, idx=None, dk_illum=None, W_free=None, W_map=None):
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        return self._pc_sum(self.Qh, zc, idx, dk_illum, W_free, W_map)

    def _forward_pc(self, a, phi, zc, shifts_um, idx, W_map, dk_illum, W_free):
        O = torch.exp(torch.complex(-a, phi)) * self.Q
        I = self._pc_sum(torch.fft.fftshift(torch.fft.fft2(O, norm="ortho")), zc, idx, dk_illum, W_free, W_map, shifts_um)
        if self.cache_on and self.Id_cache is not None and W_map is None:
            Id = self.Id_cache[idx]
        else:
            Id = self._pc_sum(self.Qh, zc, idx, dk_illum, W_free, W_map)
        dk_ = torch.tensor(self.dark[idx])
        br = ~self.dark[idx]
        fl = self.dir_floor * float(Id.detach()[torch.tensor(br)].max()) if br.any() else 1.0
        return torch.where(dk_[:, None, None], I - Id, I / torch.clamp(Id, min=fl))

    def _image(self, spec, P, idx, ramp):
        patches = torch.stack([spec[self.y0[i]:self.y0[i] + self.M, self.x0[i]:self.x0[i] + self.M] for i in idx])
        patches = torch.fft.ifftshift(patches, dim=(-2, -1))
        F = P * patches if ramp is None else P * patches * ramp
        psi = torch.fft.ifft2(F, norm="ortho") * (self.M / self.No)
        return psi.real ** 2 + psi.imag ** 2
