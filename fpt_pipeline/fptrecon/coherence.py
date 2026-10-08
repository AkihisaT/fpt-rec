"""Partial spatial coherence (illumination angular spread) as a mixed state of illumination modes.

The illumination of position n is an incoherent sum of plane waves k_n + dk_j with weights w_j (sum w_j = 1):
    I_n(x) = sum_j w_j | F^-1{ P(k) Ohat_eff(k - k_n - dk_j) } |^2
(the same object, pupil and FOV curvature for every mode).  dk_j / w_j sample a radially symmetric angular distribution S
given as fractions per annulus; this module builds the mode list on a polar grid.
"""
import numpy as np


def modes_from_radial(rho_edges, frac, n_az, rot=0.0):
    """rho_edges: (m+1,) annulus edges (any unit, e.g. um^-1); frac: (m,) fraction of S in each annulus (sum 1);
    n_az: (m,) azimuth count per annulus (1 for a central disc).  Returns dk (J, 2) (y, x) and w (J,).
    Each annulus is represented by n_az modes at its S-weighted mean radius (approximated by the mid radius),
    azimuths offset by half a step on alternate rings (plus rot)."""
    dk, w = [], []
    for i, (f, na) in enumerate(zip(frac, n_az)):
        if f <= 0:
            continue
        r0, r1 = rho_edges[i], rho_edges[i + 1]
        if na == 1 and r0 == 0:
            dk.append((0.0, 0.0)); w.append(f); continue
        rm = np.sqrt(0.5 * (r0 ** 2 + r1 ** 2))                       # radius splitting the annulus area in half
        th = rot + (np.arange(na) + 0.5 * (i % 2)) * 2 * np.pi / na
        for t in th:
            dk.append((rm * np.sin(t), rm * np.cos(t))); w.append(f / na)
    dk, w = np.array(dk), np.array(w)
    return dk, w / w.sum()


def annulus_fractions(rho, S_ann, rho_edges):
    """Re-bin an annulus distribution given on radii rho (fractions S_ann, e.g. from the NNLS decomposition) into the
    annuli rho_edges (by cumulative interpolation)."""
    cum = np.concatenate([[0.0], np.cumsum(S_ann)])
    rr = np.concatenate([[0.0], rho])
    c = np.interp(rho_edges, rr, cum[:len(rr)] if len(cum) > len(rr) else cum, right=cum[-1])
    f = np.diff(c)
    return np.clip(f, 0, None) / max(f.sum(), 1e-12)


def cauchy_core_fractions(rho_edges, eta, sigma, ell, beta, n=20000, rmax=None):
    """Annulus fractions of S = (1-eta) Gauss(sigma) + eta * (1 + rho^2/ell^2)^-beta (2-D, unit integral)."""
    rmax = rmax or rho_edges[-1]
    r = np.linspace(0, rmax, n + 1)[1:]; dr = r[1] - r[0]
    g = np.exp(-0.5 * (r / sigma) ** 2) * r; g /= g.sum()
    h = (1 + (r / ell) ** 2) ** (-beta) * r; h /= h.sum()
    s = (1 - eta) * g + eta * h
    cum = np.concatenate([[0.0], np.cumsum(s)])
    c = np.interp(rho_edges, np.concatenate([[0.0], r]), cum)
    f = np.diff(c)
    return f / f.sum()


def adaptive_modes(kn, dk_ref, w_ref, kc, r_p=1.02, core=0.13, rad_edges=(0.13, 0.3, 0.6, 1.3), sec_in=8, sec_out=4):
    """Per-image reduction of a fine mode set (dk_ref, w_ref; um^-1) for illuminations kn (n, 2):
    modes with |dk| <= core*kc are kept individually; the others are grouped by radial bin (rad_edges, units of kc),
    by whether k_n + dk falls inside the pupil (|.| <= r_p kc: bright-field contribution) or outside (dark-field), and by
    azimuth sector of dk (sec_in sectors inside, sec_out outside); each group becomes one mode at its weighted centroid
    carrying the group weight.  Returns dk (n, J, 2) and w (n, J), padded with zero-weight modes at dk = 0."""
    rho = np.hypot(dk_ref[:, 0], dk_ref[:, 1]) / kc
    th = np.arctan2(dk_ref[:, 0], dk_ref[:, 1]) % (2 * np.pi)
    per_dk, per_w = [], []
    for k in np.asarray(kn):
        inside = np.hypot(k[0] + dk_ref[:, 0], k[1] + dk_ref[:, 1]) <= r_p * kc
        keep = rho <= core
        d_list = list(dk_ref[keep]); w_list = list(w_ref[keep])
        rb = np.digitize(rho, rad_edges)
        for flag, ns in ((True, sec_in), (False, sec_out)):
            sec = np.floor(th / (2 * np.pi / ns)).astype(int)
            for b in range(1, len(rad_edges)):
                for s_ in range(ns):
                    m = (~keep) & (inside == flag) & (rb == b) & (sec == s_)
                    if m.any() and w_ref[m].sum() > 0:
                        ww = w_ref[m]; d_list.append((dk_ref[m] * ww[:, None]).sum(0) / ww.sum()); w_list.append(ww.sum())
        per_dk.append(np.array(d_list)); per_w.append(np.array(w_list))
    J = max(len(w) for w in per_w)
    dk = np.zeros((len(per_w), J, 2)); w = np.zeros((len(per_w), J))
    for i, (d_, w_) in enumerate(zip(per_dk, per_w)):
        dk[i, :len(w_)] = d_; w[i, :len(w_)] = w_
    return dk, w / w.sum(1, keepdims=True)


PC_SETS = dict(F=dict(core=0.08, sec_in=8, sec_out=2), A=dict(core=0.13, sec_in=8, sec_out=4),
               E=dict(core=0.13, sec_in=16, sec_out=6, rad_edges=(0.13, 0.22, 0.35, 0.5, 0.7, 1.0, 1.3)))
FINE_EDGES = np.array([0, .03, .07, .12, .2, .3, .45, .65, .9, 1.2]); FINE_NAZ = [1, 6, 8, 12, 16, 20, 24, 28, 32]


def pc_modes(pc, kn, kc, r_p=1.02):
    """Mode offsets (n, J, 2) um^-1 and weights (n, J) for a partial-coherence spec (config dict or None):
    {"eta": halo fraction, "sigma": core (kc), "ell": halo scale (kc), "beta": halo exponent, "modes": "F"|"A"|"E"}
    or {"eta": "1mode"} (coherent through the mixed-state path).  Returns (None, None) for pc None."""
    if not pc:
        return None, None
    kn = np.asarray(kn, float)
    if pc.get("eta") == "1mode":
        return np.zeros((len(kn), 1, 2)), np.ones((len(kn), 1))
    frac = cauchy_core_fractions(FINE_EDGES, float(pc["eta"]), float(pc.get("sigma", 0.02)), float(pc.get("ell", 0.1)), float(pc.get("beta", 2.0)))
    dref, wref = modes_from_radial(FINE_EDGES * kc, frac, FINE_NAZ)
    return adaptive_modes(kn, dref, wref, kc, r_p=r_p, **PC_SETS[pc.get("modes", "F")])
