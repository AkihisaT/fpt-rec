"""Compare the 32a posaffine result of this computer with the reference of the analysis (tests/reference/).

usage (repository root): python tests/compare_ref.py [fpt_output_dir]   (default fpt_pipeline/fpt_output_32a_bf_aff2)
Same computer type / library versions: identical (max difference 0) -> '一致'; every relative difference below 1e-3
-> 'ほぼ一致' (as before; items present on one side only are not compared).
Other CPU or BLAS: the posaffine fit (fixed numbers of L-BFGS iterations, focus loop) amplifies floating-point
differences. On an M4 Mac (2026-10-07; identical results with 6, 3 or 2 threads), two copies of the input stack
changed by +-1 ulp (relative 1.2e-7, random signs, seeds 0 and 1) and the unchanged stack differed from each other by
up to 11.9 um in defocus, 7.9 um in astigmatism, 0.034 px in the image corrections and 6.0e-3 in misfit (relative);
the reference lies within this spread (up to 12.9 um, 11.2 um, 0.029 px, 4.3e-3), while the relative difference of
items near zero (shear, axis angles) reached 110. The items set by the fit (FIT, CAL_FIT) differed from the reference
by 2e-3 to 110 relative, all other items (shift measurement, settings, calibration inputs) by less than 2e-5.
'ほぼ一致' is therefore also given when
  - the quantities used downstream are within the tolerances below (defocus and astigmatism: focus_tol_um of the
    focus loop; image corrections: TOL_CORRECTION_PX; every misfit: TOL_MISFIT_REL),
  - every item outside FIT / CAL_FIT differs by less than 1e-3 relative and no true/false value differs,
  - no item of position_affine.json is missing (focus_loop may have another number of rounds) and none differs in
    shape, NaN or text."""
import json, os, sys
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
REF = (R + "tests/reference/32a_aff2_position_affine.json", R + "tests/reference/32a_aff2_calibration.json")
TOL_CORRECTION_PX = 0.1     # image corrections applied to the stack (1 ulp spread 0.034 px; 0.1 px = 3 nm, resolution ~5 px)
TOL_MISFIT_REL = 2e-2       # every misfit in position_affine.json (1 ulp spread 6.0e-3, reference 4.3e-3)
FIT = ("pupil_predicted", "mechanical", "B_pupil_px_per_stage", "B_mech_px_per_stage", "aligned_stack", "contrast_focus",
       "focus_loop", "check_first", "check", "correction_px", "pupil_shifts_px", "pupil_defocus_um", "pupil_astig_um",
       "pupil_poly_resid_rad", "mechanical_higher_order_rms_px", "pupil_parallax_rms_px",
       "fraction_of_drift_explained_by_pupil")          # top-level items of position_affine.json set by the fit
CAL_FIT = ("defocus_um", "astig_um", "position_affine")  # the same for calibration.json

def flat(d, p=""):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else enumerate(d)):
        if isinstance(v, (dict, list)) and not (isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)):
            out.update(flat(v, f"{p}{k}."))
        else: out[f"{p}{k}"] = v
    return out

def load(out_dir):
    """(position_affine, calibration) of a posaffine output folder."""
    out_dir = out_dir.rstrip("/") + "/"
    return json.load(open(out_dir + "results/position_affine.json")), json.load(open(out_dir + "work/calibration.json"))

def load_ref():
    return json.load(open(REF[0])), json.load(open(REF[1]))

def downstream(res, ref):
    """[(name, difference, tolerance, unit)] of the quantities used after posaffine (difference None if missing)."""
    (pa, cal), (rpa, rcal) = res, ref
    tol_um = float(rpa["settings"]["focus_tol_um"])
    fp, fr = flat(pa), flat(rpa)
    mis = [k for k, v in fr.items() if k.split(".")[-1] == "misfit" and isinstance(v, (int, float))]
    def d(f):
        try: return float(f())
        except (KeyError, TypeError, ValueError, ZeroDivisionError): return None
    return [("defocus", d(lambda: abs(cal["defocus_um"] - rcal["defocus_um"])), tol_um, "um"),
            ("astigmatism", d(lambda: np.abs(np.subtract(cal["astig_um"], rcal["astig_um"])).max()), tol_um, "um"),
            ("image corrections", d(lambda: np.abs(np.subtract(pa["correction_px"], rpa["correction_px"])).max()), TOL_CORRECTION_PX, "px"),
            (f"misfit ({len(mis)} items, relative)", d(lambda: max(abs(fp[k] - fr[k]) / abs(fr[k]) for k in mis)), TOL_MISFIT_REL, "")]

def compare(res, ref):
    """res, ref = (position_affine, calibration). Returns (verdict, report lines)."""
    worst = 0.0; diff_keys = []; hard = []; n = 0
    for a, b, fit in zip(res, ref, (FIT, CAL_FIT)):
        A, B = flat(a), flat(b)
        if fit is FIT: hard += [k for k in sorted(set(A) ^ set(B)) if not k.startswith("focus_loop.")]
        for k in sorted(set(A) & set(B)):
            try:
                x, y = np.asarray(A[k], float), np.asarray(B[k], float)
                if x.shape != y.shape: diff_keys.append(k); hard.append(k); continue
                both_nan = np.isnan(x) & np.isnan(y)
                if (np.isnan(x) ^ np.isnan(y)).any(): diff_keys.append(k); hard.append(k); continue
                r = np.abs(x - y)[~both_nan] / np.maximum(np.abs(y)[~both_nan], 1e-12)
                d = float(r.max()) if r.size else 0.0
                n += 1; worst = max(worst, d)
                if d > 0: diff_keys.append(f"{k} ({d:.2e})")
                if (d >= 1e-3 and k.split(".")[0] not in fit) or (isinstance(B[k], bool) and A[k] != B[k]): hard.append(k)
            except (TypeError, ValueError):
                if A[k] != B[k] and not k.startswith(("settings", "twin_source", "pupil_calibration_source")):
                    diff_keys.append(k); hard.append(k)
    cal, rcal = res[1], ref[1]
    lines = [f"pupil defocus {cal['defocus_um']:.1f} um (reference {rcal['defocus_um']:.1f}), astig {[round(v, 1) for v in cal['astig_um']]} um",
             f"{n} numeric items compared; max relative difference {worst:.3g}"]
    if diff_keys: lines.append("differing: " + "; ".join(diff_keys[:12]) + (" ..." if len(diff_keys) > 12 else ""))
    if hard: lines.append("missing, or differing outside the fit: " + "; ".join(hard[:12]) + (" ..." if len(hard) > 12 else ""))
    dn = downstream(res, ref)
    lines.append("quantities used downstream (tolerance for another computer type):")
    lines += [f"  {name:<28s} {'missing' if v is None else f'{v:.3g} {u}'.strip():>12s}  (tolerance {f'{t:g} {u}'.strip()})  "
              + ("OK" if v is not None and v <= t else "NG") for name, v, t, u in dn]
    within = not hard and all(v is not None and v <= t for _, v, t, _ in dn)
    verdict = ("一致" if worst == 0 and not diff_keys else
               "ほぼ一致（浮動小数点の差の範囲）" if worst < 1e-3 or within else "不一致（確認が必要）")
    return verdict, lines

if __name__ == "__main__":
    O = R + (sys.argv[1] if len(sys.argv) > 1 else "fpt_pipeline/fpt_output_32a_bf_aff2") + "/"
    verdict, lines = compare(load(O), load_ref())
    print("\n".join(lines))
    print("判定:", verdict)
