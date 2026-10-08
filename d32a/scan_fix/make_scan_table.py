# -*- coding: utf-8 -*-
"""Scan table for the objective-FZP scan (32a type), with the detector computed from the FZP pulses actually sent.
usage: python make_scan_table.py [out.csv]
columns: scan no, FZP X (pulse, 2.5 um), FZP Y (pulse, 0.2 um), detector X, detector Y, exposure (s)"""
import sys
import numpy as np
UX, UY = 2.5, 0.2                 # FZP stage: um per pulse
G = 211.1                         # detector units per um of lens shift (from the 32a CSV: 527.75/2.5 = 42.22/0.2)
RINGS = [(30 + 10 * r, n, 1 if r < 4 else 10) for r, n in enumerate([15, 20, 25, 30, 35, 40, 45, 50])]   # (radius um, points, exposure s)

def make_table():
    rows = []
    for R, n, t in RINGS:
        for i in range(n):
            th = 2 * np.pi * i / n
            ix = int(np.rint(R * np.cos(th) / UX))        # FZP X pulses (nearest, not floor)
            iy = int(np.rint(R * np.sin(th) / UY))        # FZP Y pulses
            dx = int(np.rint(-G * UX * ix))               # detector from the SAME integer pulses
            dy = int(np.rint(-G * UY * iy))
            rows.append((len(rows), ix, iy, dx, dy, t))
    return np.array(rows)

def check(T):
    """Detector must follow the FZP pulses: |detX + 527.75*FZP X| and |detY + 42.22*FZP Y| of at most ~1 detector unit
    (1 unit = 4.7 nm of lens shift) in every row. A half-pulse mismatch shows up as ~264 units (X) or ~21 units (Y)."""
    ex = np.abs(T[:, 3] + G * UX * T[:, 1]); ey = np.abs(T[:, 4] + G * UY * T[:, 2])
    return bool((ex <= 1).all() and (ey <= 1).all()), float(ex.max()), float(ey.max())

if __name__ == "__main__":
    T = make_table(); ok, ex, ey = check(T)
    out = sys.argv[1] if len(sys.argv) > 1 else "scan_table_fixed.csv"
    np.savetxt(out, T, fmt="%d", delimiter=",")
    print(f"{len(T)} rows -> {out}; detector follows FZP pulses: {ok} (max |error| {ex:.2f} / {ey:.2f} detector units)")
