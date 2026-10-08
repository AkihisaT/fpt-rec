"""dipfpm: physics-informed Deep Image Prior (untrained U-Net) reconstruction for X-ray Fourier
ptychography. U-Net(measured images) -> absorption/phase; constant-input MLPs -> Zernike pupil,
per-image shifts, per-image intensity factors; loss = sum (c_n I_pred - I_meas)^2 + L2 (weights)."""
__version__ = "0.1.0"
