"""Quality metrics for hologram reconstruction.

PSNR and MSE follow standard image-quality conventions.
The log-spectral distance (and Itakura–Saito divergence) are the natural
Lyapunov candidates for *multiplicative* update rules like Overdrive — they
decay monotonically along the iteration even when the L^2 hologram-plane
residual transiently grows.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def mse(target: np.ndarray, reconstruction: np.ndarray) -> float:
    """Mean squared error between two images of equal shape."""
    return float(np.mean((target.astype(np.float64) - reconstruction.astype(np.float64)) ** 2))


def psnr(target: np.ndarray, reconstruction: np.ndarray, *, peak: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB (default peak: 1.0)."""
    e = mse(target, reconstruction)
    if e <= EPS:
        return float("inf")
    return float(10.0 * np.log10(peak**2 / e))


def psnr_intensity(target_intensity: np.ndarray, recon_intensity: np.ndarray) -> float:
    """PSNR on intensity images, peak = max of target."""
    peak = max(float(target_intensity.max()), EPS)
    return psnr(target_intensity, recon_intensity, peak=peak)


def log_spectral_distance(
    target_amp: np.ndarray, recon_amp: np.ndarray, *, mask: np.ndarray | None = None
) -> float:
    """Mean squared log-amplitude residual.

    LSD = mean over (target>0) of (log|G| - log|F|)^2.

    Natural Lyapunov function for multiplicative IFTA variants: at each
    iteration the closed-loop log-error u_k satisfies u_k = (beta - c) u_{k-1}
    plus noise, so this quantity decays geometrically with rate (beta - c)^2.
    """
    target_amp = np.asarray(target_amp, dtype=np.float64)
    recon_amp = np.asarray(recon_amp, dtype=np.float64)
    if mask is None:
        mask = target_amp > EPS
    if not mask.any():
        return 0.0
    log_resid = np.log(np.maximum(recon_amp[mask], EPS)) - np.log(target_amp[mask])
    return float(np.mean(log_resid**2))


def itakura_saito(
    target_intensity: np.ndarray,
    recon_intensity: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float:
    """Itakura–Saito divergence on intensities.

    IS(F, G) = mean of [ F/G - log(F/G) - 1 ] over the target support.

    Always non-negative; equals zero iff F == G. Dual to LSD: both serve as
    Lyapunov functions for log-domain iterations.
    """
    target_intensity = np.asarray(target_intensity, dtype=np.float64)
    recon_intensity = np.asarray(recon_intensity, dtype=np.float64)
    if mask is None:
        mask = target_intensity > EPS
    if not mask.any():
        return 0.0
    g = np.maximum(recon_intensity[mask], EPS)
    f = target_intensity[mask]
    ratio = f / g
    return float(np.mean(ratio - np.log(ratio) - 1.0))
