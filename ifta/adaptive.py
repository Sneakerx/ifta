"""Adaptive beta selection for the Overdrive algorithm.

The closed-loop pole of the linearized iteration sits at z = beta - c, where
c is the local contraction rate of the Gerchberg-Saxton step. Choosing
beta = c minimizes the modulus of the closed-loop pole (deadbeat control).

`c` itself is unknown a priori but can be estimated online from successive
log-amplitude residuals: if u_k = (beta - c) u_{k-1} + noise, then a least
squares fit of (u_{k+1}, u_k) gives an estimate of (beta - c).
"""

from __future__ import annotations

import numpy as np

from ifta.metrics import EPS


def estimate_contraction(
    log_residual_history: list[float],
    *,
    window: int = 5,
    floor: float = 0.05,
    ceil: float = 0.95,
) -> float:
    """Estimate local GS contraction c from recent log-residual energies.

    If the closed-loop transient is u_k ≈ (beta - c) u_{k-1}, then the energy
    ratio (u_k^2 / u_{k-1}^2) approximates (beta - c)^2 and we can solve for c
    given the beta we used. Without that knowledge we fall back to a robust
    estimate of the local geometric decay rate of the residual.

    Returns
    -------
    c_hat : float in [floor, ceil]
        Estimated contraction. Clamped to (floor, ceil) to avoid numerical
        edge cases (c=0 implies one-shot convergence; c=1 implies stagnation).
    """
    h = np.asarray(log_residual_history, dtype=np.float64)
    if h.size < 2:
        return 0.5
    h = h[-window:]
    h = np.maximum(h, EPS)
    # Geometric ratio between successive squared residuals
    ratios = h[1:] / h[:-1]
    ratios = np.clip(ratios, EPS, 1.0)  # ratio <= 1 in stable regime
    rate = float(np.median(ratios))  # squared decay rate (beta - c)^2 in steady state
    c_hat = float(np.sqrt(rate))
    return float(np.clip(c_hat, floor, ceil))


def adaptive_beta_from_error_power(
    target_amp: np.ndarray,
    current_amp: np.ndarray,
    *,
    safety: float = 3.0,
    floor: float = 0.0,
    ceil: float = 0.95,
) -> float:
    """Conservative beta estimate from current error power.

    Mirrors the closed-form rule used in the dissertation for the Fienup
    amplitude variant (Eq. (451) of the dissertation):

        beta_min ≈ (f - safety*sqrt(P_e)) / (f + safety*sqrt(P_e))

    Here `f` is the typical laser amplitude and P_e the error power.
    Useful as a safe default when no contraction history is available yet.
    """
    f = float(np.mean(target_amp))
    err = (target_amp - current_amp) ** 2
    p_e = float(np.mean(err))
    sqrt_pe = np.sqrt(max(p_e, 0.0))
    if f + safety * sqrt_pe < EPS:
        return floor
    beta = (f - safety * sqrt_pe) / (f + safety * sqrt_pe)
    return float(np.clip(beta, floor, ceil))
