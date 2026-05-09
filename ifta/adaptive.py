"""Adaptive beta selection for the Overdrive algorithm.

The closed-loop pole of the linearized iteration sits at z = beta - c, where
c is the local contraction rate of the Gerchberg-Saxton step. Choosing
beta = c minimizes the modulus of the closed-loop pole (deadbeat control).

Estimating c from the log-spectral history h_k = sum v_k^2:

    h_k = V_inf + alpha * rate^k,    rate = (beta - c)^2

so the floor V_inf must be subtracted before fitting a geometric decay to
recover `rate`. Then c = beta - sqrt(rate) (or beta + sqrt(rate) if the
first branch is negative). This requires knowing the beta that *produced*
the history.
"""

from __future__ import annotations

import numpy as np

from ifta.metrics import EPS


def estimate_contraction(
    log_residual_history: list[float] | np.ndarray,
    beta_used: float,
    *,
    window: int = 8,
    floor_pad_rel: float = 1e-3,
    bound_floor: float = 0.05,
    bound_ceil: float = 0.95,
    fallback: float = 0.5,
) -> float:
    """Estimate local GS contraction c from a log-residual history.

    Parameters
    ----------
    log_residual_history : sequence of float
        h_k = mean of (log|G_k| - log|F|)^2 per iteration.
    beta_used : float
        The mixing parameter beta that produced this history. Required to
        disambiguate c = beta +/- sqrt(rate).
    window : int
        Number of trailing samples to fit.
    floor_pad_rel : float
        Relative pad subtracted from h_min to avoid log(0); fraction of
        the window's range.
    bound_floor, bound_ceil : float
        Clamp output c to (bound_floor, bound_ceil) — c=0 is deadbeat,
        c=1 is stagnation; both are numerical edge cases.
    fallback : float
        Returned if the history is too short or the fit is degenerate.

    Returns
    -------
    c_hat : float in [bound_floor, bound_ceil]
    """
    h = np.asarray(log_residual_history, dtype=np.float64)
    if h.size < 4:
        return fallback
    h = h[-window:]
    if h.size < 4:
        return fallback

    h_min = float(h.min())
    h_max = float(h.max())
    pad = max(floor_pad_rel * (h_max - h_min), EPS)
    delta = h - (h_min - pad)
    delta = np.maximum(delta, EPS)

    k = np.arange(delta.size, dtype=np.float64)
    log_delta = np.log(delta)
    slope, _ = np.polyfit(k, log_delta, 1)

    if not np.isfinite(slope):
        return fallback

    rate = float(np.exp(slope))            # (beta - c)^2 under AR(1) model
    rate = float(np.clip(rate, EPS, 1.0 - EPS))
    abs_diff = float(np.sqrt(rate))        # |beta - c|

    # Branch selection: prefer c in [0, 1], staying close to beta.
    candidates = [beta_used - abs_diff, beta_used + abs_diff]
    valid = [c for c in candidates if 0.0 <= c <= 1.0]
    if not valid:
        c_hat = float(np.clip(candidates[0], bound_floor, bound_ceil))
    elif len(valid) == 1:
        c_hat = valid[0]
    else:
        # Two valid branches: prefer the one closer to beta_used (smaller pole)
        c_hat = min(valid, key=lambda c: abs(beta_used - c))

    return float(np.clip(c_hat, bound_floor, bound_ceil))


def estimate_contraction_legacy(
    log_residual_history: list[float] | np.ndarray,
    *,
    window: int = 5,
    floor: float = 0.05,
    ceil: float = 0.95,
) -> float:
    """Legacy estimator (kept for ablation; do not use).

    Computes sqrt(median(h_{k+1}/h_k)). Saturates near 1 in steady state
    because the floor V_inf is not subtracted. Independent of beta_used.
    """
    h = np.asarray(log_residual_history, dtype=np.float64)
    if h.size < 2:
        return 0.5
    h = h[-window:]
    h = np.maximum(h, EPS)
    ratios = h[1:] / h[:-1]
    ratios = np.clip(ratios, EPS, 1.0)
    rate = float(np.median(ratios))
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
