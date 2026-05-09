"""
IFTA core loop and update rules for phase-only hologram synthesis.

The iterative Fourier transform algorithm alternates between two domains:

    hologram plane  <-- inverse FFT --   image plane
                    -- forward FFT -->

In the hologram plane, the amplitude is fixed to the laser profile (here: unit
amplitude) and only the phase is free. In the image plane, an algorithm-specific
*update rule* re-imposes the target amplitude on the current reconstruction.

Algorithms differ only in their update rule. The Overdrive rule is

    |G_k'| = |G_{k-1}'|^beta * |F|^(2-beta) / |G_k|     (Bernau, 2007)

with beta in (0, 1). A log-domain analysis (see paper) shows this is an IIR
filter on log-amplitudes with pole at z = beta, which makes the convergence
behavior tractable.

Reference Matlab implementation: JDT_IFTA.m (this repository).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ifta.metrics import psnr_intensity

# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------


@dataclass
class IFTAResult:
    """Outcome of one IFTA run."""

    phase: np.ndarray  # final hologram phase, shape (H, W), in [0, 2*pi)
    intensity: np.ndarray  # final reconstructed intensity in image plane
    psnr_history: list[float] = field(default_factory=list)
    mse_history: list[float] = field(default_factory=list)
    # log-spectral residual; useful as Lyapunov function for multiplicative rules
    log_spectral_history: list[float] = field(default_factory=list)
    iterations: int = 0
    update_rule: str = ""


# -----------------------------------------------------------------------------
# Update rules: callables mapping (state) -> new virtual amplitude
# -----------------------------------------------------------------------------


def _renormalize_amplitude(virtual_amp: np.ndarray, target_amp: np.ndarray) -> np.ndarray:
    """Rescale virtual amplitude so its sum matches the target's (Parseval)."""
    target_sum = target_amp.sum()
    current_sum = virtual_amp.sum()
    if current_sum <= 0:
        return target_amp.copy()
    return virtual_amp * (target_sum / current_sum)


def gerchberg_saxton(
    target_amp: np.ndarray,
    current_amp: np.ndarray,
    prev_virtual_amp: np.ndarray,
    iteration: int,
    **kwargs,
) -> np.ndarray:
    """Vanilla GS: replace current amplitude with target."""
    return target_amp.copy()


def fienup_amplitude(
    target_amp: np.ndarray,
    current_amp: np.ndarray,
    prev_virtual_amp: np.ndarray,
    iteration: int,
    *,
    beta: float = 0.5,
    warmup: int = 3,
    renormalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """Fienup amplitude variant (1978) with linear over-correction.

    |G'| = |F| + beta * (|F| - |G|)
    """
    if iteration < warmup:
        return target_amp.copy()
    new_amp = target_amp + beta * (target_amp - current_amp)
    new_amp = np.maximum(new_amp, 0.0)
    if renormalize:
        new_amp = _renormalize_amplitude(new_amp, target_amp)
    return new_amp


def bengtsson(
    target_amp: np.ndarray,
    current_amp: np.ndarray,
    prev_virtual_amp: np.ndarray,
    iteration: int,
    *,
    beta: float = 0.35,
    warmup: int = 3,
    eps: float = 1e-12,
    renormalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """Bengtsson (1998): |G'| = |G_{k-1}'| * (|F|/|G|)^beta."""
    if iteration < warmup:
        return target_amp.copy()
    ratio = np.divide(
        target_amp,
        current_amp,
        out=np.ones_like(target_amp),
        where=current_amp > eps,
    )
    new_amp = prev_virtual_amp * ratio**beta
    if renormalize:
        new_amp = _renormalize_amplitude(new_amp, target_amp)
    return new_amp


def overdrive(
    target_amp: np.ndarray,
    current_amp: np.ndarray,
    prev_virtual_amp: np.ndarray,
    iteration: int,
    *,
    beta: float = 0.5,
    warmup: int = 3,
    eps: float = 1e-12,
    renormalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """Overdrive (Bernau, 2007):

        |G'| = |G_{k-1}'|^beta * |F|^(2-beta) / |G|

    Dimensionally balanced (exponents sum to one). The log-amplitude form is
    a first-order IIR filter with pole at z = beta — see paper.

    Note on renormalization: the global Parseval rescale in
    _renormalize_amplitude is *not* part of the analytic update rule. To run
    the algorithm exactly as analyzed in the paper, set renormalize=False.
    """
    if iteration < warmup:
        return target_amp.copy()

    # Guard against division by zero
    safe_current = np.where(current_amp > eps, current_amp, eps)

    new_amp = (
        np.power(np.maximum(prev_virtual_amp, eps), beta)
        * np.power(np.maximum(target_amp, eps), 2.0 - beta)
        / safe_current
    )
    # Pixels where the target is exactly zero must stay zero (no division done)
    new_amp = np.where(target_amp > 0, new_amp, 0.0)
    if renormalize:
        new_amp = _renormalize_amplitude(new_amp, target_amp)
    return new_amp


def overdrive_adaptive(
    target_amp: np.ndarray,
    current_amp: np.ndarray,
    prev_virtual_amp: np.ndarray,
    iteration: int,
    *,
    contraction_estimate: float = 0.5,
    warmup: int = 3,
    eps: float = 1e-12,
    renormalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """Overdrive with beta = c, the local GS contraction estimate.

    The closed-loop pole sits at z = beta - c, so beta = c gives the fastest
    asymptotic decay (deadbeat-like). `contraction_estimate` should be updated
    externally; see ifta.adaptive.
    """
    return overdrive(
        target_amp,
        current_amp,
        prev_virtual_amp,
        iteration,
        beta=contraction_estimate,
        warmup=warmup,
        eps=eps,
        renormalize=renormalize,
    )


UPDATE_RULES: dict[str, Callable] = {
    "GS": gerchberg_saxton,
    "Fienup": fienup_amplitude,
    "Bengtsson": bengtsson,
    "Overdrive": overdrive,
    "OverdriveAdaptive": overdrive_adaptive,
}


# -----------------------------------------------------------------------------
# Core IFTA loop
# -----------------------------------------------------------------------------


def _normalize_target(image: np.ndarray) -> np.ndarray:
    """Convert image to nonnegative target intensity, sum-normalized."""
    img = image.astype(np.float64)
    if img.ndim == 3:
        img = img.mean(axis=-1)  # naive grayscale conversion
    img = np.maximum(img, 0.0)
    if img.max() > 1.0 + 1e-9:
        img = img / 255.0
    return img


def run_ifta(
    image: np.ndarray,
    *,
    update_rule: str | Callable = "GS",
    iterations: int = 20,
    initial_phase: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    record_history: bool = True,
    **rule_kwargs,
) -> IFTAResult:
    """Run the iterative Fourier transform algorithm with a given update rule.

    Parameters
    ----------
    image : ndarray
        Target intensity image, real-valued, shape (H, W). Will be normalized
        to [0, 1]. RGB images are averaged to grayscale.
    update_rule : str or callable
        Either a name from UPDATE_RULES or a callable with signature
        ``rule(target_amp, current_amp, prev_virtual_amp, iteration, **kwargs)``.
    iterations : int
        Number of forward+inverse FFT cycles.
    initial_phase : ndarray, optional
        Starting hologram phase. If None, drawn uniformly from [0, 2*pi).
    rng : np.random.Generator, optional
        For reproducible random initial phase.
    record_history : bool
        If True, store PSNR/MSE/log-spectral history per iteration.
    **rule_kwargs : dict
        Forwarded to the update rule (e.g. ``beta=0.7``).

    Returns
    -------
    IFTAResult
    """
    rule_fn = UPDATE_RULES[update_rule] if isinstance(update_rule, str) else update_rule
    rule_name = update_rule if isinstance(update_rule, str) else update_rule.__name__

    # --- Normalize target intensity and derive target amplitude
    target_intensity = _normalize_target(image)
    # Centered FFT convention: shift target so DC is at the corner for FFT
    target_intensity = np.fft.fftshift(target_intensity)
    target_amp = np.sqrt(target_intensity)

    H, W = target_intensity.shape

    # --- Laser amplitude (uniform). Power is matched to target via Parseval.
    laser_intensity_per_pixel = target_intensity.sum() / target_intensity.size
    laser_amp = np.full_like(target_intensity, np.sqrt(laser_intensity_per_pixel))

    # --- Initial phase
    if initial_phase is None:
        rng = rng if rng is not None else np.random.default_rng()
        phi = rng.uniform(0.0, 2.0 * np.pi, size=target_intensity.shape)
    else:
        phi = np.asarray(initial_phase, dtype=np.float64)

    # --- State
    grating = laser_amp * np.exp(1j * phi)
    prev_virtual_amp = target_amp.copy()  # |G_{-1}'| := |F|

    psnr_hist: list[float] = []
    mse_hist: list[float] = []
    logspec_hist: list[float] = []

    eps = 1e-12

    for k in range(iterations):
        # Forward: hologram -> image
        U = np.fft.fft2(grating, norm="ortho")
        current_amp = np.abs(U)
        current_intensity = current_amp**2

        if record_history:
            # PSNR / MSE on intensity (rescaled to compensate for total power)
            scale = target_intensity.sum() / max(current_intensity.sum(), eps)
            scaled = scale * current_intensity
            err = target_intensity - scaled
            mse_val = float(np.mean(err**2))
            mse_hist.append(mse_val)
            psnr_hist.append(psnr_intensity(target_intensity, scaled))

            # Log-spectral residual: only over support of target
            mask = target_amp > eps
            if mask.any():
                log_resid = np.log(np.maximum(current_amp[mask], eps)) - np.log(
                    target_amp[mask]
                )
                logspec_hist.append(float(np.mean(log_resid**2)))
            else:
                logspec_hist.append(0.0)

        # Apply update rule -> new virtual amplitude
        new_virtual_amp = rule_fn(
            target_amp=target_amp,
            current_amp=current_amp,
            prev_virtual_amp=prev_virtual_amp,
            iteration=k,
            **rule_kwargs,
        )

        # Re-impose Fourier-domain constraint: keep phase, replace amplitude
        U_new = new_virtual_amp * np.exp(1j * np.angle(U))

        # Inverse: image -> hologram
        u_new = np.fft.ifft2(U_new, norm="ortho")

        # Apply hologram-domain constraint: phase only, fixed amplitude
        phi = np.mod(np.angle(u_new), 2.0 * np.pi)
        grating = laser_amp * np.exp(1j * phi)

        prev_virtual_amp = new_virtual_amp

    # --- Final reconstruction with the latest phase
    U_final = np.fft.fft2(laser_amp * np.exp(1j * phi), norm="ortho")
    final_intensity = np.fft.ifftshift(np.abs(U_final) ** 2)
    final_phase = np.fft.ifftshift(phi)

    return IFTAResult(
        phase=final_phase,
        intensity=final_intensity,
        psnr_history=psnr_hist,
        mse_history=mse_hist,
        log_spectral_history=logspec_hist,
        iterations=iterations,
        update_rule=rule_name,
    )
