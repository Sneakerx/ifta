"""Modern baselines for kinoform synthesis: HIO, RAAR, WGS, NWF, MGS.

These algorithms operate on full complex iterates (not just amplitudes), so
they don't fit the amplitude-only `update_rule` interface in `algorithms.py`
and are implemented as standalone runners that share its IFTAResult contract
and per-iteration metric bookkeeping.

References
----------
Fienup, J.R. (1982).  "Phase retrieval algorithms: a comparison."
    Applied Optics 21(15), 2758-2769.  [HIO]
Luke, D.R. (2005).  "Relaxed averaged alternating reflections for
    diffraction imaging."  Inverse Problems 21(1), 37-50.  [RAAR]
Di Leonardo, R., Ianni, F., Ruocco, G. (2007).  "Computer generation of
    optimal holograms for optical trap arrays."  Optics Express 15(4),
    1913-1922.  [WGS]
Candes, E.J., Li, X., Soltanolkotabi, M. (2015).  "Phase Retrieval via
    Wirtinger Flow: Theory and Algorithms."  IEEE Trans. Inf. Theory
    61(4), 1985-2007.  [WF -- we use a Nesterov-accelerated variant, NWF]

Note on Camera-in-the-Loop GS (Peng et al. 2020): not included here.
CITL is a method to close the simulation-to-hardware gap by replacing the
forward model with physical measurements during training; in a sim-only
benchmark it reduces to plain GS. See discussion in the paper.

NWF and MGS are *additive* momentum baselines included here to contrast
with the multiplicative log-amplitude momentum of the Overdrive rule.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ifta.algorithms import IFTAResult, _normalize_target
from ifta.metrics import psnr_intensity

EPS = 1e-12


# -----------------------------------------------------------------------------
# Shared scaffolding
# -----------------------------------------------------------------------------


def _setup(image: np.ndarray, initial_phase: Optional[np.ndarray],
           rng: Optional[np.random.Generator]):
    """Common preparation: normalize target, build laser, seed phase."""
    target_intensity = _normalize_target(image)
    target_intensity = np.fft.fftshift(target_intensity)
    target_amp = np.sqrt(target_intensity)
    laser_intensity_per_pixel = target_intensity.sum() / target_intensity.size
    laser_amp_scalar = float(np.sqrt(laser_intensity_per_pixel))
    laser_amp = np.full_like(target_intensity, laser_amp_scalar)
    if initial_phase is None:
        rng = rng if rng is not None else np.random.default_rng()
        phi = rng.uniform(0.0, 2.0 * np.pi, size=target_intensity.shape)
    else:
        phi = np.asarray(initial_phase, dtype=np.float64)
    grating = laser_amp * np.exp(1j * phi)
    return target_intensity, target_amp, laser_amp, laser_amp_scalar, grating


def _record(target_intensity, target_amp, U, current_amp,
            psnr_hist, mse_hist, logspec_hist):
    """Per-iteration PSNR / MSE / LSD recording (matches run_ifta)."""
    current_intensity = current_amp ** 2
    scale = target_intensity.sum() / max(current_intensity.sum(), EPS)
    scaled = scale * current_intensity
    err = target_intensity - scaled
    mse_hist.append(float(np.mean(err ** 2)))
    psnr_hist.append(psnr_intensity(target_intensity, scaled))
    mask = target_amp > EPS
    if mask.any():
        log_resid = np.log(np.maximum(current_amp[mask], EPS)) - np.log(target_amp[mask])
        logspec_hist.append(float(np.mean(log_resid ** 2)))
    else:
        logspec_hist.append(0.0)


def _final_result(grating_or_phi, laser_amp, iterations, rule_name,
                  psnr_hist, mse_hist, logspec_hist) -> IFTAResult:
    """Compute final intensity from the last hologram-plane phase."""
    if np.iscomplexobj(grating_or_phi):
        phi = np.mod(np.angle(grating_or_phi), 2.0 * np.pi)
    else:
        phi = grating_or_phi
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


# -----------------------------------------------------------------------------
# HIO (Fienup 1982), kinoform adaptation
# -----------------------------------------------------------------------------


def run_hio(
    image: np.ndarray,
    *,
    iterations: int = 20,
    beta: float = 0.9,
    warmup: int = 3,
    threshold: float = 0.1,
    initial_phase: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    record_history: bool = True,
    **kwargs,
) -> IFTAResult:
    """Hybrid Input-Output (HIO) for phase-only kinoform synthesis.

    For phase retrieval, HIO uses the *object-plane* support constraint with
    feedback: pixels satisfying the constraint are kept, violators are pulled
    by ``g_prev - β · g_after_M``.  For kinoform synthesis the corresponding
    constraint is the constant amplitude ``|g| = laser_amp`` in the hologram
    plane.  We implement the standard adaptation: a pixel is considered to
    "satisfy" S when its amplitude lies within ``threshold`` of laser_amp; on
    those pixels we apply the standard projection ``laser_amp·exp(j·arg(g))``,
    on the others we apply the Fienup feedback ``g_prev - β · g_after_M``.

    For warmup iterations we run plain GS.
    """
    target_intensity, target_amp, laser_amp, laser_amp_scalar, grating = \
        _setup(image, initial_phase, rng)

    psnr_hist, mse_hist, logspec_hist = [], [], []

    for k in range(iterations):
        U = np.fft.fft2(grating, norm="ortho")
        if record_history:
            _record(target_intensity, target_amp, U, np.abs(U),
                    psnr_hist, mse_hist, logspec_hist)

        # Image-plane projection P_M
        U_proj = target_amp * np.exp(1j * np.angle(U))
        u_after_M = np.fft.ifft2(U_proj, norm="ortho")

        if k < warmup:
            # GS warmup: hard projection onto S
            grating = laser_amp * np.exp(1j * np.angle(u_after_M))
            continue

        # HIO selection: where |u_after_M| ~ laser_amp, project; else, feedback
        amp_after_M = np.abs(u_after_M)
        satisfies_S = np.abs(amp_after_M - laser_amp_scalar) <= threshold * laser_amp_scalar

        proj = laser_amp * np.exp(1j * np.angle(u_after_M))
        feedback = grating - beta * u_after_M
        grating = np.where(satisfies_S, proj, feedback)

    return _final_result(grating, laser_amp, iterations, "HIO",
                         psnr_hist, mse_hist, logspec_hist)


# -----------------------------------------------------------------------------
# RAAR (Luke 2005)
# -----------------------------------------------------------------------------


def run_raar(
    image: np.ndarray,
    *,
    iterations: int = 20,
    beta: float = 0.75,
    warmup: int = 3,
    initial_phase: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    record_history: bool = True,
    **kwargs,
) -> IFTAResult:
    """Relaxed Averaged Alternating Reflections (RAAR) for kinoform synthesis.

    Standard formulation (Luke 2005):

        T_β(g) = (β/2) · (R_M(R_S(g)) + g) + (1 − β) · P_M(g)

    where the projections are
        P_S(g) = laser_amp · exp(j·arg(g))
        P_M(g) = IFFT(target_amp · exp(j·arg(FFT(g))))
    and the reflections are R_X = 2 P_X − I.

    The iterate is a *complex* hologram-plane field that does not generally
    satisfy |g| = laser_amp; only at the very end (and for evaluation) do we
    project onto the hologram-plane set.
    """
    target_intensity, target_amp, laser_amp, laser_amp_scalar, grating = \
        _setup(image, initial_phase, rng)

    psnr_hist, mse_hist, logspec_hist = [], [], []

    for k in range(iterations):
        U = np.fft.fft2(grating, norm="ortho")
        if record_history:
            _record(target_intensity, target_amp, U, np.abs(U),
                    psnr_hist, mse_hist, logspec_hist)

        # P_M(g) via image-plane projection
        U_proj = target_amp * np.exp(1j * np.angle(U))
        P_M_g = np.fft.ifft2(U_proj, norm="ortho")

        if k < warmup:
            grating = laser_amp * np.exp(1j * np.angle(P_M_g))
            continue

        # P_S(g)
        P_S_g = laser_amp * np.exp(1j * np.angle(grating))
        R_S_g = 2.0 * P_S_g - grating

        # P_M(R_S(g))
        G_R = np.fft.fft2(R_S_g, norm="ortho")
        G_R_proj = target_amp * np.exp(1j * np.angle(G_R))
        P_M_RS = np.fft.ifft2(G_R_proj, norm="ortho")
        R_M_RS = 2.0 * P_M_RS - R_S_g

        grating = 0.5 * beta * (R_M_RS + grating) + (1.0 - beta) * P_M_g

    # Final hologram-plane projection so the returned phase is realizable
    grating = laser_amp * np.exp(1j * np.angle(grating))

    return _final_result(grating, laser_amp, iterations, "RAAR",
                         psnr_hist, mse_hist, logspec_hist)


# -----------------------------------------------------------------------------
# 2nd-order Overdrive (heuristic extension)
# -----------------------------------------------------------------------------


def run_overdrive2(
    image: np.ndarray,
    *,
    iterations: int = 20,
    alpha: float = 0.7,
    beta2: float = 0.0,
    warmup: int = 4,
    initial_phase: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    record_history: bool = True,
    renormalize: bool = True,
    **kwargs,
) -> IFTAResult:
    """2nd-order Overdrive (heuristic extension).

        |G'_k| = |G'_{k-1}|^α · |G'_{k-2}|^β · |F|^(2-α-β) / |G_k|

    Log form:  u_k = α·u_{k-1} + β·u_{k-2} - v_k.  For β=0 this reduces
    to 1st-order Overdrive with parameter α; the 2-parameter family is
    strictly more expressive.

    Note: the empirical η-whiteness test (scripts/test_eta_whiteness.py)
    showed the AR(1) closed-loop model with white η fails three independent
    consistency checks. The 2nd-order family here is therefore purely
    heuristic — the optimum (α*, β*) is to be found by empirical search,
    not derived from a closed-form formula.
    """
    target_intensity, target_amp, laser_amp, laser_amp_scalar, grating = \
        _setup(image, initial_phase, rng)

    prev_virtual = target_amp.copy()
    prev_prev_virtual = target_amp.copy()
    psnr_hist, mse_hist, logspec_hist = [], [], []

    for k in range(iterations):
        U = np.fft.fft2(grating, norm="ortho")
        current_amp = np.abs(U)
        if record_history:
            _record(target_intensity, target_amp, U, current_amp,
                    psnr_hist, mse_hist, logspec_hist)

        if k < warmup:
            new_amp = target_amp.copy()
        else:
            safe = np.where(current_amp > EPS, current_amp, EPS)
            new_amp = (
                np.power(np.maximum(prev_virtual, EPS), alpha)
                * np.power(np.maximum(prev_prev_virtual, EPS), beta2)
                * np.power(np.maximum(target_amp, EPS), 2.0 - alpha - beta2)
                / safe
            )
            new_amp = np.where(target_amp > 0, new_amp, 0.0)
            if renormalize:
                tot = target_amp.sum()
                cur = new_amp.sum()
                if cur > EPS:
                    new_amp = new_amp * (tot / cur)

        U_new = new_amp * np.exp(1j * np.angle(U))
        u_new = np.fft.ifft2(U_new, norm="ortho")
        phi = np.mod(np.angle(u_new), 2.0 * np.pi)
        grating = laser_amp * np.exp(1j * phi)

        prev_prev_virtual = prev_virtual
        prev_virtual = new_amp

    return _final_result(grating, laser_amp, iterations,
                         f"Overdrive2(α={alpha},β={beta2})",
                         psnr_hist, mse_hist, logspec_hist)


# -----------------------------------------------------------------------------
# Weighted GS (Di Leonardo 2007), generalized to dense grayscale
# -----------------------------------------------------------------------------


def run_wgs(
    image: np.ndarray,
    *,
    iterations: int = 20,
    warmup: int = 3,
    initial_phase: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    record_history: bool = True,
    **kwargs,
) -> IFTAResult:
    """Weighted Gerchberg-Saxton (Di Leonardo 2007), grayscale generalization.

    Originally proposed for spot-array generation: weights ``w_m`` are
    cumulatively updated as ``w_m^{k+1} = w_m^k · sqrt(target_m / |G_m^k|)``,
    and the image-plane target amplitude at each spot is replaced by
    ``w_m · target_m`` before projection.  For dense grayscale targets we
    apply the same rule pixelwise:

        w(x)^{k+1} = w(x)^k · sqrt( target_amp(x) / |G^k(x)| )
        G_proj(x)  = w(x) · target_amp(x) · exp(j·arg(G^k(x)))

    Numerically, this is the cumulative-weight variant of the Bengtsson rule
    (Section 7 of the paper); we include it here for completeness.
    """
    target_intensity, target_amp, laser_amp, laser_amp_scalar, grating = \
        _setup(image, initial_phase, rng)

    weights = np.ones_like(target_amp)
    psnr_hist, mse_hist, logspec_hist = [], [], []

    for k in range(iterations):
        U = np.fft.fft2(grating, norm="ortho")
        current_amp = np.abs(U)
        if record_history:
            _record(target_intensity, target_amp, U, current_amp,
                    psnr_hist, mse_hist, logspec_hist)

        if k >= warmup:
            ratio = np.divide(
                target_amp,
                np.maximum(current_amp, EPS),
                out=np.ones_like(target_amp),
                where=target_amp > EPS,
            )
            weights = weights * np.sqrt(np.maximum(ratio, EPS))
            # Keep the weighted target's mean amplitude on par with the
            # original target's, otherwise total power drifts.
            mean_w = float(np.mean(weights))
            if mean_w > EPS:
                weights = weights / mean_w
            target_weighted = weights * target_amp
        else:
            target_weighted = target_amp

        U_proj = target_weighted * np.exp(1j * np.angle(U))
        u = np.fft.ifft2(U_proj, norm="ortho")
        grating = laser_amp * np.exp(1j * np.angle(u))

    return _final_result(grating, laser_amp, iterations, "WGS",
                         psnr_hist, mse_hist, logspec_hist)


# -----------------------------------------------------------------------------
# Momentum-GS (Polyak heavy ball on the phase)
# -----------------------------------------------------------------------------


def run_momentum_gs(
    image: np.ndarray,
    *,
    iterations: int = 20,
    momentum: float = 0.5,
    warmup: int = 3,
    initial_phase: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    record_history: bool = True,
    **kwargs,
) -> IFTAResult:
    """Heavy-ball Gerchberg-Saxton: additive Polyak momentum on the phase.

    For every pixel, after the GS phase update phi_GS = arg(u_after_M),
    apply Polyak's heavy-ball:

        delta_k       = wrap(phi_GS - phi_{k-1})    # in (-pi, pi]
        phi_k         = phi_{k-1} + delta_k + momentum * (phi_{k-1} - phi_{k-2})

    Equivalent to running GS with an extra "memory" term in the phase
    domain. Sits in the *additive Euclidean* momentum family and is the
    natural direct competitor to the multiplicative log-momentum of
    Overdrive.
    """
    target_intensity, target_amp, laser_amp, laser_amp_scalar, grating = \
        _setup(image, initial_phase, rng)

    phi = np.mod(np.angle(grating), 2.0 * np.pi)
    phi_prev = phi.copy()
    psnr_hist, mse_hist, logspec_hist = [], [], []

    for k in range(iterations):
        U = np.fft.fft2(grating, norm="ortho")
        if record_history:
            _record(target_intensity, target_amp, U, np.abs(U),
                    psnr_hist, mse_hist, logspec_hist)

        # Image-plane projection (target amplitude, keep phase)
        U_proj = target_amp * np.exp(1j * np.angle(U))
        u_after_M = np.fft.ifft2(U_proj, norm="ortho")
        phi_gs = np.angle(u_after_M)

        if k < warmup:
            phi_new = np.mod(phi_gs, 2.0 * np.pi)
        else:
            # Wrap phase change into (-pi, pi]
            delta = np.angle(np.exp(1j * (phi_gs - phi)))
            # Polyak heavy ball
            phi_new = np.mod(phi + delta + momentum * (phi - phi_prev), 2.0 * np.pi)

        phi_prev = phi
        phi = phi_new
        grating = laser_amp * np.exp(1j * phi)

    return _final_result(phi, laser_amp, iterations,
                         f"MGS(mom={momentum})",
                         psnr_hist, mse_hist, logspec_hist)


# -----------------------------------------------------------------------------
# Nesterov-accelerated Wirtinger Flow (NWF)
# -----------------------------------------------------------------------------


def run_nwf(
    image: np.ndarray,
    *,
    iterations: int = 20,
    step: float = 0.1,
    warmup: int = 3,
    initial_phase: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    record_history: bool = True,
    **kwargs,
) -> IFTAResult:
    """Nesterov-accelerated Wirtinger Flow, adapted to kinoform synthesis.

    Standard WF minimizes
        f(z) = ||A|z|^2 - y||^2,     where y = |F|^2 (target intensity).
    The Wirtinger gradient is
        grad_z f = A^*( (|Az|^2 - y) * Az ).
    For us A is the (orthonormal) FFT, so
        grad_z f = IFFT( (|U|^2 - y) * U ),     U = FFT(z).

    Nesterov's accelerated variant uses a lookahead:
        y_k = z_k + gamma_k (z_k - z_{k-1})
        z_{k+1} = y_k - step * grad_z f(y_k)
    with gamma_k = (k-1)/(k+2). After each update we project onto the
    hologram-plane constraint set ``|z| = laser_amp``.

    NWF is the principal *additive* modern-momentum competitor and
    represents what the 2026 literature is doing (e.g. Nesterov-WF for
    dual-plane phase retrieval). Note that strictly speaking Wirtinger Flow
    is a phase-*retrieval* algorithm; we apply it here heuristically to
    kinoform synthesis, with the hologram-plane projection serving as a
    constraint enforcement after each step.
    """
    target_intensity, target_amp, laser_amp, laser_amp_scalar, grating = \
        _setup(image, initial_phase, rng)

    z_prev = grating.copy()
    z = grating.copy()
    psnr_hist, mse_hist, logspec_hist = [], [], []

    for k in range(iterations):
        U_eval = np.fft.fft2(z, norm="ortho")
        if record_history:
            _record(target_intensity, target_amp, U_eval, np.abs(U_eval),
                    psnr_hist, mse_hist, logspec_hist)

        if k < warmup:
            # GS warmup
            U_proj = target_amp * np.exp(1j * np.angle(U_eval))
            u_new = np.fft.ifft2(U_proj, norm="ortho")
            z_new = laser_amp * np.exp(1j * np.angle(u_new))
        else:
            gamma = max(0.0, (k - 1.0) / (k + 2.0))
            y = z + gamma * (z - z_prev)
            U_y = np.fft.fft2(y, norm="ortho")
            residual = np.abs(U_y) ** 2 - target_intensity
            grad = np.fft.ifft2(residual * U_y, norm="ortho")
            z_new = y - step * grad
            # Project onto |z| = laser_amp
            z_new = laser_amp * np.exp(1j * np.angle(z_new))

        z_prev = z
        z = z_new
        grating = z

    return _final_result(grating, laser_amp, iterations,
                         f"NWF(step={step})",
                         psnr_hist, mse_hist, logspec_hist)
