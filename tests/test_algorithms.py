"""Sanity checks for the IFTA implementation."""

from __future__ import annotations

import numpy as np
import pytest

from ifta import run_ifta, UPDATE_RULES


@pytest.fixture
def simple_target() -> np.ndarray:
    """A 32x32 grayscale image with two bright blobs."""
    rng = np.random.default_rng(0)
    img = np.zeros((32, 32), dtype=np.float64)
    img[8:14, 8:14] = 0.8
    img[18:26, 16:24] = 0.5
    img += 0.02 * rng.standard_normal(img.shape)
    return np.clip(img, 0.0, 1.0)


@pytest.mark.parametrize("rule", list(UPDATE_RULES.keys()))
def test_runs_without_crashing(simple_target: np.ndarray, rule: str) -> None:
    rng = np.random.default_rng(1)
    initial_phase = rng.uniform(0.0, 2.0 * np.pi, size=simple_target.shape)
    result = run_ifta(
        simple_target,
        update_rule=rule,
        iterations=5,
        initial_phase=initial_phase,
    )
    assert result.phase.shape == simple_target.shape
    assert result.intensity.shape == simple_target.shape
    assert len(result.psnr_history) == 5
    # Nothing should be NaN
    assert np.all(np.isfinite(result.intensity))
    assert np.all(np.isfinite(result.phase))


def test_gs_psnr_improves(simple_target: np.ndarray) -> None:
    """GS should not make the reconstruction strictly worse over 10 iters."""
    rng = np.random.default_rng(2)
    initial_phase = rng.uniform(0.0, 2.0 * np.pi, size=simple_target.shape)
    result = run_ifta(
        simple_target,
        update_rule="GS",
        iterations=10,
        initial_phase=initial_phase,
    )
    assert result.psnr_history[-1] >= result.psnr_history[0] - 1.0


def test_overdrive_at_fixed_point() -> None:
    """If the current reconstruction matches the target exactly, the
    Overdrive update should return the target (within floating-point noise).
    """
    from ifta.algorithms import overdrive

    target = np.array([[0.5, 0.7, 0.2], [0.9, 0.3, 0.6], [0.4, 0.8, 0.1]])
    # All states are at the fixed point
    out = overdrive(
        target_amp=target,
        current_amp=target,
        prev_virtual_amp=target,
        iteration=10,
        beta=0.5,
        warmup=3,
    )
    np.testing.assert_allclose(out, target, atol=1e-10)


def test_log_spectral_decay_overdrive(simple_target: np.ndarray) -> None:
    """The log-spectral residual should *not* increase on average for Overdrive."""
    rng = np.random.default_rng(3)
    initial_phase = rng.uniform(0.0, 2.0 * np.pi, size=simple_target.shape)
    result = run_ifta(
        simple_target,
        update_rule="Overdrive",
        iterations=20,
        initial_phase=initial_phase,
        beta=0.5,
    )
    h = np.asarray(result.log_spectral_history)
    # Compare second half to first half
    first_half = h[: len(h) // 2].mean()
    second_half = h[len(h) // 2 :].mean()
    assert second_half <= first_half
