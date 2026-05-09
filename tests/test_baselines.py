"""Sanity checks for the modern-baseline runners."""

from __future__ import annotations

import numpy as np
import pytest

from ifta.baselines import run_hio, run_raar, run_wgs


@pytest.fixture
def simple_target() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.zeros((32, 32), dtype=np.float64)
    img[8:14, 8:14] = 0.8
    img[18:26, 16:24] = 0.5
    img += 0.02 * rng.standard_normal(img.shape)
    return np.clip(img, 0.0, 1.0)


@pytest.fixture
def initial_phase(simple_target: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.uniform(0.0, 2.0 * np.pi, size=simple_target.shape)


def test_hio_runs(simple_target: np.ndarray, initial_phase: np.ndarray) -> None:
    result = run_hio(simple_target, iterations=5, beta=0.5, threshold=0.3,
                    initial_phase=initial_phase)
    assert result.phase.shape == simple_target.shape
    assert result.intensity.shape == simple_target.shape
    assert len(result.psnr_history) == 5
    assert np.all(np.isfinite(result.intensity))


def test_raar_runs(simple_target: np.ndarray, initial_phase: np.ndarray) -> None:
    result = run_raar(simple_target, iterations=5, beta=0.5,
                     initial_phase=initial_phase)
    assert result.phase.shape == simple_target.shape
    assert len(result.psnr_history) == 5
    assert np.all(np.isfinite(result.intensity))


def test_wgs_runs(simple_target: np.ndarray, initial_phase: np.ndarray) -> None:
    result = run_wgs(simple_target, iterations=5, initial_phase=initial_phase)
    assert result.phase.shape == simple_target.shape
    assert len(result.psnr_history) == 5
    assert np.all(np.isfinite(result.intensity))


def test_hio_high_threshold_matches_gs(
    simple_target: np.ndarray, initial_phase: np.ndarray
) -> None:
    """At threshold=10 (effectively infinite), HIO never triggers feedback
    and should match plain GS to within numerical noise."""
    from ifta import run_ifta

    hio = run_hio(simple_target, iterations=10, beta=0.5, threshold=10.0,
                 warmup=0, initial_phase=initial_phase)
    gs = run_ifta(simple_target, update_rule="GS", iterations=10,
                  initial_phase=initial_phase)
    # Compare final scaled PSNR; should be very close
    assert abs(hio.psnr_history[-1] - gs.psnr_history[-1]) < 0.1
