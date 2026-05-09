"""Validate `estimate_contraction` against synthetic AR(1) residual histories.

Model:
    u_k = (beta - c) * u_{k-1} - eta_k,   eta_k ~ N(0, sigma^2)
    h_k = mean of v_k^2 ≈ V_inf + alpha * (beta - c)^{2k}

We test two regimes:
    (a) "transient": fit window = first n_fit iterations after warmup, when
        h_k still contains decay information.
    (b) "steady-state": fit window = trailing n_fit iterations; this is
        the regime the estimator was originally designed for, and where it
        fundamentally cannot recover the pole if the trajectory has already
        reached the floor.

For each (beta, c, sigma) we run multiple trials and report MAE in c-hat.

Outputs:
    results/estimator_validation.csv
    results/estimator_validation.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ifta.adaptive import estimate_contraction, estimate_contraction_legacy


def synth_history(
    beta: float,
    c: float,
    *,
    sigma: float = 0.05,
    n_iter: int = 30,
    n_pixels: int = 4096,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a synthetic h_k = mean v_k^2 trajectory under AR(1) model."""
    pole = beta - c
    u = rng.normal(0.0, 1.0, size=n_pixels)
    h = np.zeros(n_iter)
    for k in range(n_iter):
        eta = rng.normal(0.0, sigma, size=n_pixels)
        u_next = pole * u - eta
        v = c * u + eta
        h[k] = float(np.mean(v ** 2))
        u = u_next
    return h


def run_grid(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=2026)

    betas = [0.3, 0.5, 0.7, 0.9]
    cs = np.linspace(0.1, 0.9, 9)
    sigmas = [0.02, 0.05]
    n_trials = 25
    n_iter = 30
    n_fit = 8

    rows: list[dict] = []
    for beta in betas:
        for c_true in cs:
            for sigma in sigmas:
                err_new_trans = []
                err_old_trans = []
                err_new_steady = []
                err_old_steady = []
                for _ in range(n_trials):
                    h = synth_history(beta, float(c_true), sigma=sigma,
                                      n_iter=n_iter, rng=rng)
                    h_trans = h[:n_fit]
                    h_steady = h[-n_fit:]
                    c_new_t = estimate_contraction(h_trans, beta_used=beta,
                                                   window=n_fit)
                    c_old_t = estimate_contraction_legacy(h_trans, window=n_fit)
                    c_new_s = estimate_contraction(h_steady, beta_used=beta,
                                                   window=n_fit)
                    c_old_s = estimate_contraction_legacy(h_steady, window=n_fit)
                    err_new_trans.append(c_new_t - c_true)
                    err_old_trans.append(c_old_t - c_true)
                    err_new_steady.append(c_new_s - c_true)
                    err_old_steady.append(c_old_s - c_true)
                rows.append(
                    {
                        "beta": beta,
                        "c_true": float(c_true),
                        "sigma": sigma,
                        "mae_new_transient": float(np.mean(np.abs(err_new_trans))),
                        "mae_old_transient": float(np.mean(np.abs(err_old_trans))),
                        "mae_new_steady": float(np.mean(np.abs(err_new_steady))),
                        "mae_old_steady": float(np.mean(np.abs(err_old_steady))),
                        "bias_new_transient": float(np.mean(err_new_trans)),
                        "bias_old_transient": float(np.mean(err_old_trans)),
                    }
                )

    csv_path = out_dir / "estimator_validation.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    # Two-panel plot per beta: (left) transient window, (right) steady-state.
    fig, axes = plt.subplots(len(betas), 2, figsize=(9, 10), sharex=True, sharey=True)
    sigma_plot = sigmas[0]
    for i, beta in enumerate(betas):
        sel = [r for r in rows if r["beta"] == beta and r["sigma"] == sigma_plot]
        cs_true = np.array([r["c_true"] for r in sel])
        ax_t, ax_s = axes[i, 0], axes[i, 1]
        ax_t.plot(cs_true, [r["mae_new_transient"] for r in sel],
                  "o-", label="new", color="C0")
        ax_t.plot(cs_true, [r["mae_old_transient"] for r in sel],
                  "s--", label="legacy", color="C3")
        ax_s.plot(cs_true, [r["mae_new_steady"] for r in sel],
                  "o-", label="new", color="C0")
        ax_s.plot(cs_true, [r["mae_old_steady"] for r in sel],
                  "s--", label="legacy", color="C3")
        for ax in (ax_t, ax_s):
            ax.axvline(beta, color="k", lw=0.5, alpha=0.4,
                       label=f"$\\beta$={beta}" if i == 0 else None)
            ax.grid(True, alpha=0.3)
        ax_t.set_ylabel(f"$\\beta$={beta}\nMAE in $\\hat c$")
        if i == 0:
            ax_t.set_title("transient window (iters 1–8)")
            ax_s.set_title("steady-state window (last 8)")
            ax_t.legend(fontsize=8, loc="upper left")
        if i == len(betas) - 1:
            ax_t.set_xlabel("true $c$")
            ax_s.set_xlabel("true $c$")
    fig.suptitle(
        f"Adaptive estimator validation, $\\sigma$={sigma_plot}, AR(1) model",
        y=1.0,
    )
    fig.tight_layout()
    plot_path = out_dir / "estimator_validation.png"
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    run_grid(Path("results"))
