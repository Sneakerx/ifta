"""Empirical 2-parameter sweep of 2nd-order Overdrive.

Rule:  |G'_k| = |G'_{k-1}|^α · |G'_{k-2}|^β · |F|^(2-α-β) / |G_k|

The η-whiteness experiment showed the AR(1)-with-white-η model is
empirically misspecified, so the 2-parameter family is purely heuristic;
we tune (α, β) by grid search on a held-out training subset and report
performance on the disjoint test subset.

Outputs:
    results/overdrive2_sweep.csv         per-image, per-(α, β) PSNR
    results/overdrive2_sweep.png         heatmap of mean PSNR over train set
    results/overdrive2_summary.txt       train/test summary
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ifta.algorithms import run_ifta
from ifta.baselines import run_overdrive2
from ifta.utils import load_image_grayscale


def main(data_dir: Path, output_dir: Path, *, train_ratio: float = 0.345,
         iterations: int = 20, warmup: int = 3, seed: int = 42) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(data_dir.glob("*.bmp"))
    if not paths:
        raise SystemExit(f"No .bmp images found in {data_dir}")

    n = len(paths)
    n_train = max(1, int(round(train_ratio * n)))
    train_paths = paths[:n_train]
    test_paths = paths[n_train:]
    print(f"Train: {n_train} images, Test: {len(test_paths)} images")

    # Grid: alpha in [0.5, 1.2] step 0.1; beta in [-0.4, 0.2] step 0.1
    alphas = np.round(np.arange(0.5, 1.21, 0.1), 2)
    betas = np.round(np.arange(-0.4, 0.21, 0.1), 2)
    print(f"Grid: alpha {list(alphas)}, beta {list(betas)}")
    print(f"Total configs: {len(alphas) * len(betas)}")

    # Cache image array + initial phase
    train_imgs = [(p, load_image_grayscale(p)) for p in train_paths]
    test_imgs = [(p, load_image_grayscale(p)) for p in test_paths]

    def initial_phase_for(img_idx, shape):
        return np.random.default_rng(seed + img_idx).uniform(0, 2 * np.pi, shape)

    # --- Train sweep
    print("\n--- TRAIN SWEEP ---")
    train_grid = np.zeros((len(alphas), len(betas)))
    rows: list[dict] = []
    for ia, alpha in enumerate(alphas):
        for ib, beta2 in enumerate(betas):
            psnrs = []
            for img_idx, (p, img) in enumerate(train_imgs):
                phi0 = initial_phase_for(img_idx, img.shape)
                r = run_overdrive2(img, alpha=float(alpha), beta2=float(beta2),
                                  iterations=iterations, warmup=warmup,
                                  initial_phase=phi0)
                psnrs.append(r.psnr_history[-1])
                rows.append({
                    "split": "train",
                    "image": p.name,
                    "alpha": float(alpha),
                    "beta2": float(beta2),
                    "psnr_db": r.psnr_history[-1],
                })
            train_grid[ia, ib] = float(np.mean(psnrs))
        print(f"  alpha={alpha:.1f}: best beta on train = "
              f"{betas[np.argmax(train_grid[ia])]:.1f} -> "
              f"PSNR = {train_grid[ia].max():.3f} dB")

    # Best on train
    ia_star, ib_star = np.unravel_index(np.argmax(train_grid), train_grid.shape)
    alpha_star = float(alphas[ia_star])
    beta_star = float(betas[ib_star])
    train_best = train_grid[ia_star, ib_star]
    print(f"\nBest on train: alpha={alpha_star}, beta={beta_star} -> "
          f"PSNR = {train_best:.3f} dB")

    # --- Test set: evaluate best (α, β) and a few baselines
    print("\n--- TEST SET ---")
    summary_lines = []

    def eval_on_test(label, fn):
        psnrs = []
        for img_idx, (p, img) in enumerate(test_imgs):
            phi0 = initial_phase_for(img_idx + n_train, img.shape)
            r = fn(img, phi0)
            psnrs.append(r.psnr_history[-1])
            rows.append({
                "split": "test",
                "image": p.name,
                "alpha": label,
                "beta2": "",
                "psnr_db": r.psnr_history[-1],
            })
        mean = float(np.mean(psnrs))
        std = float(np.std(psnrs, ddof=1))
        line = f"  {label:<32}  mean = {mean:6.3f} dB  std = {std:5.3f}"
        print(line)
        summary_lines.append(line)
        return mean, std

    overdrive_baseline = lambda img, phi0: run_ifta(
        img, update_rule="Overdrive", beta=0.7, warmup=warmup,
        iterations=iterations, initial_phase=phi0,
    )
    overdrive2_best = lambda img, phi0: run_overdrive2(
        img, alpha=alpha_star, beta2=beta_star, warmup=warmup,
        iterations=iterations, initial_phase=phi0,
    )

    od_mean, od_std = eval_on_test("Overdrive(beta=0.7)", overdrive_baseline)
    od2_mean, od2_std = eval_on_test(
        f"Overdrive2(a={alpha_star},b={beta_star})", overdrive2_best
    )
    delta = od2_mean - od_mean
    summary_lines.append(f"\n  delta (test) = {delta:+.4f} dB")
    print(f"\n  delta (test) = {delta:+.4f} dB")

    # --- Save
    csv_path = output_dir / "overdrive2_sweep.csv"
    with csv_path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["split", "image", "alpha", "beta2", "psnr_db"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    summary_path = output_dir / "overdrive2_summary.txt"
    with summary_path.open("w") as fp:
        fp.write(f"Train images: {n_train}, Test images: {len(test_paths)}\n")
        fp.write(f"Best (alpha, beta) on train: ({alpha_star}, {beta_star})\n")
        fp.write(f"Train mean PSNR: {train_best:.3f} dB\n\n")
        fp.write("TEST SET\n")
        fp.write("\n".join(summary_lines))
        fp.write("\n")
    print(f"Wrote {summary_path}")

    # --- Heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(train_grid, origin="lower", aspect="auto",
                   extent=[float(betas[0]) - 0.05, float(betas[-1]) + 0.05,
                           float(alphas[0]) - 0.05, float(alphas[-1]) + 0.05],
                   cmap="viridis")
    ax.set_xlabel("$\\beta_2$ (weight on $|G'_{k-2}|$)")
    ax.set_ylabel("$\\alpha$ (weight on $|G'_{k-1}|$)")
    ax.set_title(f"Train mean PSNR (n={n_train}), best: $\\alpha$={alpha_star}, $\\beta$={beta_star}")
    fig.colorbar(im, ax=ax, label="mean PSNR (dB)")
    # Mark the 1st-order Overdrive(0.7) point
    ax.plot([0.0], [0.7], "o", color="white", ms=10, mec="black",
            label="Overdrive(0.7) = (0.7, 0)")
    ax.plot([beta_star], [alpha_star], "*", color="red", ms=18, mec="black",
            label=f"best: ({alpha_star}, {beta_star})")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    plot_path = output_dir / "overdrive2_sweep.png"
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")

    return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_dir", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/overdrive2"))
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.345,
                   help="Fraction of images for tuning (0.345 = 10/29)")
    args = p.parse_args()
    raise SystemExit(main(args.data_dir, args.output,
                          train_ratio=args.train_ratio,
                          iterations=args.iterations,
                          warmup=args.warmup, seed=args.seed))
