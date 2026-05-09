"""Benchmark all algorithms across an image directory.

Produces:
    results/summary.csv          per-image, per-rule final PSNR / log-spec
    results/curves.npz           PSNR-vs-iteration curves for plotting
    results/benchmark.png        averaged convergence curves

Usage:
    python scripts/benchmark.py data/ --iterations 20 --output results/
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ifta import run_ifta
from ifta.utils import load_image_grayscale


DEFAULT_RULES = [
    ("GS", {}),
    ("Fienup", {"beta": 0.5}),
    ("Bengtsson", {"beta": 0.35}),
    ("Overdrive", {"beta": 0.5}),
    ("Overdrive", {"beta": 0.7}),
    ("Overdrive", {"beta": 0.9}),
]


def label_for(name: str, kwargs: dict) -> str:
    if not kwargs:
        return name
    parts = [f"{k}={v}" for k, v in kwargs.items()]
    return f"{name}({', '.join(parts)})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for ext in args.extensions for p in args.data_dir.glob(f"*.{ext}")
    )
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        raise SystemExit(f"No images found in {args.data_dir}")

    print(f"Found {len(image_paths)} images in {args.data_dir}")

    # PSNR curves: dict[label] -> ndarray (n_images, iterations)
    curves_psnr: dict[str, list[list[float]]] = {label_for(n, k): [] for n, k in DEFAULT_RULES}
    curves_logspec: dict[str, list[list[float]]] = {label_for(n, k): [] for n, k in DEFAULT_RULES}

    summary_rows: list[dict] = []

    for img_idx, path in enumerate(image_paths):
        try:
            img = load_image_grayscale(path)
        except Exception as exc:
            print(f"  skip {path.name}: {exc}")
            continue

        rng = np.random.default_rng(args.seed + img_idx)
        initial_phase = rng.uniform(0.0, 2.0 * np.pi, size=img.shape)

        print(f"[{img_idx + 1}/{len(image_paths)}] {path.name} {img.shape}")
        for rule_name, rule_kwargs in DEFAULT_RULES:
            label = label_for(rule_name, rule_kwargs)
            t0 = time.perf_counter()
            result = run_ifta(
                img,
                update_rule=rule_name,
                iterations=args.iterations,
                initial_phase=initial_phase,
                **rule_kwargs,
            )
            dt = time.perf_counter() - t0

            curves_psnr[label].append(result.psnr_history)
            curves_logspec[label].append(result.log_spectral_history)

            summary_rows.append(
                {
                    "image": path.name,
                    "rule": label,
                    "final_psnr_db": result.psnr_history[-1] if result.psnr_history else "",
                    "final_log_spec": result.log_spectral_history[-1]
                    if result.log_spectral_history
                    else "",
                    "wall_seconds": f"{dt:.3f}",
                }
            )
            print(f"    {label:<28} PSNR={result.psnr_history[-1]:6.2f} dB  ({dt:.2f}s)")

    # --- Save CSV summary
    summary_path = args.output / "summary.csv"
    with summary_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["image", "rule", "final_psnr_db", "final_log_spec", "wall_seconds"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {summary_path}")

    # --- Save raw curves
    curves_path = args.output / "curves.npz"
    np.savez(
        curves_path,
        **{f"psnr__{k}": np.asarray(v) for k, v in curves_psnr.items()},
        **{f"logspec__{k}": np.asarray(v) for k, v in curves_logspec.items()},
    )
    print(f"Wrote {curves_path}")

    # --- Plot averaged convergence curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    iters = np.arange(1, args.iterations + 1)
    for label, curves in curves_psnr.items():
        if not curves:
            continue
        arr = np.asarray(curves)
        mean = arr.mean(axis=0)
        ax1.plot(iters, mean, label=label, lw=1.5)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("PSNR (dB), averaged")
    ax1.set_title("Reconstruction PSNR")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    for label, curves in curves_logspec.items():
        if not curves:
            continue
        arr = np.asarray(curves)
        mean = arr.mean(axis=0)
        ax2.semilogy(iters, mean, label=label, lw=1.5)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("log-spectral residual (mean of $\\log^2$)")
    ax2.set_title("Lyapunov candidate")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    plot_path = args.output / "benchmark.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
