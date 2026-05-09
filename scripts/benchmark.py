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
from ifta.baselines import run_hio, run_overdrive2, run_raar, run_wgs
from ifta.utils import load_image_grayscale


# Each entry: (display_name, runner, kwargs)
# Modern baselines (HIO, RAAR, WGS) use standalone runners because they
# operate on complex iterates rather than amplitudes only.
DEFAULT_RULES = [
    ("GS",                {"runner": "ifta", "update_rule": "GS"}),
    ("Fienup(beta=0.5)",  {"runner": "ifta", "update_rule": "Fienup",    "beta": 0.5}),
    ("Bengtsson(beta=0.35)", {"runner": "ifta", "update_rule": "Bengtsson", "beta": 0.35}),
    ("HIO(beta=0.5,thr=0.3)", {"runner": "hio",  "beta": 0.5, "threshold": 0.3}),
    ("HIO(beta=0.9,thr=0.3)", {"runner": "hio",  "beta": 0.9, "threshold": 0.3}),
    ("RAAR(beta=0.5)",    {"runner": "raar", "beta": 0.5}),
    ("WGS",               {"runner": "wgs"}),
    ("Overdrive(beta=0.5)", {"runner": "ifta", "update_rule": "Overdrive", "beta": 0.5}),
    ("Overdrive(beta=0.7)", {"runner": "ifta", "update_rule": "Overdrive", "beta": 0.7}),
    ("Overdrive(beta=0.9)", {"runner": "ifta", "update_rule": "Overdrive", "beta": 0.9}),
    ("Overdrive2(a=1.0,b=-0.2)", {"runner": "od2", "alpha": 1.0, "beta2": -0.2}),
]


_RUNNER_DISPATCH = {
    "ifta": run_ifta,
    "hio":  run_hio,
    "raar": run_raar,
    "wgs":  run_wgs,
    "od2":  run_overdrive2,
}


def psnr_unscaled(target_intensity: np.ndarray, current_intensity: np.ndarray) -> float:
    """PSNR without per-iteration power rescale.

    The reconstructed intensity is taken as is (laser power was matched to the
    target via Parseval at run start, so this only fails if the algorithm
    drifts). Peak = max of target. Reflects what a real holographic projector
    would deliver, since output power cannot be freely rescaled at runtime.
    """
    peak = max(float(target_intensity.max()), 1e-12)
    err = (target_intensity - current_intensity).astype(np.float64)
    mse_val = float(np.mean(err ** 2))
    if mse_val <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(peak ** 2 / mse_val))


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
    parser.add_argument(
        "--no-renormalize",
        action="store_true",
        help="Disable the global Parseval rescale in update rules. "
             "Use to run the algorithm exactly as analyzed in the paper.",
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

    # PSNR curves: dict[label] -> list of per-iter histories
    curves_psnr: dict[str, list[list[float]]] = {label: [] for label, _ in DEFAULT_RULES}
    curves_logspec: dict[str, list[list[float]]] = {label: [] for label, _ in DEFAULT_RULES}

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
        for label, rule_kwargs in DEFAULT_RULES:
            kw = dict(rule_kwargs)
            runner_key = kw.pop("runner")
            runner = _RUNNER_DISPATCH[runner_key]
            if args.no_renormalize and runner_key == "ifta":
                kw["renormalize"] = False

            t0 = time.perf_counter()
            result = runner(
                img,
                iterations=args.iterations,
                initial_phase=initial_phase,
                **kw,
            )
            dt = time.perf_counter() - t0

            psnr_unsc = psnr_unscaled(img.astype(np.float64), result.intensity)

            curves_psnr[label].append(result.psnr_history)
            curves_logspec[label].append(result.log_spectral_history)

            summary_rows.append(
                {
                    "image": path.name,
                    "rule": label,
                    "final_psnr_db": result.psnr_history[-1] if result.psnr_history else "",
                    "final_psnr_unscaled_db": f"{psnr_unsc:.4f}",
                    "final_log_spec": result.log_spectral_history[-1]
                    if result.log_spectral_history
                    else "",
                    "wall_seconds": f"{dt:.3f}",
                }
            )
            print(f"    {label:<28} PSNR={result.psnr_history[-1]:6.2f} dB "
                  f"(unscaled={psnr_unsc:6.2f} dB) ({dt:.2f}s)")

    # --- Save CSV summary
    summary_path = args.output / "summary.csv"
    with summary_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "image",
                "rule",
                "final_psnr_db",
                "final_psnr_unscaled_db",
                "final_log_spec",
                "wall_seconds",
            ],
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
