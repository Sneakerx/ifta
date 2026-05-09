"""Quick end-to-end check on a single image.

Usage:
    python scripts/smoke_test.py path/to/image.bmp [--iterations 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ifta import run_ifta
from ifta.utils import load_image_grayscale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rule",
        type=str,
        default=None,
        help="If set, run only this rule. Otherwise compare GS vs Overdrive.",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    img = load_image_grayscale(args.image)
    print(f"Loaded {args.image.name}: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")

    rng = np.random.default_rng(args.seed)
    initial_phase = rng.uniform(0.0, 2.0 * np.pi, size=img.shape)

    if args.rule:
        rules = [args.rule]
    else:
        rules = ["GS", "Fienup", "Bengtsson", "Overdrive"]

    print(f"\n{'Rule':<14} {'PSNR (dB)':>10} {'log-spec':>10}")
    print("-" * 40)
    for rule in rules:
        result = run_ifta(
            img,
            update_rule=rule,
            iterations=args.iterations,
            initial_phase=initial_phase,
        )
        final_psnr = result.psnr_history[-1] if result.psnr_history else float("nan")
        final_logspec = (
            result.log_spectral_history[-1] if result.log_spectral_history else float("nan")
        )
        print(f"{rule:<14} {final_psnr:>10.2f} {final_logspec:>10.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
