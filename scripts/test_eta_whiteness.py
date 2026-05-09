"""Test whether the noise process eta_k in the closed-loop Overdrive
recursion is white.

Model (paper §4-5):
    u_k = (beta - c) * u_{k-1} - eta_k                    (AR(1) with white eta)

where  u_k = log(|G_k'|/|F|)  is the *post-update* (virtual) residual.
The pre-update residual v_k = log(|G_k|/|F|) follows ARMA(1,1) even when
eta is white, so we must fit u_k, not v_k.

If eta is white, 1st-order Overdrive at beta=c is already optimal in the
mean-squared sense. If eta is *colored* (autocorrelation at lag 1 or higher),
a higher-order IIR rule can lower the synthesis floor by whitening the
innovation. This script measures the lag-1..5 autocorrelation of the
empirical eta to decide which regime we are in.

Procedure:
    1. Run Overdrive(beta=0.7, renormalize=False) for n_iter iterations
       on each Klasse-1 image, capturing both v_k = log(|G_k|/|F|) and
       u_k = log(|G_k'|/|F|) fields per iteration (mask: target > eps).
    2. Per pixel, regress u_k on u_{k-1} over the transient window
       [k_start, k_end). Slope = beta - c, intercept absorbs u*.
    3. Innovations eta_hat_k = u_k - slope*u_{k-1} - intercept.
    4. Per-pixel lag-l autocorrelation of eta_hat for l=1..5.
    5. Aggregate over pixels and images. Report mean/median/IQR.

For comparison we also fit AR(1) to v_k. Under the AR(1)-with-white-eta
model the v-residuals MUST be colored (lag-1 ACF should be near -beta/(1+beta^2)
≈ -0.47 for beta=0.7); this is a sanity check on the methodology.

Outputs:
    results/eta_whiteness.csv   per-image per-lag mean ACF
    results/eta_whiteness.png   ACF histogram + per-lag bar chart
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ifta.algorithms import _normalize_target
from ifta.utils import load_image_grayscale


EPS = 1e-12


def run_overdrive_capture(image, *, beta=0.7, iterations=50, warmup=3, seed=42):
    """Run Overdrive without renormalize, capturing both
        v_k = log(|G_k|/|F|)   (pre-update, ARMA(1,1) under white-eta model)
        u_k = log(|G_k'|/|F|)  (post-update, AR(1) under white-eta model)
    fields at every iteration. Returns (v_history, u_history, mask)."""
    target_intensity = _normalize_target(image)
    target_intensity = np.fft.fftshift(target_intensity)
    target_amp = np.sqrt(target_intensity)
    laser_amp_scalar = float(np.sqrt(target_intensity.mean()))
    laser_amp = np.full_like(target_intensity, laser_amp_scalar)

    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=target_intensity.shape)
    grating = laser_amp * np.exp(1j * phi)
    prev_virtual = target_amp.copy()

    mask = target_amp > EPS
    log_target = np.log(np.maximum(target_amp, EPS))

    v_history = np.zeros((iterations, *target_amp.shape), dtype=np.float32)
    u_history = np.zeros((iterations, *target_amp.shape), dtype=np.float32)

    for k in range(iterations):
        U = np.fft.fft2(grating, norm="ortho")
        current_amp = np.abs(U)
        v_k = np.zeros_like(target_amp)
        v_k[mask] = np.log(np.maximum(current_amp[mask], EPS)) - log_target[mask]
        v_history[k] = v_k.astype(np.float32)

        if k < warmup:
            new_amp = target_amp.copy()
        else:
            safe = np.where(current_amp > EPS, current_amp, EPS)
            new_amp = (
                np.power(np.maximum(prev_virtual, EPS), beta)
                * np.power(np.maximum(target_amp, EPS), 2.0 - beta)
                / safe
            )
            new_amp = np.where(target_amp > 0, new_amp, 0.0)
            # NB: deliberately no _renormalize_amplitude — the analysis
            # in §3-§4 omits it, and we want the noise we measure to be
            # exactly the noise in the analyzed iteration.

        u_k = np.zeros_like(target_amp)
        u_k[mask] = np.log(np.maximum(new_amp[mask], EPS)) - log_target[mask]
        u_history[k] = u_k.astype(np.float32)

        U_new = new_amp * np.exp(1j * np.angle(U))
        u_new = np.fft.ifft2(U_new, norm="ortho")
        phi = np.mod(np.angle(u_new), 2.0 * np.pi)
        grating = laser_amp * np.exp(1j * phi)
        prev_virtual = new_amp

    return v_history, u_history, mask


def fit_ar1_and_innovations(v_history, mask, k_start, k_end):
    """Per-pixel AR(1) fit  v_k = a*v_{k-1} + b + eta_k  on iterations
    [k_start, k_end). Returns (a_field, b_field, innovations) where
    innovations has shape (k_end-k_start, H, W)."""
    v_t = v_history[k_start:k_end].astype(np.float64)            # (T, H, W)
    v_lag = v_history[k_start - 1:k_end - 1].astype(np.float64)  # (T, H, W)

    mean_t = v_t.mean(axis=0)
    mean_lag = v_lag.mean(axis=0)
    centered_t = v_t - mean_t
    centered_lag = v_lag - mean_lag

    cov = (centered_t * centered_lag).mean(axis=0)
    var_lag = (centered_lag ** 2).mean(axis=0)

    a_field = np.zeros_like(mean_t)
    valid = mask & (var_lag > EPS)
    a_field[valid] = cov[valid] / var_lag[valid]
    b_field = np.where(valid, mean_t - a_field * mean_lag, 0.0)

    innovations = v_t - (a_field[None] * v_lag + b_field[None])
    return a_field, b_field, innovations, valid


def aggregate_lsd_decay(v_history, mask, k_start, k_end):
    """Compute per-iteration LSD h_k = mean(v_k^2) over the support, fit
    geometric decay h_k - h_floor = A * rho^k via log-linear regression on
    a window. Returns rho, h_floor estimate, and AR(1) residuals on the
    *scalar* h_k trajectory."""
    H, W = mask.shape
    n_iter = v_history.shape[0]
    n_valid = int(mask.sum())
    h_k = np.zeros(n_iter)
    for k in range(n_iter):
        h_k[k] = float((v_history[k][mask] ** 2).sum() / max(n_valid, 1))
    # Floor estimate: median of last 10 iterations
    h_floor = float(np.median(h_k[-10:]))
    delta = np.maximum(h_k - h_floor, EPS)
    k_arr = np.arange(k_start, min(k_end, n_iter), dtype=np.float64)
    log_delta = np.log(delta[k_start:k_end])
    if len(log_delta) < 4 or not np.isfinite(log_delta).all():
        return float("nan"), h_floor, h_k
    slope, _ = np.polyfit(k_arr, log_delta, 1)
    rho = float(np.exp(slope))
    return rho, h_floor, h_k


def per_pixel_lag_acf(innovations, valid, max_lag=5):
    """Compute lag-l per-pixel autocorrelation of innovations for l=1..max_lag.
    Returns dict lag -> (H, W) field, masked invalid pixels to NaN."""
    T = innovations.shape[0]
    out = {}
    for lag in range(1, max_lag + 1):
        et = innovations[lag:]
        el = innovations[:T - lag]
        mt = et.mean(axis=0); ml = el.mean(axis=0)
        ct = et - mt;          cl = el - ml
        cov = (ct * cl).mean(axis=0)
        var_t = (ct ** 2).mean(axis=0)
        var_l = (cl ** 2).mean(axis=0)
        denom = np.sqrt(var_t * var_l)
        rho = np.full_like(cov, np.nan)
        ok = valid & (denom > EPS)
        rho[ok] = cov[ok] / denom[ok]
        out[lag] = rho
    return out


def main(data_dir: Path, output_dir: Path, *, max_images: int = 10,
         beta: float = 0.7, n_iter: int = 50, k_start: int = 15,
         k_end: int = 40, max_lag: int = 5) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(str(data_dir / "*.bmp")))[:max_images]
    if not paths:
        raise SystemExit(f"No .bmp images found in {data_dir}")
    print(f"Whiteness test on {len(paths)} images, beta={beta}, iters={n_iter}, "
          f"window=[{k_start},{k_end})")

    rows = []
    samples_by_lag_u: dict[int, list[np.ndarray]] = {l: [] for l in range(1, max_lag + 1)}
    samples_by_lag_v: dict[int, list[np.ndarray]] = {l: [] for l in range(1, max_lag + 1)}

    for i, path in enumerate(paths):
        img = load_image_grayscale(path)
        v_history, u_history, mask = run_overdrive_capture(
            img, beta=beta, iterations=n_iter, seed=42 + i
        )

        # PRIMARY TEST: AR(1) on u (the post-update residual).
        # Under the white-eta model, residuals here ARE eta, so they should be white.
        au, _, innov_u, valid_u = fit_ar1_and_innovations(
            u_history, mask, k_start, k_end
        )
        acf_u = per_pixel_lag_acf(innov_u, valid_u, max_lag=max_lag)

        # SANITY CHECK: AR(1) on v. Under the white-eta model v is ARMA(1,1),
        # so AR(1) residuals here MUST be colored with lag-1 ACF ~ -beta/(1+beta^2).
        av, _, innov_v, valid_v = fit_ar1_and_innovations(
            v_history, mask, k_start, k_end
        )
        acf_v = per_pixel_lag_acf(innov_v, valid_v, max_lag=max_lag)

        au_vals = au[valid_u]
        au_med = float(np.median(au_vals))
        c_est = beta - au_med   # AR(1) slope on u  ==  beta - c

        # GLOBAL aggregate test on h_k = mean v_k^2 trajectory
        rho_lsd, h_floor, h_traj = aggregate_lsd_decay(
            v_history, mask, k_start, k_end
        )
        # Implied |beta - c| = sqrt(rho_lsd) since h_k decays as (beta-c)^(2k)
        impl_abs_diff = float(np.sqrt(max(rho_lsd, EPS)))

        row = {
            "image": Path(path).name,
            "n_valid": int(valid_u.sum()),
            "ar1_slope_u_median": au_med,
            "c_estimate_perpixel": c_est,
            "lsd_decay_rho": rho_lsd,
            "implied_abs_beta_minus_c": impl_abs_diff,
            "h_floor": h_floor,
            "ar1_slope_v_median": float(np.median(av[valid_v])),
        }
        for lag in range(1, max_lag + 1):
            vals_u = acf_u[lag][valid_u]; vals_u = vals_u[np.isfinite(vals_u)]
            vals_v = acf_v[lag][valid_v]; vals_v = vals_v[np.isfinite(vals_v)]
            samples_by_lag_u[lag].append(vals_u)
            samples_by_lag_v[lag].append(vals_v)
            row[f"acf_u_lag{lag}_median"] = float(np.median(vals_u))
            row[f"acf_v_lag{lag}_median"] = float(np.median(vals_v))
        rows.append(row)

        print(f"  [{i+1:2d}/{len(paths)}] {Path(path).name:<24} "
              f"slope_u={au_med:+.3f} c~{c_est:+.2f}  "
              f"|β-c|≈{impl_abs_diff:.3f} (LSD)  "
              f"u-ACF1={row['acf_u_lag1_median']:+.3f} "
              f"v-ACF1={row['acf_v_lag1_median']:+.3f}")

    # Save CSV
    csv_path = output_dir / "eta_whiteness.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    # Aggregate plot: 2 panels, primary (u) and sanity (v)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Predicted v-AR(1)-residual lag-1 ACF under white-eta model is
    #   -beta / (1 + beta^2)
    v_pred_lag1 = -beta / (1 + beta ** 2)

    lags = list(range(1, max_lag + 1))
    for samples, ax, label, color in [
        (samples_by_lag_u, axes[0], "primary: u-AR(1) residuals (= $\\hat\\eta$)", "C0"),
        (samples_by_lag_v, axes[1], "sanity: v-AR(1) residuals", "C2"),
    ]:
        meds, q25s, q75s = [], [], []
        for lag in lags:
            vals = np.concatenate(samples[lag])
            meds.append(float(np.median(vals)))
            q25s.append(float(np.percentile(vals, 25)))
            q75s.append(float(np.percentile(vals, 75)))
        meds = np.array(meds)
        err = [meds - np.array(q25s), np.array(q75s) - meds]
        bar_colors = [color if abs(m) < 0.05 else "C3" for m in meds]
        ax.bar(lags, meds, yerr=err, capsize=4, color=bar_colors, alpha=0.85)
        ax.axhline(0.0, color="k", lw=0.7)
        ax.axhspan(-0.05, 0.05, color="k", alpha=0.07,
                   label="white threshold ±0.05")
        ax.set_xlabel("lag")
        ax.set_ylabel("ACF (median ± IQR over pixels)")
        ax.set_title(label, fontsize=10)
        ax.set_xticks(lags)
        ax.legend(fontsize=8, loc="upper right")
    # Overlay the predicted v-AR(1) lag-1 on the sanity panel
    axes[1].axhline(v_pred_lag1, color="C1", lw=1.5, ls="--",
                    label=f"predicted lag-1 = $-\\beta/(1+\\beta^2)$ = {v_pred_lag1:+.3f}")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Innovation autocorrelation, n={len(paths)} images, $\\beta$={beta}",
                 y=1.02)
    fig.tight_layout()
    plot_path = output_dir / "eta_whiteness.png"
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # Verdict
    u_lag1 = float(np.median(np.concatenate(samples_by_lag_u[1])))
    v_lag1 = float(np.median(np.concatenate(samples_by_lag_v[1])))
    median_au = float(np.median([r["ar1_slope_u_median"] for r in rows]))
    median_c = float(np.median([r["c_estimate_perpixel"] for r in rows]))
    median_implied = float(np.median([r["implied_abs_beta_minus_c"] for r in rows]))
    print()
    print("=" * 72)
    print("DIAGNOSTICS")
    print("=" * 72)
    print(f"  per-pixel u-AR(1) slope (median across images) : {median_au:+.4f}")
    print(f"  implied c estimate (= beta - slope)            : {median_c:+.4f}")
    print(f"  global |beta - c| from LSD geometric decay     : {median_implied:+.4f}")
    print(f"  u-AR(1) residual lag-1 ACF  (white if ~0)      : {u_lag1:+.4f}")
    print(f"  v-AR(1) residual lag-1 ACF                     : {v_lag1:+.4f}")
    print(f"     predicted under white-eta + AR(1) model     : {v_pred_lag1:+.4f}")
    print()

    sanity_ok = abs(v_lag1 - v_pred_lag1) < 0.15
    c_in_range = 0.0 <= median_c <= 1.0
    eta_white = abs(u_lag1) < 0.10

    print("INTERPRETATION")
    print("-" * 72)
    if not c_in_range:
        print(f"  [FAIL] Implied c = {median_c:+.3f} is outside [0, 1] required by")
        print(f"         the GS-firmly-nonexpansive bound.  The scalar-c")
        print(f"         linearization in the relevant log-amplitude metric is")
        print(f"         empirically false.")
    if not sanity_ok:
        print(f"  [FAIL] v-residual lag-1 ACF differs from white-eta prediction by")
        print(f"         {abs(v_lag1 - v_pred_lag1):+.3f}.  Even granting a fitted c,")
        print(f"         the AR(1)-with-white-eta model fails its own consistency test.")
    if not eta_white:
        print(f"  [FAIL] u-AR(1) residuals (= eta_hat) have lag-1 ACF {u_lag1:+.3f},")
        print(f"         far from the 0 expected for white eta.")

    if c_in_range and sanity_ok and eta_white:
        verdict = ("MODEL CONSISTENT and ETA WHITE.  1st-order Overdrive is at "
                   "the noise floor; higher-order IIR will not help.")
    elif c_in_range and sanity_ok and not eta_white:
        verdict = ("MODEL CONSISTENT but ETA COLORED.  2nd-order or ARMA "
                   "Overdrive has real gain potential via innovation whitening.")
    else:
        verdict = ("MODEL MISSPECIFIED.  Per-pixel scalar AR(1) does not capture "
                   "the FFT-coupled dynamics.  An IIR-design improvement requires "
                   "either (a) operator-spectral framing of the GS Jacobian, or "
                   "(b) empirical 2nd-order tuning without claiming a closed-form "
                   "optimum.  Symbolic 2nd-order Overdrive will exhibit the same "
                   "structural problem unless reformulated multivariately.")
    print()
    print("VERDICT")
    print("-" * 72)
    print(f"  {verdict}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--k-start", type=int, default=15)
    parser.add_argument("--k-end", type=int, default=40)
    parser.add_argument("--max-lag", type=int, default=5)
    args = parser.parse_args()
    raise SystemExit(main(
        args.data_dir, args.output, max_images=args.max_images, beta=args.beta,
        n_iter=args.iterations, k_start=args.k_start, k_end=args.k_end,
        max_lag=args.max_lag,
    ))
