"""
fft_compare_variants.py
=======================

Train every v2 variant back-to-back and produce a side-by-side comparison
table + bar plots of test-set metrics.  Reuses the same train/val/test
split for all four variants so numbers are directly comparable.

What this compares
------------------
- baseline        : v1 recipe (pure FFT-MSE, linear period head).  Sanity.
- phase           : + phase-aware time-domain loss, + softplus period head.
- residual        : + cubic-spline speed prior, + softplus period head.
- phase_residual  : all three improvements combined.

Outputs (under ``kfold_results/review_v2/comparison/``)
-------------------------------------------------------
- ``variants_summary.csv``           one row per variant, all metrics
- ``variants_time_mse.png``          bar chart of time-domain MSE
- ``variants_coef_mse.png``          bar chart of FFT-coef MSE
- ``variants_period_mae.png``        bar chart of period MAE (seconds)
- ``variants_trajectory_overlay.png`` overlay of pred vs GT for 4 test
                                     samples per variant

Run
---
    python fft_compare_variants.py                # full run (all 4 variants)
    python fft_compare_variants.py --quick        # smoke run
    python fft_compare_variants.py --skip-train   # just re-plot existing metrics

Note: training 4 variants on CPU will take roughly 4x the wall time of
a single v1 run.  Use ``--quick`` during development.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import re-usable pieces from v2 (so we don't duplicate code)
from fft_datacreate_review_v2 import (
    DATA_DIR, RESULTS_ROOT, REVIEW_DIR,
    INPUT_NPY, OUTPUT_NPY, PERIOD_NPY,
    MEAN_TRAIN_NPY, STD_TRAIN_NPY, PERIOD_STATS_NPY, SPLINE_PRIOR_NPZ,
    SimpleFCNN, SplinePrior, FREQ_DIM, DEVICE,
    variant_config,
)


VARIANTS = ("baseline", "phase", "residual", "phase_residual")
OUT_DIR  = Path(REVIEW_DIR) / "comparison"


# ---------------------------------------------------------------------------
# Training loop: shell out to the v2 script so each variant gets a clean
# Python process (avoids any hidden global-state carryover between runs).
# ---------------------------------------------------------------------------
def train_variant(variant: str, quick: bool, no_aug: bool) -> int:
    cmd = [sys.executable, "fft_datacreate_review_v2.py", "--variant", variant]
    if quick:  cmd.append("--quick")
    if no_aug: cmd.append("--no-aug")
    print(f"\n{'='*70}\n[train] {variant}\n{'='*70}")
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------
def load_variant_metrics(variant: str) -> dict:
    """Load test_metrics_<variant>.json produced by fft_datacreate_review_v2.py."""
    path = Path(REVIEW_DIR) / f"test_metrics_{variant}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run training for '{variant}' first.")
    with path.open() as f:
        return json.load(f)


def build_summary_table(variants: tuple[str, ...]) -> list[dict]:
    rows = []
    for v in variants:
        m = load_variant_metrics(v)
        rows.append(dict(
            variant=v,
            coef_mse=m["coef_mse"],
            time_domain_mse=m["time_domain_mse"],
            fft_fidelity_h1_h3=m["fft_fidelity_h1_h3"],
            period_mae_s=m["period_mae_s"],
            period_mape=m["period_mape"],
            n_period_nonpositive=m.get("n_period_nonpositive", -1),
            best_val=m["best_hp"]["best_val"],
            best_epoch=m["best_hp"]["best_epoch"],
        ))
    return rows


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    keys = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"[csv]  wrote {out_path}")


def print_leaderboard(rows: list[dict]) -> None:
    keys = [("variant", "<16s", "s"),
            ("coef_mse",          ">10s", ".5f"),
            ("time_domain_mse",   ">12s", ".5f"),
            ("fft_fidelity_h1_h3",">12s", ".4f"),
            ("period_mae_s",      ">12s", ".4f"),
            ("n_period_nonpositive", ">10s", "d")]
    print("\n[leaderboard] lower is better (except variant column)")
    hdr = "  ".join(f"{k[0]:{k[1]}}" for k in keys)
    print("  " + hdr)
    for r in rows:
        line = []
        for k, sw, fw in keys:
            v = r[k]
            if fw == "s":
                line.append(f"{v:{sw}}")
            elif fw == "d":
                line.append(f"{int(v):{sw[:-1]}d}")
            else:
                line.append(f"{v:{sw[:-1]}{fw}}")
        print("  " + "  ".join(line))


# ---------------------------------------------------------------------------
# Bar plots
# ---------------------------------------------------------------------------
def plot_bar(rows, key: str, ylabel: str, title: str, out_path: Path,
             yscale: str = "linear") -> None:
    variants = [r["variant"] for r in rows]
    vals     = [r[key] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(variants, vals, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"])
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_yscale(yscale)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=15)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# Trajectory overlay (pred vs GT) on a few test samples
# ---------------------------------------------------------------------------
def load_variant_model(variant: str, hidden_size: int = 512) -> SimpleFCNN:
    path = Path(RESULTS_ROOT) / f"FINAL_BEST_MODEL_V2_{variant}.pth"
    model = SimpleFCNN(hidden_size=hidden_size)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model


def load_spline_prior() -> SplinePrior:
    d = np.load(SPLINE_PRIOR_NPZ)
    return SplinePrior(d["knot_speeds"], d["freq_knots"], d["period_knots"])


def plot_trajectory_overlay(
    variants: tuple[str, ...],
    n_samples: int = 4,
    out_path: Path | None = None,
) -> None:
    """Overlay pred vs GT time-domain trajectories for a few test samples.

    Picks samples uniformly across the speed range so every variant is tested
    at low / mid / high speed.
    """
    inputs  = np.load(INPUT_NPY).astype(np.float32)
    raw_fft = np.load(OUTPUT_NPY).astype(np.float32)
    split   = np.load(Path(DATA_DIR) / "split_indices.npz")
    test_idx = split["test"]

    mean_perbin = np.load(MEAN_TRAIN_NPY)
    std_global  = float(np.load(STD_TRAIN_NPY).reshape(-1)[0])
    spline_prior = load_spline_prior()

    # pick n_samples test samples uniformly along speed axis
    speeds = inputs[test_idx, 0] * 2.4
    order  = np.argsort(speeds)
    pick   = order[np.linspace(0, len(order) - 1, n_samples).astype(int)]
    sel_idx = test_idx[pick]

    models = {v: load_variant_model(v) for v in variants}

    joint_names = ("rhip", "rknee", "lhip", "lknee")
    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 2.8 * n_samples), sharex=True)
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, t_idx in enumerate(sel_idx):
        x_in = torch.tensor(inputs[t_idx:t_idx+1], dtype=torch.float32, device=DEVICE)
        speed_cmd = float(inputs[t_idx, 0] * 2.4)

        # GT
        gt_cx = raw_fft[t_idx, ..., 0] + 1j * raw_fft[t_idx, ..., 1]     # (17,4)
        gt_t  = np.fft.irfft(gt_cx, n=32, axis=0)                         # (32,4)

        # Each variant's prediction
        for v_name in variants:
            cfg = variant_config(v_name)
            with torch.no_grad():
                pf_raw, pp = models[v_name](x_in)
            pf_raw = pf_raw.cpu().numpy()[0]
            if cfg["use_residual"]:
                prior_freq_denorm, _ = spline_prior(np.array([speed_cmd], dtype=np.float32))
                prior_freq_norm = (prior_freq_denorm[0] - mean_perbin) / std_global
                pf = pf_raw + prior_freq_norm
            else:
                pf = pf_raw
            denorm = pf * std_global + mean_perbin
            denorm = denorm.reshape(17, 4, 2)
            cx = denorm[..., 0] + 1j * denorm[..., 1]
            pred_t = np.fft.irfft(cx, n=32, axis=0)                       # (32,4)

            for j in range(4):
                axes[row, j].plot(pred_t[:, j], label=v_name, alpha=0.8, linewidth=1.2)

        for j in range(4):
            axes[row, j].plot(gt_t[:, j], "k--", label="ground_truth", linewidth=1.5)
            if row == 0:
                axes[row, j].set_title(joint_names[j])
            if j == 0:
                axes[row, j].set_ylabel(f"{speed_cmd:.2f} m/s")
            axes[row, j].grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("cycle sample (0-31)")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150); plt.close(fig)
        print(f"[plot] wrote {out_path}")
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Smoke run (200 epochs).")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--skip-train", action="store_true",
                    help="Only collect metrics + plot; don't retrain.")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- train each variant
    if not args.skip_train:
        for v in args.variants:
            rc = train_variant(v, quick=args.quick, no_aug=args.no_aug)
            if rc != 0:
                print(f"[error] training '{v}' returned rc={rc}")
                return rc

    # ---- collect metrics
    rows = build_summary_table(tuple(args.variants))
    write_summary_csv(rows, OUT_DIR / "variants_summary.csv")
    print_leaderboard(rows)

    # ---- bar plots
    plot_bar(rows, "time_domain_mse", "Time-domain MSE (rad^2)",
             "Time-domain MSE on test split (lower=better)",
             OUT_DIR / "variants_time_mse.png")
    plot_bar(rows, "coef_mse", "FFT coefficient MSE",
             "FFT-coef MSE on test split",
             OUT_DIR / "variants_coef_mse.png")
    plot_bar(rows, "period_mae_s", "Period MAE (s)",
             "Period prediction MAE (seconds)",
             OUT_DIR / "variants_period_mae.png")
    plot_bar(rows, "fft_fidelity_h1_h3", "|FFT mag err| @ h1-h3",
             "Spectral fidelity on first 3 harmonics",
             OUT_DIR / "variants_fft_fidelity.png")

    # ---- trajectory overlay
    plot_trajectory_overlay(
        tuple(args.variants),
        n_samples=4,
        out_path=OUT_DIR / "variants_trajectory_overlay.png",
    )

    print(f"\n[done] comparison written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())