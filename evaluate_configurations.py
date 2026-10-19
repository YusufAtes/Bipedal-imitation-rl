"""Ablation-study evaluation over the ``configurations/`` folder.

Each sub-folder under ``configurations/`` is a different training
configuration for the ablation study. Inside each there is a single run
folder (``PPO_*`` / ``RecurrentPPO_*``) containing per-episode demo CSV logs.
For the noisy step-plane experiment the relevant files are::

    demo_data_noisy_mlp_0.csv / demo_data_noisy_lstm_0.csv   (noisy plane 0)
    demo_data_noisy_mlp_1.csv / demo_data_noisy_lstm_1.csv   (noisy plane 1)

The CMA-ES SIMBICON baseline is not a learned configuration; its data lives in
``cma-es_simbicon/simbicon_noisyplane_{0,1}.csv`` and is included so that the
regenerated figures match the legend used in ``report figures/``.

The script produces four figures:

    1. Noise amplitude vs success rate      -- plane 0 and plane 1
    2. Noise amplitude vs average max range  -- plane 0 and plane 1

Figures are written to both ``report figures/`` and ``configurations/``.

Usage
-----
    python evaluate_configurations.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = REPO_ROOT / "configurations"
CMA_ES_DIR = REPO_ROOT / "cma-es_simbicon"

# Output directories (figures are written to both).
OUT_DIRS = [REPO_ROOT / "report figures", CONFIG_ROOT]

# Ordered legend definition. The order, labels, colors and markers reproduce
# the figures already stored under ``report figures/``. ``folder`` is the name
# of the sub-folder under ``configurations/``; ``None`` marks the external
# CMA-ES SIMBICON baseline.
#
# (folder, legend label, marker)
SERIES: list[tuple[str | None, str, str]] = [
    ("nodecay_mlp_rsi", "NN 256-256 No Decay", "o"),
    ("nodecay_lstm_256_rsi", "LSTM 64 + NN 256", "s"),
    ("nodecay_lstm_256_256_rsi", "LSTM 64 + NN 256-256", "x"),
    ("025decay_mlp_rsi", r"Config 1 with Decay $\alpha$ = 0.25", "^"),
    ("05decay_mlp_rsi", r"Config 1 with Decay $\alpha$ = 0.5", "^"),
    ("nodecay_mlp_norsi", "NN 256-256 No RSI", "d"),
    (None, "CMA-ES SIMBICON", "*"),
    ("noimreward_mlp", "PPO", "v"),
]


def _to_success_float(series: pd.Series) -> pd.Series:
    """Coerce the ``success`` column to {0.0, 1.0} regardless of dtype."""
    if series.dtype == bool:
        return series.astype(float)
    as_str = series.astype(str).str.strip().str.lower()
    mapped = as_str.map({"true": 1.0, "false": 0.0})
    if mapped.notna().all():
        return mapped
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _find_noisy_csv(folder: str, plane: int) -> Path | None:
    """Locate the noisy-plane CSV for a configuration folder (mlp or lstm)."""
    config_dir = CONFIG_ROOT / folder
    if not config_dir.is_dir():
        return None
    matches = sorted(config_dir.glob(f"**/demo_data_noisy_*_{plane}.csv"))
    return matches[0] if matches else None


def _load_series_csv(folder: str | None, plane: int) -> pd.DataFrame | None:
    """Load the per-episode CSV for a series on a given noisy plane."""
    if folder is None:  # CMA-ES SIMBICON baseline.
        path = CMA_ES_DIR / f"simbicon_noisyplane_{plane}.csv"
    else:
        path = _find_noisy_csv(folder, plane)
    if path is None or not path.exists():
        print(f"[warn] missing noisy plane {plane} CSV for "
              f"{folder or 'CMA-ES SIMBICON'}")
        return None
    return pd.read_csv(path)


def _aggregate(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (noise_levels, success_rate, mean_max_range) sorted by noise."""
    df = df.copy()
    df["noise level"] = pd.to_numeric(df["noise level"], errors="coerce")
    df["success_num"] = _to_success_float(df["success"])
    df["max range"] = pd.to_numeric(df["max range"], errors="coerce")
    df = df.dropna(subset=["noise level"])

    grouped = df.groupby("noise level")
    success = grouped["success_num"].mean()
    max_range = grouped["max range"].mean()

    noise_levels = success.index.to_numpy()
    return noise_levels, success.to_numpy(), max_range.to_numpy()


def _plot(metric: str, plane: int, agg: dict[str, tuple], out_name: str) -> None:
    """Draw one figure for a given metric ("success" or "range") and plane."""
    title_metric = {
        "success": "Success Rate",
        "range": "Average Max Range",
    }[metric]
    ylabel = {
        "success": "Success Rate",
        "range": "Average Max Range (m)",
    }[metric]

    plt.figure(figsize=(10, 6))
    for folder, label, marker in SERIES:
        if label not in agg:
            continue
        noise, success, max_range = agg[label]
        y = success if metric == "success" else max_range
        plt.plot(noise, y, marker=marker, label=label)

    plt.title(f"Noise Level vs {title_metric} (Noisy Plane {plane})")
    plt.xlabel("Noise Level (cm)")
    plt.ylabel(ylabel)
    if metric == "success":
        plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_name
        plt.savefig(out_path, dpi=200)
        print(f"  saved {out_path}")
    plt.close()


def main() -> None:
    for plane in (0, 1):
        print(f"Aggregating noisy plane {plane} ...")
        agg: dict[str, tuple] = {}
        for folder, label, _marker in SERIES:
            df = _load_series_csv(folder, plane)
            if df is None:
                continue
            agg[label] = _aggregate(df)

        _plot("success", plane, agg, f"success_rate_plane{plane}.png")
        _plot("range", plane, agg, f"max_range_plane{plane}.png")

    print("Done.")


if __name__ == "__main__":
    main()
