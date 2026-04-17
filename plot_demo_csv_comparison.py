"""Comparison plots and rebuttal-ready tables for the four demo CSVs.

The script has two complementary jobs:

1. **Plots** (original behaviour) -- overlay the four configurations on the
   commanded-vs-actual speed, ramp-angle success-rate and step-plane noise
   plots, writing PNGs into ``comparison_plots/``.

2. **Per-demo tables with 95% bootstrap CIs** -- replace the paper's
   monolithic Table 3 with four narrower tables (``vel_diff``,
   ``rotation``, ``noisy``, ``track``), each reporting mean +- 95% CI over
   trials. Tables are emitted as:

   - CSV (machine-readable)
   - Markdown (human-readable, stdout or file)
   - LaTeX (``\\begin{tabular}...``) that drops into ``paper.tex`` verbatim

Usage
-----
    # original: just plots
    python plot_demo_csv_comparison.py

    # also write the four tables (CSV + Markdown + LaTeX) into tables_demo/
    python plot_demo_csv_comparison.py --tables --out-tables tables_demo

    # point at an arbitrary set of config folders instead of the defaults
    python plot_demo_csv_comparison.py --tables \\
        --config-dirs configurations/025decay_mlp_rsi/PPO_1:alpha0.25 \\
                       configurations/nodecay_mlp_rsi/PPO_39:no_decay
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIGS: list[tuple[str, str]] = [
    ("new_decay_0.5_1.2", r"$\alpha = 0.5$"),
    ("new_decay_0.25_1.2", r"$\alpha = 0.25$"),
    ("no_decay_new", "no decay"),
    ("no_imstate_new", "no imitation (baseline)"),
]

VEL_FILE = "demo_data_vel_diff_mlp_0.csv"
ROT_FILE = "demo_data_rotation_mlp_0.csv"
NOISY_FILE = "demo_data_noisy_mlp_0.csv"
TRACK_FILE = "demo_data_track_mlp_0.csv"


def _to_success_float(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    as_str = series.astype(str).str.strip().str.lower()
    mapped = as_str.map({"true": 1.0, "false": 0.0})
    if mapped.notna().all():
        return mapped
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def load_csvs(base_path: Path) -> dict:
    data: dict = {}
    for folder, label in CONFIGS:
        folder_path = base_path / folder
        vel_path = folder_path / VEL_FILE
        rot_path = folder_path / ROT_FILE
        noisy_path = folder_path / NOISY_FILE
        if not vel_path.exists() or not rot_path.exists() or not noisy_path.exists():
            missing = [str(p) for p in [vel_path, rot_path, noisy_path] if not p.exists()]
            raise FileNotFoundError(f"Missing files for {folder}: {missing}")
        data[label] = {
            "vel": pd.read_csv(vel_path),
            "rot": pd.read_csv(rot_path),
            "noisy": pd.read_csv(noisy_path),
        }
    return data


def plot_cmd_vs_actual(data: dict, out_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    all_cmd_speeds: list[float] = []
    for _, label in CONFIGS:
        df = data[label]["vel"].copy()
        grouped = (
            df.groupby("cmd speed", as_index=False)["mean speed"]
            .mean()
            .sort_values("cmd speed")
        )
        all_cmd_speeds.extend(grouped["cmd speed"].tolist())
        mse = float(np.mean((grouped["mean speed"] - grouped["cmd speed"]) ** 2))
        plt.plot(
            grouped["cmd speed"],
            grouped["mean speed"],
            marker="o",
            label=f"{label} (MSE={mse:.4f})",
        )
    if all_cmd_speeds:
        min_v = min(all_cmd_speeds)
        max_v = max(all_cmd_speeds)
        plt.plot([min_v, max_v], [min_v, max_v], "k--", label="ideal tracking")
    plt.title("Commanded Speed vs Actual Speed")
    plt.xlabel("Commanded speed (m/s)")
    plt.ylabel("Actual mean speed (m/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_cmd_vs_actual_speed.png", dpi=200)
    plt.close()


def plot_ramp_success(data: dict, out_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for _, label in CONFIGS:
        df = data[label]["rot"].copy()
        df["success_num"] = _to_success_float(df["success"])
        grouped = (
            df.groupby("angle", as_index=False)["success_num"]
            .mean()
            .sort_values("angle")
        )
        plt.plot(grouped["angle"], grouped["success_num"], marker="o", label=label)
    plt.title("Ramp Angle Success Rate")
    plt.xlabel("Ramp angle (degrees)")
    plt.ylabel("Success rate")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_ramp_angle_success_rate.png", dpi=200)
    plt.close()


def plot_noise_success(data: dict, out_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for _, label in CONFIGS:
        df = data[label]["noisy"].copy()
        df["success_num"] = _to_success_float(df["success"])
        grouped = (
            df.groupby("noise level", as_index=False)["success_num"]
            .mean()
            .sort_values("noise level")
        )
        plt.plot(grouped["noise level"], grouped["success_num"], marker="o", label=label)
    plt.title("Step Plane: Noise Level Success Rate")
    plt.xlabel("Noise level")
    plt.ylabel("Success rate")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_step_plane_noise_success_rate.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# C1 -- per-demo tables with 95% bootstrap CIs and LaTeX exporter
# ---------------------------------------------------------------------------


@dataclass
class TableRow:
    config: str
    metric: str
    mean: float
    ci_lo: float
    ci_hi: float
    n: int


def bootstrap_ci(
    values: Iterable[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 123,
) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v, v
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(arr.mean()), float(lo), float(hi)


def _load_demo_csvs(
    config_pairs: list[tuple[Path, str]],
    files: dict[str, str],
) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for folder, label in config_pairs:
        per_cfg: dict[str, pd.DataFrame] = {}
        for demo, fname in files.items():
            path = folder / fname
            if path.exists():
                per_cfg[demo] = pd.read_csv(path)
        out[label] = per_cfg
    return out


def build_vel_diff_table(
    data_by_cfg: dict[str, dict[str, pd.DataFrame]],
) -> list[TableRow]:
    rows: list[TableRow] = []
    for cfg, tables in data_by_cfg.items():
        df = tables.get("vel")
        if df is None or df.empty:
            continue
        df = df.copy()
        df["success_num"] = _to_success_float(df["success"])
        per_trial_sq_err = (df["mean speed"] - df["cmd speed"]) ** 2
        mean, lo, hi = bootstrap_ci(per_trial_sq_err.dropna().tolist())
        rows.append(TableRow(cfg, "velocity MSE (m^2/s^2)", mean, lo, hi, int(per_trial_sq_err.count())))
        mean, lo, hi = bootstrap_ci(df["success_num"].dropna().tolist())
        rows.append(TableRow(cfg, "success rate", mean, lo, hi, int(df["success_num"].count())))
    return rows


def build_rotation_table(
    data_by_cfg: dict[str, dict[str, pd.DataFrame]],
) -> list[TableRow]:
    rows: list[TableRow] = []
    for cfg, tables in data_by_cfg.items():
        df = tables.get("rot")
        if df is None or df.empty:
            continue
        df = df.copy()
        df["success_num"] = _to_success_float(df["success"])
        mean, lo, hi = bootstrap_ci(df["success_num"].dropna().tolist())
        rows.append(TableRow(cfg, "success rate", mean, lo, hi, int(df["success_num"].count())))
        if "max range" in df.columns:
            mean, lo, hi = bootstrap_ci(pd.to_numeric(df["max range"], errors="coerce").dropna().tolist())
            rows.append(TableRow(cfg, "max range (m)", mean, lo, hi, int(pd.to_numeric(df["max range"], errors="coerce").count())))
    return rows


def build_noisy_table(
    data_by_cfg: dict[str, dict[str, pd.DataFrame]],
) -> list[TableRow]:
    rows: list[TableRow] = []
    for cfg, tables in data_by_cfg.items():
        df = tables.get("noisy")
        if df is None or df.empty:
            continue
        df = df.copy()
        df["success_num"] = _to_success_float(df["success"])
        mean, lo, hi = bootstrap_ci(df["success_num"].dropna().tolist())
        rows.append(TableRow(cfg, "success rate", mean, lo, hi, int(df["success_num"].count())))
        if "max range" in df.columns:
            mean, lo, hi = bootstrap_ci(pd.to_numeric(df["max range"], errors="coerce").dropna().tolist())
            rows.append(TableRow(cfg, "max range (m)", mean, lo, hi, int(pd.to_numeric(df["max range"], errors="coerce").count())))
    return rows


def build_track_table(
    data_by_cfg: dict[str, dict[str, pd.DataFrame]],
) -> list[TableRow]:
    rows: list[TableRow] = []
    for cfg, tables in data_by_cfg.items():
        df = tables.get("track")
        if df is None or df.empty:
            continue
        df = df.copy()
        # The track demo writes time-series rows with per-step (cmd_speed,
        # actual_speed) pairs. If the file uses the shared schema, rely on
        # ``cmd speed`` and ``mean speed`` as proxies.
        if "cmd speed" in df.columns and "mean speed" in df.columns:
            err = (df["mean speed"] - df["cmd speed"]).dropna()
            mean, lo, hi = bootstrap_ci((err ** 2).tolist())
            rows.append(TableRow(cfg, "tracking MSE (m^2/s^2)", mean, lo, hi, int(err.count())))
        if "success" in df.columns:
            s = _to_success_float(df["success"]).dropna()
            mean, lo, hi = bootstrap_ci(s.tolist())
            rows.append(TableRow(cfg, "success rate", mean, lo, hi, int(s.count())))
    return rows


def rows_to_dataframe(rows: list[TableRow]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "config": [r.config for r in rows],
            "metric": [r.metric for r in rows],
            "mean": [r.mean for r in rows],
            "ci_lo": [r.ci_lo for r in rows],
            "ci_hi": [r.ci_hi for r in rows],
            "n": [r.n for r in rows],
        }
    )


def rows_to_markdown(rows: list[TableRow], title: str) -> str:
    if not rows:
        return f"### {title}\n\n_(no data)_\n"
    lines = [f"### {title}", "", "| config | metric | mean | 95% CI | n |", "|---|---|---|---|---|"]
    for r in rows:
        ci = f"[{r.ci_lo:.4g}, {r.ci_hi:.4g}]" if np.isfinite(r.ci_lo) else "n/a"
        lines.append(f"| {r.config} | {r.metric} | {r.mean:.4g} | {ci} | {r.n} |")
    return "\n".join(lines) + "\n"


def rows_to_latex(rows: list[TableRow], caption: str, label: str) -> str:
    if not rows:
        return f"% Empty table for {label}\n"
    header = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:{label}}}\n"
        "\\begin{tabular}{llcccr}\n"
        "\\toprule\n"
        "Config & Metric & Mean & CI low & CI high & $n$ \\\\\n"
        "\\midrule\n"
    )
    body = []
    for r in rows:
        body.append(
            f"{r.config} & {r.metric} & {r.mean:.4g} & {r.ci_lo:.4g} & {r.ci_hi:.4g} & {r.n} \\\\"
        )
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + "\n".join(body) + footer


def build_all_tables(
    config_pairs: list[tuple[Path, str]],
) -> dict[str, list[TableRow]]:
    files = {
        "vel": VEL_FILE,
        "rot": ROT_FILE,
        "noisy": NOISY_FILE,
        "track": TRACK_FILE,
    }
    data_by_cfg = _load_demo_csvs(config_pairs, files)
    return {
        "vel_diff": build_vel_diff_table(data_by_cfg),
        "rotation": build_rotation_table(data_by_cfg),
        "noisy": build_noisy_table(data_by_cfg),
        "track": build_track_table(data_by_cfg),
    }


def emit_tables(
    tables: dict[str, list[TableRow]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    captions = {
        "vel_diff": "Velocity-tracking demo -- per-config metrics with 95\\% bootstrap CIs.",
        "rotation": "Rotation / ramp demo -- per-config metrics with 95\\% bootstrap CIs.",
        "noisy": "Step-plane noisy-terrain demo -- per-config metrics with 95\\% bootstrap CIs.",
        "track": "Velocity-tracking (profile) demo -- per-config metrics with 95\\% bootstrap CIs.",
    }
    md_sections: list[str] = []
    tex_sections: list[str] = []
    for key, rows in tables.items():
        df = rows_to_dataframe(rows)
        df.to_csv(out_dir / f"{key}.csv", index=False)
        md_sections.append(rows_to_markdown(rows, captions[key]))
        tex_sections.append(rows_to_latex(rows, captions[key], label=key))
    (out_dir / "all_tables.md").write_text("\n".join(md_sections), encoding="utf-8")
    (out_dir / "all_tables.tex").write_text("\n".join(tex_sections), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_config_dirs(raw: list[str] | None) -> list[tuple[Path, str]]:
    """Parse ``PATH:LABEL`` pairs; fall back to global CONFIGS if empty."""
    if not raw:
        return [(Path(folder), label) for folder, label in CONFIGS]
    out: list[tuple[Path, str]] = []
    for item in raw:
        if ":" not in item:
            raise ValueError(f"--config-dirs entry must be 'PATH:LABEL', got: {item!r}")
        path_str, label = item.rsplit(":", 1)
        out.append((Path(path_str), label))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comparison plots and tables from demo CSVs")
    p.add_argument("--out", type=Path, default=Path("comparison_plots"))
    p.add_argument("--tables", action="store_true", help="Emit the four per-demo tables.")
    p.add_argument("--out-tables", type=Path, default=Path("tables_demo"))
    p.add_argument(
        "--config-dirs", nargs="+", default=None,
        help="Override the hard-coded CONFIGS with 'PATH:LABEL' pairs.",
    )
    p.add_argument("--skip-plots", action="store_true", help="Only compute tables; no plots.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    base_path = Path(".")

    if not args.skip_plots:
        out_dir = args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            data = load_csvs(base_path)
            plot_cmd_vs_actual(data, out_dir)
            plot_ramp_success(data, out_dir)
            plot_noise_success(data, out_dir)
            print(f"Saved plots to: {out_dir.resolve()}")
        except FileNotFoundError as exc:
            print(f"[plot] skipping plots because of missing files: {exc}")

    if args.tables:
        config_pairs = _parse_config_dirs(args.config_dirs)
        tables = build_all_tables(config_pairs)
        emit_tables(tables, args.out_tables)
        print(f"Saved per-demo tables to: {args.out_tables.resolve()}")


if __name__ == "__main__":
    main()
