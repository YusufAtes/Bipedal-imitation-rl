"""Regenerate the closed-loop figure panels of the paper (Fig. 5 a-d) with
readable fonts (Reviewer 1, point 6), plus 95 % bootstrap CIs for the
success columns of Table 5 (Reviewer 1, point 2).

Data: the per-episode demo CSVs under ``configurations/<cfg>/<run>/`` and
``cma-es_simbicon/``. Output: two PDF figures in ``sn-article-template/figs``
and ``tables/table5_ci.csv``.

    python plot_paper_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stats_utils import mean_ci

REPO = Path(__file__).resolve().parent
CFG = REPO / "configurations"
CMA = REPO / "cma-es_simbicon"
FIGS = REPO / "sn-article-template" / "figs"

# label, folder (None = CMA-ES), colour, marker
SERIES = [
    ("Config. 1",                 "nodecay_mlp_rsi",          "#1f4e9c", "o"),
    ("Config. 2",                 "nodecay_lstm_256_rsi",     "#d1701a", "s"),
    ("Config. 3",                 "nodecay_lstm_256_256_rsi", "#2e8b57", "^"),
    (r"Config. 1, $\alpha=0.25$", "025decay_mlp_rsi",         "#c0392b", "v"),
    (r"Config. 1, $\alpha=0.5$",  "05decay_mlp_rsi",          "#7d3c98", "D"),
    ("Config. 1, no RSI",         "nodecay_mlp_norsi",        "#8c564b", "x"),
    ("CMA-ES SIMBICON",           None,                       "#e377c2", "*"),
    ("PPO baseline",              "noimreward_mlp",           "#555555", "P"),
]

plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.titlesize": 10,
                     "font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.2})


def _succ(s: pd.Series) -> pd.Series:
    m = s.astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})
    return m if m.notna().all() else pd.to_numeric(s, errors="coerce").fillna(0.0)


def load(folder: str | None, kind: str, plane: int = 0) -> pd.DataFrame:
    if folder is None:
        name = {"rot": "simbicon_rotation_data.csv", "vel": "simbicon_velocity_diff.csv",
                "noisy": f"simbicon_noisyplane_{plane}.csv"}[kind]
        return pd.read_csv(CMA / name)
    pat = {"rot": "demo_data_rotation_*_0.csv", "vel": "demo_data_vel_diff_*_0.csv",
           "noisy": f"demo_data_noisy_*_{plane}.csv"}[kind]
    files = sorted((CFG / folder).glob(f"*/{pat}"))
    if not files:
        raise FileNotFoundError(f"{folder}/{pat}")
    return pd.read_csv(files[0])


def panel_ramp(ax):
    for label, folder, c, m in SERIES:
        df = load(folder, "rot"); df["s"] = _succ(df["success"])
        g = df.groupby("angle")["s"].mean()
        ax.plot(g.index, g.values, color=c, marker=m, label=label)
    ax.axvspan(-5, 5, color="k", alpha=0.06, lw=0)
    ax.set_xlabel("ramp angle [deg]"); ax.set_ylabel("success rate"); ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3, lw=0.4); ax.set_title("(a)", loc="left")


def panel_vel(ax):
    for label, folder, c, m in SERIES:
        df = load(folder, "vel")
        g = df.groupby("cmd speed")["mean speed"].mean()
        ax.plot(g.index, g.values, color=c, marker=m, label=label)
    x = np.array([0.0, 2.2])
    ax.plot(x, x, "k--", lw=0.8, label="commanded")
    ax.fill_between(x, 0.9 * x, 1.1 * x, color="k", alpha=0.07, lw=0, label=r"$\pm10\%$ band")
    ax.set_xlabel("commanded speed [m/s]"); ax.set_ylabel("realized speed [m/s]")
    ax.set_xlim(0, 2.25); ax.set_ylim(0, 2.4); ax.grid(alpha=0.3, lw=0.4); ax.set_title("(b)", loc="left")


def panel_range(ax, plane: int, tag: str):
    for label, folder, c, m in SERIES:
        df = load(folder, "noisy", plane)
        df["noise level"] = pd.to_numeric(df["noise level"], errors="coerce")
        df["max range"] = pd.to_numeric(df["max range"], errors="coerce")
        g = df.dropna(subset=["noise level"]).groupby("noise level")["max range"].mean()
        ax.plot(g.index, g.values, color=c, marker=m, label=label)
    ax.set_xlabel("terrain noise amplitude [cm]"); ax.set_ylabel("mean max.\\ range [m]" if False else "mean max range [m]")
    ax.grid(alpha=0.3, lw=0.4); ax.set_title(f"{tag} Noisy Plane {plane}", loc="left")


def table5_ci() -> list[dict]:
    rows = []
    for label, folder, _c, _m in SERIES:
        r = dict(configuration=label)
        df = load(folder, "vel"); g = df.groupby("cmd speed")["mean speed"].mean()
        r["vel_mse"] = float(np.mean((g.values - g.index.values) ** 2))
        for kind, key in (("rot", "rot"), ("noisy0", "np0"), ("noisy1", "np1")):
            df = load(folder, "rot") if kind == "rot" else load(folder, "noisy", int(kind[-1]))
            s = _succ(df["success"]).to_numpy()
            ci = mean_ci(s)
            r[f"succ_{key}"], r[f"succ_{key}_lo"], r[f"succ_{key}_hi"], r[f"n_{key}"] = ci.mean, ci.ci_lo, ci.ci_hi, ci.n
            if kind != "rot":
                rg = pd.to_numeric(df["max range"], errors="coerce").dropna().to_numpy()
                ci = mean_ci(rg)
                r[f"range_{key}"], r[f"range_{key}_lo"], r[f"range_{key}_hi"] = ci.mean, ci.ci_lo, ci.ci_hi
        rows.append(r)
    return rows


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    panel_ramp(axes[0]); panel_vel(axes[1])
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(rect=[0, 0.14, 1, 1]); fig.savefig(FIGS / "fig_ramp_vel.pdf", bbox_inches="tight", pad_inches=0.01); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.6), sharey=True)
    panel_range(axes[0], 0, "(c)"); panel_range(axes[1], 1, "(d)")
    axes[1].set_ylabel("")
    fig.tight_layout(); fig.savefig(FIGS / "fig_range.pdf", bbox_inches="tight", pad_inches=0.01); plt.close(fig)
    print("wrote", FIGS / "fig_ramp_vel.pdf", "and", FIGS / "fig_range.pdf")

    rows = table5_ci()
    out = REPO / "tables"; out.mkdir(exist_ok=True)
    with (out / "table5_ci.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n{'configuration':28s} {'velMSE':>7} {'succ rot [95% CI]':>22} {'succ NP0 [95% CI]':>22} {'succ NP1 [95% CI]':>22} {'range NP0':>9} {'range NP1':>9}")
    for r in rows:
        print(f"{r['configuration']:28s} {r['vel_mse']:7.4f} "
              + " ".join(f"{r[f'succ_{k}']:.3f} [{r[f'succ_{k}_lo']:.3f},{r[f'succ_{k}_hi']:.3f}]" for k in ("rot", "np0", "np1"))
              + f" {r['range_np0']:9.3f} {r['range_np1']:9.3f}")
    print("wrote", out / "table5_ci.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
