"""Stand-alone evaluation of every gait generator (B1).

Without touching PPO, this script quantifies how well each
:class:`gait_generators.BaseGaitGenerator` matches the mocap ground truth
and how robust it is to morphology changes. It produces:

1. ``<out>/per_speed_mse.png``    -- MSE vs speed for each generator.
2. ``<out>/per_speed_dtw.png``    -- Dynamic Time Warping distance vs speed.
3. ``<out>/fft_fidelity.png``     -- Spectral magnitude error at the first
                                     three harmonics of the gait frequency.
4. ``<out>/morphology.png``       -- Output drift as leg length varies
                                     +/- 20 % at a fixed 1.0 m/s.
5. ``<out>/summary.csv``          -- Same numbers in tabular form.

Run with:

    python analyse_gait_generators.py --out figs_demo/gait_gen_b1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gait_generators import build_generator
from gait_generators.raw_mocap import RawMocapGenerator


GENERATOR_NAMES = ("fft_mlp", "raw_mocap", "cubic_spline", "cpg_matsuoka")


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    """O(n^2) multivariate DTW distance. a,b: (T, D)."""
    n, m = a.shape[0], b.shape[0]
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, m] / max(n, m))


def _fft_fidelity(a: np.ndarray, b: np.ndarray, n_harmonics: int = 3) -> float:
    fa = np.fft.rfft(a, axis=0)
    fb = np.fft.rfft(b, axis=0)
    k = 1 + n_harmonics
    ma = np.abs(fa[:k])
    mb = np.abs(fb[:k])
    return float(np.mean(np.abs(ma - mb)))


def _period_sample(traj: np.ndarray, n_points: int = 32) -> np.ndarray:
    """Resample a (T, D) trajectory to ``n_points`` samples covering the first period.

    We infer one period by looking at the first full cycle of joint 0 — the
    interval between the first two zero-crossings of the mean-removed signal.
    """
    signal = traj[:, 0] - float(np.mean(traj[:, 0]))
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    if len(zero_crossings) >= 3:
        start = int(zero_crossings[0])
        end = int(zero_crossings[2])  # full period covers two sign flips
    else:
        start, end = 0, min(traj.shape[0] - 1, 400)
    window = traj[start : end + 1]
    if window.shape[0] < 2:
        window = traj[: min(200, traj.shape[0])]
    idx = np.linspace(0, window.shape[0] - 1, n_points)
    return np.stack(
        [np.interp(idx, np.arange(window.shape[0]), window[:, j])
         for j in range(window.shape[1])],
        axis=1,
    )


def _reference_at_speed(mocap: RawMocapGenerator, speed: float) -> np.ndarray:
    """Ground-truth mocap trajectory at the query speed, normalised over one period."""
    idx = int(np.argmin(np.abs(mocap.speeds - speed)))
    ref4 = mocap._reconstruct_from_freqs(mocap.targets[idx, :136])  # (32, 4)
    ref6 = np.zeros((32, 6))
    ref6[:, 0] = ref4[:, 0]
    ref6[:, 1] = ref4[:, 1]
    ref6[:, 3] = ref4[:, 2]
    ref6[:, 4] = ref4[:, 3]
    return ref6


def evaluate_per_speed(
    speeds: np.ndarray,
    generators: dict,
    mocap: RawMocapGenerator,
) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {
        name: {"mse": np.zeros(len(speeds)),
               "dtw": np.zeros(len(speeds)),
               "fft": np.zeros(len(speeds))}
        for name in generators
    }
    for j, spd in enumerate(speeds):
        ref = _reference_at_speed(mocap, float(spd))
        for name, gen in generators.items():
            traj = gen.predict(float(spd), (0.94, 0.94))
            sampled = _period_sample(traj, n_points=32)
            results[name]["mse"][j] = float(np.mean((sampled - ref) ** 2))
            results[name]["dtw"][j] = _dtw(sampled, ref)
            results[name]["fft"][j] = _fft_fidelity(sampled, ref, n_harmonics=3)
    return results


def evaluate_morphology(
    leg_lengths: np.ndarray, generators: dict, speed: float = 1.0,
) -> dict[str, np.ndarray]:
    base = {name: gen.predict(speed, (0.94, 0.94)) for name, gen in generators.items()}
    drift: dict[str, np.ndarray] = {name: np.zeros(len(leg_lengths)) for name in generators}
    for j, leg in enumerate(leg_lengths):
        for name, gen in generators.items():
            new = gen.predict(speed, (float(leg), float(leg)))
            n = min(base[name].shape[0], new.shape[0], 400)
            drift[name][j] = float(np.sqrt(np.mean((new[:n] - base[name][:n]) ** 2)))
    return drift


def _plot_curves(
    xs: np.ndarray,
    ys: dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7, 4))
    for name, y in ys.items():
        plt.plot(xs, y, marker="o", label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="figs_demo/gait_gen_b1")
    parser.add_argument("--speeds", type=float, nargs="*", default=None)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    speeds = np.asarray(args.speeds or [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    leg_lengths = np.linspace(0.94 * 0.8, 0.94 * 1.2, 7)

    generators = {name: build_generator(name, dt=1e-3, tile_repeats=1) for name in GENERATOR_NAMES}
    mocap = RawMocapGenerator(dt=1e-3, tile_repeats=1)

    per_speed = evaluate_per_speed(speeds, generators, mocap)
    _plot_curves(
        speeds, {n: per_speed[n]["mse"] for n in GENERATOR_NAMES},
        "Command speed (m/s)", "Per-joint MSE (rad^2)",
        "Gait-generator accuracy vs. mocap", out / "per_speed_mse.png",
    )
    _plot_curves(
        speeds, {n: per_speed[n]["dtw"] for n in GENERATOR_NAMES},
        "Command speed (m/s)", "Mean DTW distance",
        "Dynamic Time Warping distance vs. mocap", out / "per_speed_dtw.png",
    )
    _plot_curves(
        speeds, {n: per_speed[n]["fft"] for n in GENERATOR_NAMES},
        "Command speed (m/s)", "Mean |FFT mag error| @ first 3 harmonics",
        "Spectral fidelity vs. mocap", out / "fft_fidelity.png",
    )

    morph = evaluate_morphology(leg_lengths, generators)
    _plot_curves(
        leg_lengths, morph,
        "Leg length (m)", "RMS deviation from 0.94 m baseline (rad)",
        "Morphology generalisation (speed = 1.0 m/s)", out / "morphology.png",
    )

    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generator", "speed", "mse", "dtw", "fft_fidelity",
        ])
        for name in GENERATOR_NAMES:
            for j, spd in enumerate(speeds):
                writer.writerow([
                    name, float(spd),
                    float(per_speed[name]["mse"][j]),
                    float(per_speed[name]["dtw"][j]),
                    float(per_speed[name]["fft"][j]),
                ])
        for name in GENERATOR_NAMES:
            for j, leg in enumerate(leg_lengths):
                writer.writerow([
                    f"{name}__morphology", float(leg),
                    float(morph[name][j]), 0.0, 0.0,
                ])

    print(f"[done] analysis written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
