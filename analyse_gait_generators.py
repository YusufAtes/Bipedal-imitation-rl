"""Stand-alone evaluation of every gait generator (B1) — FAIR PROTOCOL.

Reviewer-raised issue in v1
---------------------------
The original ``analyse_gait_generators.py`` evaluated every generator
against ``RawMocapGenerator``'s nearest-speed sample at a handful of query
speeds.  This gave ``raw_mocap`` MSE -> 0 on samples it had memorised and
penalised the FFT-MLP for having to compress 1121 samples into 369 k
weights.  Apples-to-oranges.

Fair protocol (this script)
---------------------------
1. Load the train / val / test indices produced by ``fft_datacreate_review.py``
   (``gait reference phase 2/split_indices.npz``).
2. Fit / restrict every generator to TRAIN samples only:
     * ``raw_mocap_train``   -- nearest-neighbour over train corpus only
     * ``cubic_spline_train`` -- per-speed bucket over train corpus only
     * ``cpg_matsuoka``       -- data-free, no restriction needed
     * ``fft_mlp_review``     -- already trained on train split
3. Iterate over TEST samples.  For each held-out sample with its own
   ``(speed_i, r_leg_i, l_leg_i)`` and ground-truth time-domain trajectory
   (reconstructed from its own FFT coefficients), evaluate every generator.
4. Aggregate per speed bucket + produce the same plots.  Now nobody can
   cheat by looking up their own training sample.

Outputs
-------
* ``<out>/per_speed_mse.png``
* ``<out>/per_speed_dtw.png``
* ``<out>/fft_fidelity.png``
* ``<out>/morphology.png``  (unchanged — still a held-speed sweep)
* ``<out>/summary.csv``     per-speed & morphology numbers
* ``<out>/per_sample.csv``  one row per (generator, test_idx)

Run with:
    python analyse_gait_generators.py --out figs_demo/gait_gen_review
    python analyse_gait_generators.py --out figs_demo/gait_gen_loso --loso

The ``--loso`` flag is a stub that documents the leave-one-subject-out
variant.  Subject identity is not encoded in ``input_vector.npy``; re-parsing
the C3D folder tree is required.  Out of scope for this commit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gait_generators import build_generator
from gait_generators.raw_mocap import RawMocapGenerator
from gait_generators.cubic_spline import CubicSplineGenerator


# Generators evaluated in the fair protocol. Order dictates legend order.
GENERATOR_NAMES = ("fft_mlp", "fft_mlp_review", "raw_mocap", "cubic_spline", "cpg_matsuoka")

SPLIT_PATH = Path("gait reference phase 2") / "split_indices.npz"

# Speed buckets used for the per-speed curves (edge-inclusive).
DEFAULT_SPEED_BINS = np.array([0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.4])


# ----------------------------------------------------------------------------
# DTW + FFT metrics (unchanged)
# ----------------------------------------------------------------------------
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
    return float(np.mean(np.abs(np.abs(fa[:k]) - np.abs(fb[:k]))))


def _period_sample(traj: np.ndarray, n_points: int = 32) -> np.ndarray:
    """Resample a (T, D) trajectory to ``n_points`` samples over one period.

    One period is inferred from the first two zero-crossings of joint 0
    (mean-removed). Falls back to a fixed 400-sample window if the signal
    is too short / non-oscillatory.
    """
    signal = traj[:, 0] - float(np.mean(traj[:, 0]))
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    if len(zero_crossings) >= 3:
        start = int(zero_crossings[0])
        end   = int(zero_crossings[2])
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


# ----------------------------------------------------------------------------
# Train-restricted generator wrappers
# ----------------------------------------------------------------------------
def _restrict_mocap_to_train(mocap: RawMocapGenerator, train_idx: np.ndarray) -> None:
    """In-place restriction of a RawMocapGenerator's corpus."""
    mocap.input_vector = mocap.input_vector[train_idx]
    mocap.targets      = mocap.targets[train_idx]
    mocap.period       = mocap.period[train_idx]
    mocap.speeds       = mocap.speeds[train_idx]


class TrainRestrictedCubicSpline(CubicSplineGenerator):
    """Cubic spline that interpolates only over train-index mocap samples."""

    def __init__(self, train_idx: np.ndarray, **kwargs) -> None:
        super().__init__(**kwargs)
        _restrict_mocap_to_train(self._mocap, train_idx)


def build_fair_generators(train_idx: np.ndarray) -> dict[str, object]:
    """Every generator is fitted / restricted to the train split only."""
    gens: dict[str, object] = {}

    # -- fft_mlp_review: already trained on train split by fft_datacreate_review.py
    gens["fft_mlp_review"] = build_generator(
        "fft_mlp_review", dt=1e-3, tile_repeats=1
    )
    gens["fft_mlp"] = build_generator(
        "fft_mlp", dt=1e-3, tile_repeats=1
    )
    # -- raw_mocap: load full corpus, then filter in-place to train_idx
    raw = RawMocapGenerator(dt=1e-3, tile_repeats=1)
    _restrict_mocap_to_train(raw, train_idx)
    gens["raw_mocap"] = raw

    # -- cubic_spline: restricted corpus inside
    gens["cubic_spline"] = TrainRestrictedCubicSpline(
        train_idx=train_idx, dt=1e-3, tile_repeats=1
    )

    # -- cpg_matsuoka: data-free, nothing to restrict
    gens["cpg_matsuoka"] = build_generator(
        "cpg_matsuoka", dt=1e-3, tile_repeats=1
    )
    return gens


# ----------------------------------------------------------------------------
# Ground-truth reconstruction for a TEST sample
# ----------------------------------------------------------------------------
def _reconstruct_test_sample(
    test_raw_fft: np.ndarray,   # (17, 4, 2)
) -> np.ndarray:
    """IRFFT a single test sample's FFT coefficients to a 32-sample, 6-joint cycle.

    Ankles stay zero (mocap doesn't record them).
    """
    complex_coefs = test_raw_fft[..., 0] + 1j * test_raw_fft[..., 1]  # (17, 4)
    time4 = np.fft.irfft(complex_coefs, n=32, axis=0)                 # (32, 4)

    time6 = np.zeros((time4.shape[0], 6), dtype=np.float64)
    time6[:, 0] = time4[:, 0]   # rhip
    time6[:, 1] = time4[:, 1]   # rknee
    time6[:, 3] = time4[:, 2]   # lhip
    time6[:, 4] = time4[:, 3]   # lknee
    return time6


# ----------------------------------------------------------------------------
# Fair per-test-sample evaluation
# ----------------------------------------------------------------------------
def evaluate_per_test_sample(
    test_idx: np.ndarray,
    inputs: np.ndarray,        # (N, 3) encoder_vec  (speed/2.4, r_leg, l_leg)
    raw_fft: np.ndarray,       # (N, 17, 4, 2)
    generators: dict[str, object],
) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    """
    Returns
    -------
    rows     : one dict per (test_sample, generator)
    per_gen  : per-generator concatenated metric arrays for binning later
    """
    rows: list[dict] = []
    per_gen: dict[str, dict[str, list[float]]] = {
        name: {"speed": [], "mse": [], "dtw": [], "fft": []}
        for name in generators
    }

    for i_test in test_idx:
        # inputs[i_test] stored as (speed/2.4, r_leg/1.0, l_leg/1.0)
        speed_cmd = float(inputs[i_test, 0]) * 2.4
        r_leg     = float(inputs[i_test, 1]) * 1.0
        l_leg     = float(inputs[i_test, 2]) * 1.0

        # Ground truth trajectory from the test sample's OWN FFT
        gt_full = _reconstruct_test_sample(raw_fft[i_test])   # (32, 6)
        gt_sampled = _period_sample(gt_full, n_points=32)

        for name, gen in generators.items():
            traj = gen.predict(speed_cmd, (r_leg, l_leg))
            sampled = _period_sample(traj, n_points=32)
            mse = float(np.mean((sampled - gt_sampled) ** 2))
            dtw = _dtw(sampled, gt_sampled)
            fft = _fft_fidelity(sampled, gt_sampled, n_harmonics=3)
            rows.append({
                "generator": name,
                "test_idx":  int(i_test),
                "speed":     speed_cmd,
                "r_leg":     r_leg,
                "l_leg":     l_leg,
                "mse":       mse,
                "dtw":       dtw,
                "fft":       fft,
            })
            per_gen[name]["speed"].append(speed_cmd)
            per_gen[name]["mse"].append(mse)
            per_gen[name]["dtw"].append(dtw)
            per_gen[name]["fft"].append(fft)

    return rows, {n: {k: np.asarray(v) for k, v in d.items()} for n, d in per_gen.items()}


# ----------------------------------------------------------------------------
# Speed-bucket aggregation
# ----------------------------------------------------------------------------
def aggregate_by_speed(
    per_gen: dict[str, dict[str, np.ndarray]],
    bin_edges: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Mean of each metric inside speed bins; bin centres returned too."""
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    agg: dict[str, dict[str, np.ndarray]] = {}
    for name, d in per_gen.items():
        speeds = d["speed"]
        means = {k: np.full(len(centres), np.nan) for k in ("mse", "dtw", "fft")}
        counts = np.zeros(len(centres), dtype=int)
        for i in range(len(centres)):
            mask = (speeds >= bin_edges[i]) & (speeds < bin_edges[i + 1])
            counts[i] = int(mask.sum())
            if mask.any():
                for k in ("mse", "dtw", "fft"):
                    means[k][i] = float(d[k][mask].mean())
        agg[name] = {
            "centres": centres,
            "counts":  counts,
            **means,
        }
    return agg


# ----------------------------------------------------------------------------
# Morphology sweep (unchanged in spirit, but uses the TRAIN-restricted gens)
# ----------------------------------------------------------------------------
def evaluate_morphology(
    leg_lengths: np.ndarray,
    generators: dict[str, object],
    speed: float = 1.0,
) -> dict[str, np.ndarray]:
    base = {name: gen.predict(speed, (0.94, 0.94)) for name, gen in generators.items()}
    drift: dict[str, np.ndarray] = {name: np.zeros(len(leg_lengths)) for name in generators}
    for j, leg in enumerate(leg_lengths):
        for name, gen in generators.items():
            new = gen.predict(speed, (float(leg), float(leg)))
            n = min(base[name].shape[0], new.shape[0], 400)
            drift[name][j] = float(np.sqrt(np.mean((new[:n] - base[name][:n]) ** 2)))
    return drift


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
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
        # skip NaN-only series (empty bin)
        if np.all(np.isnan(y)):
            continue
        plt.plot(xs, y, marker="o", label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load_split(path: Path = SPLIT_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python fft_datacreate_review.py` first "
            "to produce the train/val/test split."
        )
    npz = np.load(path)
    return npz["train"], npz["val"], npz["test"]


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    base = Path("gait reference phase 2")
    inputs  = np.load(base / "input_vector.npy")
    raw_fft = np.load(base / "output_fft_constants.npy")
    return inputs, raw_fft


def write_summary_csv(
    out_path: Path,
    agg: dict[str, dict[str, np.ndarray]],
    morph_leg: np.ndarray,
    morph_vals: dict[str, np.ndarray],
) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generator", "speed_bin_centre", "n_samples",
                    "mse", "dtw", "fft_fidelity"])
        for name, d in agg.items():
            for i in range(len(d["centres"])):
                w.writerow([
                    name,
                    float(d["centres"][i]),
                    int(d["counts"][i]),
                    float(d["mse"][i]) if not np.isnan(d["mse"][i]) else "",
                    float(d["dtw"][i]) if not np.isnan(d["dtw"][i]) else "",
                    float(d["fft"][i]) if not np.isnan(d["fft"][i]) else "",
                ])
        # morphology rows (mirrors v1 layout)
        for name, vals in morph_vals.items():
            for j, leg in enumerate(morph_leg):
                w.writerow([
                    f"{name}__morphology", float(leg), 1,
                    float(vals[j]), "", "",
                ])


def write_per_sample_csv(out_path: Path, rows: list[dict]) -> None:
    keys = ["generator", "test_idx", "speed", "r_leg", "l_leg", "mse", "dtw", "fft"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="figs_demo/gait_gen_review")
    parser.add_argument(
        "--speed-bins",
        type=float,
        nargs="*",
        default=None,
        help="Bin edges for the per-speed plots (default: 0.2,0.5,0.8,1.1,1.4,1.7,2.0,2.4).",
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Leave-one-subject-out (stub — requires re-parsing dataset/*/*.c3d).",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.loso:
        raise NotImplementedError(
            "LOSO mode requires subject IDs from C3D folder names. "
            "Extend fft_datacreate_review.py to emit a subject_id array "
            "alongside split_indices.npz, then fold it into build_fair_generators."
        )

    # --- 1. data + split
    train_idx, val_idx, test_idx = load_split()
    inputs, raw_fft = load_dataset()
    print(f"[data] train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # --- 2. generators restricted to train corpus
    generators = build_fair_generators(train_idx)
    print(f"[gen]  fitted: {list(generators)}")

    # --- 3. per-test-sample metrics
    rows, per_gen = evaluate_per_test_sample(test_idx, inputs, raw_fft, generators)
    write_per_sample_csv(out / "per_sample.csv", rows)
    print(f"[eval] {len(rows)} rows written to {out/'per_sample.csv'}")

    # --- 4. per-speed aggregation
    bin_edges = (np.asarray(args.speed_bins, dtype=float)
                 if args.speed_bins else DEFAULT_SPEED_BINS)
    agg = aggregate_by_speed(per_gen, bin_edges)
    centres = agg[GENERATOR_NAMES[0]]["centres"]

    _plot_curves(
        centres,
        {n: agg[n]["mse"] for n in GENERATOR_NAMES},
        "Commanded speed (m/s, bin centre)",
        "Per-joint MSE (rad^2)",
        "Gait-generator accuracy vs. held-out test samples",
        out / "per_speed_mse.png",
    )
    _plot_curves(
        centres,
        {n: agg[n]["dtw"] for n in GENERATOR_NAMES},
        "Commanded speed (m/s, bin centre)",
        "Mean DTW distance",
        "DTW vs. held-out test samples",
        out / "per_speed_dtw.png",
    )
    _plot_curves(
        centres,
        {n: agg[n]["fft"] for n in GENERATOR_NAMES},
        "Commanded speed (m/s, bin centre)",
        "Mean |FFT mag error| @ first 3 harmonics",
        "Spectral fidelity vs. held-out test samples",
        out / "fft_fidelity.png",
    )

    # --- 5. morphology sweep (unchanged metric, train-restricted gens)
    leg_lengths = np.linspace(0.94 * 0.8, 0.94 * 1.2, 7)
    morph = evaluate_morphology(leg_lengths, generators)
    _plot_curves(
        leg_lengths, morph,
        "Leg length (m)",
        "RMS deviation from 0.94 m baseline (rad)",
        "Morphology generalisation (speed = 1.0 m/s)",
        out / "morphology.png",
    )

    # --- 6. summary CSV
    write_summary_csv(out / "summary.csv", agg, leg_lengths, morph)

    # --- 7. print a quick textual leaderboard
    print("\n[leaderboard] mean over all test samples:")
    print(f"  {'generator':<18s}  {'MSE':>8s}  {'DTW':>8s}  {'FFT':>8s}  n")
    for n in GENERATOR_NAMES:
        d = per_gen[n]
        print(f"  {n:<18s}  {d['mse'].mean():>8.4f}  "
              f"{d['dtw'].mean():>8.4f}  {d['fft'].mean():>8.4f}  {len(d['mse'])}")

    print(f"\n[done] analysis written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())