"""Post-training aggregator for the B3 gait-generator sweep.

Walks a sweep directory produced by :mod:`run_b3_sweep`, loads each
``final_model.zip``, runs a fixed evaluation protocol, and emits:

* ``per_seed.csv``      one row per (generator, seed) with all metrics
* ``summary.csv``       mean and 95% bootstrap CI per generator
* ``comparison.png``    bar charts of the four headline metrics

The evaluation protocol mirrors demo.py/``vel_diff`` but is intentionally
lean so it can run in a couple of minutes per seed:

* commanded speeds ``np.linspace(0.3, 2.0, 9)``
* ``trials_per_speed`` Monte-Carlo rollouts at each speed
* ``episode_seconds`` per rollout (default 4 s)

Metrics:

* **velocity_mse**     MSE between commanded and achieved forward speed
* **success_rate**     fraction of rollouts that did not early-terminate
* **cost_of_transport** mean ``info['cost_of_transport']`` at episode end
* **travel_range_m**   mean forward distance travelled
* **symmetry_index**   Robinson's symmetry index averaged over hip/knee/ankle

Usage
-----
    # aggregate runs/b3 (the full 20-run sweep)
    python evaluate_b3_sweep.py --sweep-root runs/b3 --out figs_demo/b3

    # quick-and-dirty: 2 trials, only speed 1.0
    python evaluate_b3_sweep.py --sweep-root runs/b3_smoke \
        --out figs_demo/b3_smoke --speeds 1.0 --trials-per-speed 2
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from biped_config import load_run_config
from biped_env import BipedEnv


# observation indices for the 55-D default layout (see biped_env.state_label_map)
IDX_RHIP, IDX_RKNEE, IDX_RANKLE = 4, 5, 6
IDX_LHIP, IDX_LKNEE, IDX_LANKLE = 7, 8, 9


@dataclass
class SeedResult:
    generator: str
    seed: int
    velocity_mse: float
    success_rate: float
    cost_of_transport: float
    travel_range_m: float
    symmetry_index: float


def _robinson_si(right: np.ndarray, left: np.ndarray, eps: float = 1e-6) -> float:
    r = np.asarray(right, dtype=np.float64)
    l = np.asarray(left, dtype=np.float64)
    if r.size == 0 or l.size == 0:
        return float("nan")
    denom = 0.5 * (np.abs(r) + np.abs(l)) + eps
    return float(np.mean(np.abs(r - l) / denom) * 100.0)


def _episode_symmetry(traces: dict[str, list[float]]) -> float:
    per_pair = [
        _robinson_si(np.asarray(traces["rhip"]), np.asarray(traces["lhip"])),
        _robinson_si(np.asarray(traces["rknee"]), np.asarray(traces["lknee"])),
        _robinson_si(np.asarray(traces["rankle"]), np.asarray(traces["lankle"])),
    ]
    return float(np.nanmean(per_pair))


def _run_rollout(
    env: BipedEnv,
    model: PPO,
    speed: float,
    max_steps: int,
) -> tuple[bool, float, float, float, dict[str, list[float]]]:
    obs, _ = env.reset(test_speed=speed, test_angle=0.0, demo_max_steps=max_steps)
    traces: dict[str, list[float]] = {k: [] for k in ("rhip", "rknee", "rankle", "lhip", "lknee", "lankle")}
    last_info: dict[str, float] = {}
    succeeded = True
    for _ in range(max_steps // 10):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, done, _trunc, info = env.step(action)
        traces["rhip"].append(float(obs[IDX_RHIP]))
        traces["rknee"].append(float(obs[IDX_RKNEE]))
        traces["rankle"].append(float(obs[IDX_RANKLE]))
        traces["lhip"].append(float(obs[IDX_LHIP]))
        traces["lknee"].append(float(obs[IDX_LKNEE]))
        traces["lankle"].append(float(obs[IDX_LANKLE]))
        last_info = info
        if done:
            succeeded = False
            break

    ext = env.return_external_state()
    travel = float(ext[1]) if len(ext) > 1 else 0.0
    cot = float(last_info.get("cost_of_transport", float("nan")))
    return succeeded, travel, cot, speed, traces


def _evaluate_seed(
    run_dir: Path,
    speeds: np.ndarray,
    trials_per_speed: int,
    episode_seconds: float,
) -> SeedResult | None:
    model_path = run_dir / "final_model.zip"
    cfg_path = run_dir / "config.yaml"
    if not model_path.exists() or not cfg_path.exists():
        print(f"[b3-eval] SKIP {run_dir.name} (missing config/final_model)")
        return None
    cfg = load_run_config(cfg_path)
    generator = cfg.env.gait_generator
    seed = int(cfg.train.seed)

    env = BipedEnv(config=cfg.env, demo_mode=True, demo_type="vel_diff")
    model = PPO.load(str(model_path), device="cpu")
    max_steps = int(episode_seconds / cfg.env.dt)

    vel_err_sq: list[float] = []
    cots: list[float] = []
    travels: list[float] = []
    si_scores: list[float] = []
    successes: list[int] = []

    for spd in speeds:
        for _ in range(trials_per_speed):
            ok, travel, cot, _, traces = _run_rollout(env, model, float(spd), max_steps)
            successes.append(1 if ok else 0)
            if ok:
                mean_speed = travel / episode_seconds if episode_seconds > 0 else 0.0
                vel_err_sq.append((mean_speed - spd) ** 2)
                travels.append(travel)
                if math.isfinite(cot):
                    cots.append(cot)
                si_scores.append(_episode_symmetry(traces))
    env.close()

    return SeedResult(
        generator=generator,
        seed=seed,
        velocity_mse=float(np.mean(vel_err_sq)) if vel_err_sq else float("nan"),
        success_rate=float(np.mean(successes)) if successes else float("nan"),
        cost_of_transport=float(np.mean(cots)) if cots else float("nan"),
        travel_range_m=float(np.mean(travels)) if travels else float("nan"),
        symmetry_index=float(np.mean(si_scores)) if si_scores else float("nan"),
    )


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(123)
    draws = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


METRICS = ("velocity_mse", "success_rate", "cost_of_transport", "travel_range_m", "symmetry_index")


def _write_per_seed(rows: list[SeedResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["generator", "seed", *METRICS])
        for r in rows:
            w.writerow([r.generator, r.seed, *[getattr(r, m) for m in METRICS]])


def _safe_nanmean(arr: np.ndarray) -> float:
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(arr))


def _write_summary(rows: list[SeedResult], path: Path) -> None:
    by_gen: dict[str, list[SeedResult]] = {}
    for r in rows:
        by_gen.setdefault(r.generator, []).append(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    header: list[str] = ["generator", "n_seeds"]
    for m in METRICS:
        header += [f"{m}_mean", f"{m}_ci_lo", f"{m}_ci_hi"]
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        for gen, items in sorted(by_gen.items()):
            row: list[object] = [gen, len(items)]
            for m in METRICS:
                arr = np.array([getattr(it, m) for it in items], dtype=np.float64)
                mean = _safe_nanmean(arr)
                lo, hi = _bootstrap_ci(arr)
                row += [mean, lo, hi]
            w.writerow(row)


def _plot_comparison(rows: list[SeedResult], out: Path) -> None:
    by_gen: dict[str, list[SeedResult]] = {}
    for r in rows:
        by_gen.setdefault(r.generator, []).append(r)
    gens = sorted(by_gen.keys())
    if not gens:
        return
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 4))
    if len(METRICS) == 1:
        axes = [axes]
    for ax, m in zip(axes, METRICS):
        means = [_safe_nanmean(np.asarray([getattr(it, m) for it in by_gen[g]], dtype=np.float64)) for g in gens]
        stds = []
        for g in gens:
            arr = np.asarray([getattr(it, m) for it in by_gen[g]], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            stds.append(float(np.std(arr)) if arr.size > 0 else 0.0)
        ax.bar(gens, means, yerr=stds, capsize=4, color=["#4c72b0", "#dd8452", "#55a467", "#c44e52", "#8172b2", "#937860"][: len(gens)])
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("B3: gait-generator comparison (mean +- std across seeds)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=Path, default=Path("runs/b3"))
    p.add_argument("--out", type=Path, default=Path("figs_demo/b3"))
    p.add_argument("--speeds", type=float, nargs="+", default=None,
                   help="Commanded speeds. Default: np.linspace(0.3, 2.0, 9)")
    p.add_argument("--trials-per-speed", type=int, default=3)
    p.add_argument("--episode-seconds", type=float, default=4.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    speeds = (
        np.asarray(args.speeds, dtype=np.float64)
        if args.speeds
        else np.linspace(0.3, 2.0, 9)
    )
    run_dirs = sorted(p for p in args.sweep_root.iterdir() if p.is_dir())
    if not run_dirs:
        print(f"[b3-eval] no run dirs under {args.sweep_root}")
        return 1

    results: list[SeedResult] = []
    for rd in run_dirs:
        print(f"[b3-eval] evaluating {rd.name}")
        res = _evaluate_seed(rd, speeds, args.trials_per_speed, args.episode_seconds)
        if res is not None:
            results.append(res)

    if not results:
        print("[b3-eval] nothing evaluated (no final_model.zip found).")
        return 1

    _write_per_seed(results, args.out / "per_seed.csv")
    _write_summary(results, args.out / "summary.csv")
    _plot_comparison(results, args.out / "comparison.png")
    print(f"[b3-eval] wrote {args.out}/per_seed.csv, summary.csv, comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
