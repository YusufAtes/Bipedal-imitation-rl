"""Push-recovery robustness demo (C5).

Rolls a trained PPO policy through episodes that start in nominal walking
and, at a uniformly random gait-phase time, apply a single-timestep external
impulse (``applyExternalForce`` in link-local frame) to the torso link. The
robot is then observed for the remainder of the episode; success means the
default termination conditions were never triggered before the episode ended.

Metrics produced per direction / magnitude cell:

* ``success_rate``         fraction of trials that survived the episode
* ``time_to_fall_mean``    mean time in seconds between push and termination
                           (NaN if every trial survived)
* ``cot_mean``             mean Cost of Transport of survivors

Outputs:

* ``push_recovery_trials.csv``  one row per trial
* ``push_recovery_summary.csv`` aggregated per (direction, magnitude)
* ``push_recovery.png``         success rate vs impulse magnitude

Usage
-----
    python push_recovery_demo.py \\
        --model-path configurations/025decay_mlp_rsi/PPO_1/final_model.zip \\
        --config configs/config_025decay_mlp_rsi.yaml \\
        --include-pad-dims \\
        --out figs_demo/push_recovery/alpha0.25

The paper's frozen snapshots were trained without DR; use
``--config configs/config1_with_dr.yaml`` on DR-trained policies for the full
robustness claim."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from biped_config import load_run_config
from biped_env import BipedEnv


_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "forward":  (1.0,  0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left":     (0.0,  1.0, 0.0),
    "right":    (0.0, -1.0, 0.0),
}


@dataclass
class TrialResult:
    direction: str
    magnitude: float
    trial: int
    push_time: float
    survived: bool
    time_to_fall: float
    cost_of_transport: float


def _rollout_with_push(
    env: BipedEnv, model: PPO, speed: float, max_env_steps: int,
    push_step: int, direction: tuple[float, float, float], magnitude: float,
) -> tuple[bool, float, float]:
    obs, _ = env.reset(test_speed=speed, test_angle=0.0,
                       demo_max_steps=int(max_env_steps * 10))
    survived = True
    time_to_fall = float("nan")
    last_cot = float("nan")
    for step in range(max_env_steps):
        if step == push_step:
            env.apply_external_impulse(magnitude=magnitude, direction=direction)
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, done, _t, info = env.step(action)
        last_cot = float(info.get("cost_of_transport", last_cot))
        if done:
            survived = False
            time_to_fall = max(0.0, (step - push_step)) * env.dt * 10.0
            break
    return survived, time_to_fall, last_cot


def _analyse(
    model_path: Path, cfg_path: Path, out_root: Path,
    magnitudes: np.ndarray, trials_per_cell: int, speed: float,
    episode_seconds: float, min_push_time: float, max_push_time: float,
    directions: list[str], include_pad_dims: bool | None, seed: int,
) -> None:
    cfg = load_run_config(cfg_path)
    if include_pad_dims is not None:
        cfg.env.include_pad_dims = bool(include_pad_dims)
    env = BipedEnv(config=cfg.env, demo_mode=True, demo_type="track")
    model = PPO.load(str(model_path), device="cpu")

    rng = np.random.default_rng(seed)
    env_steps_per_episode = int(episode_seconds / (cfg.env.dt * 10))
    min_push_step = int(min_push_time / (cfg.env.dt * 10))
    max_push_step = max(min_push_step + 1,
                        int(max_push_time / (cfg.env.dt * 10)))

    rows: list[TrialResult] = []
    for direction in directions:
        dvec = _DIRECTIONS[direction]
        for mag in magnitudes:
            for trial in range(trials_per_cell):
                push_step = int(rng.integers(min_push_step, max_push_step))
                surv, ttf, cot = _rollout_with_push(
                    env=env, model=model, speed=speed,
                    max_env_steps=env_steps_per_episode,
                    push_step=push_step, direction=dvec, magnitude=float(mag),
                )
                rows.append(TrialResult(
                    direction=direction, magnitude=float(mag), trial=trial,
                    push_time=push_step * cfg.env.dt * 10,
                    survived=surv, time_to_fall=ttf, cost_of_transport=cot,
                ))
    env.close()

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "push_recovery_trials.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["direction", "magnitude_N", "trial", "push_time_s",
                    "survived", "time_to_fall_s", "cost_of_transport"])
        for r in rows:
            w.writerow([r.direction, r.magnitude, r.trial, r.push_time,
                        r.survived, r.time_to_fall, r.cost_of_transport])

    agg: dict[tuple[str, float], dict[str, list[float]]] = {}
    for r in rows:
        bucket = agg.setdefault((r.direction, r.magnitude),
                                dict(surv=[], ttf=[], cot=[]))
        bucket["surv"].append(float(r.survived))
        if not r.survived:
            bucket["ttf"].append(r.time_to_fall)
        else:
            bucket["cot"].append(r.cost_of_transport)

    with (out_root / "push_recovery_summary.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["direction", "magnitude_N", "n", "success_rate",
                    "time_to_fall_mean_s", "cot_mean"])
        for (direction, mag) in sorted(agg.keys()):
            b = agg[(direction, mag)]
            n = len(b["surv"])
            sr = float(np.mean(b["surv"])) if n else float("nan")
            ttf = float(np.mean(b["ttf"])) if b["ttf"] else float("nan")
            cot = float(np.mean(b["cot"])) if b["cot"] else float("nan")
            w.writerow([direction, mag, n, sr, ttf, cot])

    _plot_success(agg, magnitudes, directions, out_root / "push_recovery.png")


def _plot_success(agg: dict, magnitudes: np.ndarray,
                  directions: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = {"forward": "#4c72b0", "backward": "#dd8452",
               "left": "#55a467", "right": "#c44e52"}
    for direction in directions:
        mags = sorted({m for (d, m) in agg.keys() if d == direction})
        if not mags:
            continue
        ys = [float(np.mean(agg[(direction, m)]["surv"])) for m in mags]
        ax.plot(mags, ys, marker="o",
                color=palette.get(direction, None),
                label=direction)
    ax.set_title("Push recovery success rate vs impulse magnitude")
    ax.set_xlabel("Impulse magnitude (N)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Push direction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("figs_demo/push_recovery"))
    p.add_argument("--magnitudes", type=float, nargs="+",
                   default=[50.0, 100.0, 150.0, 200.0, 250.0, 300.0])
    p.add_argument("--directions", nargs="+",
                   default=list(_DIRECTIONS.keys()),
                   choices=list(_DIRECTIONS.keys()))
    p.add_argument("--trials-per-cell", type=int, default=5)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--episode-seconds", type=float, default=3.0)
    p.add_argument("--min-push-time", type=float, default=0.5)
    p.add_argument("--max-push-time", type=float, default=1.5)
    p.add_argument("--include-pad-dims", action=argparse.BooleanOptionalAction,
                   default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _analyse(
        model_path=args.model_path, cfg_path=args.config, out_root=args.out,
        magnitudes=np.asarray(args.magnitudes, dtype=np.float64),
        trials_per_cell=args.trials_per_cell, speed=args.speed,
        episode_seconds=args.episode_seconds,
        min_push_time=args.min_push_time, max_push_time=args.max_push_time,
        directions=list(args.directions),
        include_pad_dims=args.include_pad_dims, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
