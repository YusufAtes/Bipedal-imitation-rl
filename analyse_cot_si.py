"""Cost of Transport + per-joint Symmetry Index analysis (C4).

The paper aggregated a single averaged Symmetry Index and did not report
Cost of Transport at all. Reviewers flagged both gaps. This script closes
them by running a speed sweep against a trained PPO policy and recording,
per speed:

* ``cost_of_transport``   mean of ``info['cost_of_transport']`` at episode
                          end across successful trials
* ``si_hip``              Robinson's SI (%) on the right/left hip angles
* ``si_knee``             ditto for knee
* ``si_ankle``            ditto for ankle
* ``si_avg``              the previously reported all-joint mean

Outputs:

* ``cot_si_per_speed.csv``  one row per (speed, trial) with all metrics
* ``cot_si_summary.csv``    mean per speed
* ``cost_of_transport.png`` CoT vs commanded speed
* ``symmetry_by_joint.png`` hip / knee / ankle SI vs commanded speed

Usage
-----
    python analyse_cot_si.py \\
        --model-path configurations/025decay_mlp_rsi/PPO_1/final_model.zip \\
        --config configs/config_025decay_mlp_rsi.yaml \\
        --include-pad-dims \\
        --out figs_demo/cot_si/alpha0.25

    python analyse_cot_si.py --sweep-root runs/b3 --out figs_demo/cot_si/b3
"""

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


# Joint indices for the two layouts
_IDX_55 = dict(rhip=4, rknee=5, rankle=6, lhip=7, lknee=8, lankle=9)
_IDX_58 = dict(rhip=7, rknee=8, rankle=9, lhip=10, lknee=11, lankle=12)


def _joint_idx(include_pad_dims: bool) -> dict[str, int]:
    return _IDX_58 if include_pad_dims else _IDX_55


def _robinson_si(right: np.ndarray, left: np.ndarray, eps: float = 1e-6) -> float:
    r = np.asarray(right, dtype=np.float64)
    l = np.asarray(left, dtype=np.float64)
    if r.size == 0 or l.size == 0:
        return float("nan")
    denom = 0.5 * (np.abs(r) + np.abs(l)) + eps
    return float(np.mean(np.abs(r - l) / denom) * 100.0)


@dataclass
class TrialRow:
    speed: float
    trial: int
    success: bool
    cost_of_transport: float
    si_hip: float
    si_knee: float
    si_ankle: float
    si_avg: float


def _rollout(
    env: BipedEnv, model: PPO, speed: float, max_steps: int, idx: dict[str, int],
) -> TrialRow:
    obs, _ = env.reset(test_speed=speed, test_angle=0.0, demo_max_steps=max_steps)
    traces = {k: [] for k in idx}
    last_cot = float("nan")
    success = True
    for _ in range(max_steps // 10):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, done, _t, info = env.step(action)
        for k, i in idx.items():
            traces[k].append(float(obs[i]))
        last_cot = float(info.get("cost_of_transport", last_cot))
        if done:
            success = False
            break

    def arr(key: str) -> np.ndarray:
        return np.asarray(traces[key], dtype=np.float64)

    si_hip = _robinson_si(arr("rhip"), arr("lhip"))
    si_knee = _robinson_si(arr("rknee"), arr("lknee"))
    si_ankle = _robinson_si(arr("rankle"), arr("lankle"))
    si_avg = float(np.nanmean([si_hip, si_knee, si_ankle]))

    return TrialRow(
        speed=float(speed), trial=-1, success=success,
        cost_of_transport=last_cot,
        si_hip=si_hip, si_knee=si_knee, si_ankle=si_ankle, si_avg=si_avg,
    )


def _analyse(
    model_path: Path, cfg_path: Path, out_root: Path,
    speeds: np.ndarray, trials_per_speed: int, episode_seconds: float,
    include_pad_dims: bool | None,
) -> None:
    cfg = load_run_config(cfg_path)
    if include_pad_dims is not None:
        cfg.env.include_pad_dims = bool(include_pad_dims)
    env = BipedEnv(config=cfg.env, demo_mode=True, demo_type="symmetry_index")
    model = PPO.load(str(model_path), device="cpu")
    idx = _joint_idx(cfg.env.include_pad_dims)
    max_steps = int(episode_seconds / cfg.env.dt)

    rows: list[TrialRow] = []
    for speed in speeds:
        for trial in range(trials_per_speed):
            r = _rollout(env, model, float(speed), max_steps, idx)
            r.trial = trial
            rows.append(r)
    env.close()

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "cot_si_per_speed.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["speed", "trial", "success", "cost_of_transport",
                    "si_hip", "si_knee", "si_ankle", "si_avg"])
        for r in rows:
            w.writerow([r.speed, r.trial, r.success, r.cost_of_transport,
                        r.si_hip, r.si_knee, r.si_ankle, r.si_avg])

    # aggregate per speed
    summary: dict[float, dict[str, list[float]]] = {}
    for r in rows:
        if not r.success:
            continue
        bucket = summary.setdefault(r.speed, dict(cot=[], si_hip=[], si_knee=[], si_ankle=[], si_avg=[]))
        bucket["cot"].append(r.cost_of_transport)
        bucket["si_hip"].append(r.si_hip)
        bucket["si_knee"].append(r.si_knee)
        bucket["si_ankle"].append(r.si_ankle)
        bucket["si_avg"].append(r.si_avg)

    with (out_root / "cot_si_summary.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["speed", "n", "cot_mean", "si_hip_mean", "si_knee_mean",
                    "si_ankle_mean", "si_avg_mean"])
        for sp in sorted(summary):
            b = summary[sp]
            n = len(b["cot"])
            if n == 0:
                continue
            w.writerow([sp, n, float(np.nanmean(b["cot"])),
                        float(np.nanmean(b["si_hip"])),
                        float(np.nanmean(b["si_knee"])),
                        float(np.nanmean(b["si_ankle"])),
                        float(np.nanmean(b["si_avg"]))])

    _plot_cot(summary, out_root / "cost_of_transport.png")
    _plot_si(summary, out_root / "symmetry_by_joint.png")


def _plot_cot(summary: dict, path: Path) -> None:
    if not summary:
        return
    sps = sorted(summary.keys())
    cot = [float(np.nanmean(summary[s]["cot"])) for s in sps]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sps, cot, marker="o", color="#4c72b0")
    ax.set_title("Cost of Transport vs commanded speed")
    ax.set_xlabel("Commanded speed (m/s)")
    ax.set_ylabel("CoT (dimensionless)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_si(summary: dict, path: Path) -> None:
    if not summary:
        return
    sps = sorted(summary.keys())
    hip = [float(np.nanmean(summary[s]["si_hip"])) for s in sps]
    knee = [float(np.nanmean(summary[s]["si_knee"])) for s in sps]
    ankle = [float(np.nanmean(summary[s]["si_ankle"])) for s in sps]
    avg = [float(np.nanmean(summary[s]["si_avg"])) for s in sps]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sps, hip, marker="o", label="hip", color="#4c72b0")
    ax.plot(sps, knee, marker="s", label="knee", color="#dd8452")
    ax.plot(sps, ankle, marker="^", label="ankle", color="#55a467")
    ax.plot(sps, avg, marker="d", label="mean", color="#c44e52", linestyle="--")
    ax.set_title("Symmetry Index per joint vs commanded speed")
    ax.set_xlabel("Commanded speed (m/s)")
    ax.set_ylabel("Robinson SI (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--sweep-root", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("figs_demo/cot_si"))
    p.add_argument("--speeds", type=float, nargs="+", default=None,
                   help="Default: np.linspace(0.3, 2.0, 7)")
    p.add_argument("--trials-per-speed", type=int, default=3)
    p.add_argument("--episode-seconds", type=float, default=4.0)
    p.add_argument("--include-pad-dims", action=argparse.BooleanOptionalAction,
                   default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    speeds = (
        np.asarray(args.speeds, dtype=np.float64)
        if args.speeds else np.linspace(0.3, 2.0, 7)
    )
    jobs: list[tuple[str, Path, Path, bool | None]] = []
    if args.sweep_root is not None:
        for rd in sorted(p for p in args.sweep_root.iterdir() if p.is_dir()):
            mp = rd / "final_model.zip"
            cfg = rd / "config.yaml"
            if mp.exists() and cfg.exists():
                jobs.append((rd.name, mp, cfg, None))
    else:
        if args.model_path is None or args.config is None:
            print("[cot-si] supply either --sweep-root or (--model-path and --config).")
            return 2
        jobs.append((args.model_path.parent.name, args.model_path, args.config, args.include_pad_dims))

    for name, model, cfg, pad in jobs:
        out = args.out / name
        print(f"[cot-si] {name} -> {out}")
        _analyse(model_path=model, cfg_path=cfg, out_root=out,
                 speeds=speeds, trials_per_speed=args.trials_per_speed,
                 episode_seconds=args.episode_seconds,
                 include_pad_dims=(args.include_pad_dims if pad is None else pad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
