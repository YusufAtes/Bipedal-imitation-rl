"""Velocity-tracking demo across every configuration (C2).

The paper's ``track`` demo was previously run only on Configuration 1 with a
hard-coded constant command. This script replaces that narrow protocol:

1. For every (model, config) pair the user supplies, roll out a time-varying
   speed profile (ramp up, hold, step down, ramp down) for
   ``--episode-seconds`` seconds.
2. Record the commanded and achieved forward speed at a regular interval.
3. Compute:
   - **RMSE**    root-mean-square tracking error across the episode.
   - **MAE**     mean absolute tracking error.
   - **Phase lag** of actual relative to commanded, inferred from the peak
     of the cross-correlation of zero-mean signals. Reported in seconds.
4. Emit:
   - per-config ``<config>/track.csv``  (t, cmd_speed, actual_speed, error)
   - per-config ``<config>/track.png``  overlay plot
   - aggregate ``summary.csv``          one row per config

Usage
-----
    # run the four legacy configurations + the B3 sweep in one shot
    python track_demo.py \\
        --models \\
            configurations/025decay_mlp_rsi/PPO_1:alpha=0.25:configs/config_025decay_mlp_rsi.yaml:58 \\
            configurations/nodecay_mlp_rsi/PPO_39:no_decay:configs/config1_nodecay_mlp_rsi.yaml:58 \\
        --out figs_demo/track

    # analyse every sub-folder of a sweep root
    python track_demo.py --sweep-root runs/b3 --out figs_demo/track_b3
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


@dataclass
class TrackSummary:
    config: str
    rmse: float
    mae: float
    phase_lag_s: float
    pearson_r: float
    n_samples: int


def _speed_profile(total_s: float, hold_s: float = 3.0) -> callable:
    """A reproducible ramp / hold / step / ramp profile bounded in [0.3, 2.0] m/s."""
    seg = total_s / 4.0

    def fn(t: float) -> float:
        if t < seg:
            return 0.3 + (1.2 - 0.3) * (t / seg)
        if t < 2 * seg:
            return 1.2
        if t < 3 * seg:
            return 1.8
        return 1.8 - (1.8 - 0.3) * ((t - 3 * seg) / seg)

    return fn


def _rollout_track(
    env: BipedEnv,
    model: PPO,
    episode_seconds: float,
    sample_every_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = env.cfg.dt
    max_steps = int(episode_seconds / dt)
    profile = _speed_profile(episode_seconds)
    obs, _ = env.reset(test_speed=profile(0.0), test_angle=0.0, demo_max_steps=max_steps)

    control_dt = dt * 10.0
    samples_per_window = max(1, int(round(sample_every_s / control_dt)))

    t_hist: list[float] = []
    cmd_hist: list[float] = []
    actual_hist: list[float] = []

    prev_pos = env.return_external_state()[1]
    cur_cmd = profile(0.0)
    env.change_ref_speed(cur_cmd)

    for i in range(max_steps // 10):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, done, _t, _info = env.step(action)
        sim_time = (i + 1) * control_dt

        # Update command on schedule.
        new_cmd = profile(sim_time)
        if abs(new_cmd - cur_cmd) > 1e-6:
            cur_cmd = new_cmd
            env.change_ref_speed(cur_cmd)

        if (i + 1) % samples_per_window == 0:
            cur_pos = env.return_external_state()[1]
            window_s = samples_per_window * control_dt
            actual_speed = (cur_pos - prev_pos) / max(window_s, 1e-6)
            prev_pos = cur_pos
            t_hist.append(sim_time)
            cmd_hist.append(cur_cmd)
            actual_hist.append(actual_speed)

        if done:
            break

    return np.asarray(t_hist), np.asarray(cmd_hist), np.asarray(actual_hist)


def _phase_lag(cmd: np.ndarray, actual: np.ndarray, dt: float) -> float:
    if cmd.size < 4 or actual.size < 4:
        return float("nan")
    n = min(cmd.size, actual.size)
    c = cmd[:n] - cmd[:n].mean()
    a = actual[:n] - actual[:n].mean()
    if np.allclose(c, 0) or np.allclose(a, 0):
        return float("nan")
    corr = np.correlate(a, c, mode="full")
    lags = np.arange(-n + 1, n)
    # Only consider physically plausible lags: within a quarter window.
    mask = np.abs(lags) <= n // 4
    corr_masked = corr[mask]
    lags_masked = lags[mask]
    if corr_masked.size == 0:
        return float("nan")
    best = int(lags_masked[int(np.argmax(corr_masked))])
    return float(best * dt)


def _pearson_r(cmd: np.ndarray, actual: np.ndarray) -> float:
    if cmd.size < 3 or actual.size < 3:
        return float("nan")
    n = min(cmd.size, actual.size)
    c = cmd[:n]
    a = actual[:n]
    if c.std() == 0 or a.std() == 0:
        return float("nan")
    return float(np.corrcoef(c, a)[0, 1])


def _write_traces(t: np.ndarray, cmd: np.ndarray, actual: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["t", "cmd_speed", "actual_speed", "error"])
        for i in range(t.size):
            w.writerow([t[i], cmd[i], actual[i], actual[i] - cmd[i]])


def _plot_overlay(t: np.ndarray, cmd: np.ndarray, actual: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, cmd, label="commanded", color="#4c72b0", linewidth=2)
    ax.plot(t, actual, label="actual", color="#c44e52", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Forward speed (m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _analyse(
    model_path: Path,
    cfg_path: Path,
    label: str,
    out_root: Path,
    episode_seconds: float,
    sample_every_s: float,
    include_pad_dims: bool | None = None,
) -> TrackSummary | None:
    cfg = load_run_config(cfg_path)
    if include_pad_dims is not None:
        cfg.env.include_pad_dims = bool(include_pad_dims)
    env = BipedEnv(config=cfg.env, demo_mode=True, demo_type="track")
    model = PPO.load(str(model_path), device="cpu")

    t, cmd, actual = _rollout_track(env, model, episode_seconds, sample_every_s)
    env.close()
    if t.size < 4:
        print(f"[track] {label}: fell before first sample, skipping")
        return None

    err = actual - cmd
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    lag = _phase_lag(cmd, actual, sample_every_s)
    r = _pearson_r(cmd, actual)

    _write_traces(t, cmd, actual, out_root / label / "track.csv")
    _plot_overlay(t, cmd, actual,
                  title=f"Velocity tracking -- {label} (RMSE={rmse:.3f} m/s, lag={lag:.2f}s)",
                  path=out_root / label / "track.png")
    return TrackSummary(label, rmse, mae, lag, r, int(t.size))


def _write_summary(rows: list[TrackSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["config", "rmse", "mae", "phase_lag_s", "pearson_r", "n_samples"])
        for r in rows:
            w.writerow([r.config, r.rmse, r.mae, r.phase_lag_s, r.pearson_r, r.n_samples])


def _parse_models(
    raw: list[str] | None,
    sweep_root: Path | None,
) -> list[tuple[str, Path, Path, bool | None]]:
    """Return a list of (label, model_path, config_path, include_pad_dims?)."""
    entries: list[tuple[str, Path, Path, bool | None]] = []
    if raw:
        for item in raw:
            parts = item.split(":")
            if len(parts) < 3:
                raise ValueError(
                    f"--models expects 'MODEL_DIR:LABEL:CONFIG_YAML[:58]', got {item!r}"
                )
            model_dir, label, cfg = parts[0], parts[1], parts[2]
            pad = bool(int(parts[3])) if len(parts) > 3 and parts[3].strip() == "58" else None
            if Path(model_dir).is_dir():
                mp = Path(model_dir) / "final_model.zip"
            else:
                mp = Path(model_dir)
            entries.append((label, mp, Path(cfg), True if pad else None))
    if sweep_root is not None:
        for rd in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
            mp = rd / "final_model.zip"
            cfg = rd / "config.yaml"
            if mp.exists() and cfg.exists():
                entries.append((rd.name, mp, cfg, None))
    return entries


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None,
                   help="MODEL_DIR:LABEL:CONFIG_YAML[:58] triples. The trailing :58 forces legacy padded obs.")
    p.add_argument("--sweep-root", type=Path, default=None,
                   help="Auto-discover (final_model.zip, config.yaml) pairs in each subfolder.")
    p.add_argument("--out", type=Path, default=Path("figs_demo/track"))
    p.add_argument("--episode-seconds", type=float, default=16.0)
    p.add_argument("--sample-every-s", type=float, default=0.5)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    jobs = _parse_models(args.models, args.sweep_root)
    if not jobs:
        print("[track] no jobs; supply --models or --sweep-root.")
        return 1

    rows: list[TrackSummary] = []
    for label, model_path, cfg_path, pad in jobs:
        print(f"[track] {label}  model={model_path}  cfg={cfg_path}")
        summary = _analyse(
            model_path=model_path,
            cfg_path=cfg_path,
            label=label,
            out_root=args.out,
            episode_seconds=args.episode_seconds,
            sample_every_s=args.sample_every_s,
            include_pad_dims=pad,
        )
        if summary is not None:
            rows.append(summary)
            print(f"[track]  -> RMSE={summary.rmse:.3f}  lag={summary.phase_lag_s:.2f}s  r={summary.pearson_r:.2f}")

    _write_summary(rows, args.out / "summary.csv")
    print(f"[track] wrote {args.out}/summary.csv and per-config outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
