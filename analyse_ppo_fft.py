"""FFT analysis of PPO-learned joint trajectories vs. mocap reference.

The paper's central framing — "the gait generator lives in the frequency
domain" — is never quantitatively verified against the trained policy. This
script closes that loop by:

1. Rolling out a trained PPO policy at a grid of commanded speeds,
2. Recording per-joint time-domain trajectories,
3. Computing single-sided magnitude spectra (FFT) of both the learned
   trajectory and the corresponding mocap-reference trajectory,
4. Reporting per-speed summary metrics:
   - fundamental frequency ``f0_policy`` and ``f0_mocap``
   - relative error of the fundamental ``delta_f0 = |f0_p - f0_m| / f0_m``
   - spectral centroid error
   - spectral magnitude MAE over the first ``--n-harmonics`` harmonics
   - signal-to-reference ratio (log10 power ratio)

Outputs:

* ``spectra/<joint>_spd<speed>.png``  per-joint spectrum overlay
* ``time_domain/<joint>_spd<speed>.png`` matching time-domain overlay
* ``ppo_fft_summary.csv``  aggregate table suitable for a paper appendix

Notes
-----
The mocap reference trajectory is produced by
:class:`gait_generators.raw_mocap.RawMocapGenerator`. Its absolute time
scaling depends on the ``period`` array in ``gait reference phase 2/``.
If the dataset's period is expressed in a non-seconds unit, the reference's
``f0_mocap`` values will be off by that same factor - the overall shape and
harmonic pattern remain valid, but readers should interpret absolute Hz
values with that caveat in mind.

Usage
-----
    python analyse_ppo_fft.py --model-path configurations/025decay_mlp_rsi/PPO_1/final_model.zip \
        --config configs/config_025decay_mlp_rsi.yaml \
        --out figs_demo/ppo_fft/025decay

    # multiple models at once, one subfolder per model
    python analyse_ppo_fft.py \
        --sweep-root runs/b3 \
        --out figs_demo/ppo_fft/b3
"""

from __future__ import annotations

import argparse
import csv
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
from gait_generators.raw_mocap import RawMocapGenerator


JOINT_NAMES = ("rhip", "rknee", "rankle", "lhip", "lknee", "lankle")
# 55-D layout indices; legacy 58-D padded layout shifts by +3.
JOINT_OBS_IDX_55 = (4, 5, 6, 7, 8, 9)
JOINT_OBS_IDX_58 = (7, 8, 9, 10, 11, 12)


def _joint_indices(include_pad_dims: bool) -> tuple[int, ...]:
    return JOINT_OBS_IDX_58 if include_pad_dims else JOINT_OBS_IDX_55


@dataclass
class Spectrum:
    freqs: np.ndarray
    mag: np.ndarray

    @property
    def fundamental(self) -> float:
        if self.mag.size <= 1:
            return 0.0
        # Drop DC.
        k = int(np.argmax(self.mag[1:])) + 1
        return float(self.freqs[k])

    @property
    def centroid(self) -> float:
        m = self.mag.copy()
        denom = float(m.sum())
        if denom <= 0.0:
            return 0.0
        return float((self.freqs * m).sum() / denom)


def _compute_spectrum(signal: np.ndarray, dt: float) -> Spectrum:
    sig = np.asarray(signal, dtype=np.float64)
    if sig.size < 4:
        return Spectrum(freqs=np.array([0.0]), mag=np.array([0.0]))
    sig = sig - sig.mean()
    n = sig.size
    w = np.hanning(n)
    X = np.fft.rfft(sig * w)
    mag = 2.0 / n * np.abs(X)
    freqs = np.fft.rfftfreq(n, d=dt)
    return Spectrum(freqs=freqs, mag=mag)


def _rollout_policy(
    env: BipedEnv,
    model: PPO,
    speed: float,
    max_steps: int,
    joint_idx: tuple[int, ...],
) -> np.ndarray | None:
    obs, _ = env.reset(test_speed=speed, test_angle=0.0, demo_max_steps=max_steps)
    traces = np.zeros((max_steps // 10, len(joint_idx)), dtype=np.float64)
    n = 0
    for _ in range(max_steps // 10):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, done, _t, _info = env.step(action)
        for j, idx in enumerate(joint_idx):
            traces[n, j] = float(obs[idx])
        n += 1
        if done:
            break
    if n < 32:
        return None
    return traces[:n]


def _mocap_reference_trace(
    mocap: RawMocapGenerator, speed: float, dt: float, n_samples: int
) -> np.ndarray:
    """Nearest-speed mocap reference, resampled to ``n_samples`` rows."""
    traj = mocap.predict(speed=speed, leg_lengths=[0.94, 0.94])  # (len, 6)
    if traj.shape[0] < n_samples:
        reps = int(np.ceil(n_samples / traj.shape[0]))
        traj = np.tile(traj, (reps, 1))
    return traj[:n_samples]


def _spectral_mag_mae(a: Spectrum, b: Spectrum, n_harmonics: int) -> float:
    if a.mag.size == 0 or b.mag.size == 0:
        return float("nan")
    k_max = min(n_harmonics + 1, a.mag.size, b.mag.size)
    am = a.mag[:k_max]
    bm = b.mag[:k_max]
    norm_a = am / max(am.max(), 1e-9)
    norm_b = bm / max(bm.max(), 1e-9)
    return float(np.mean(np.abs(norm_a - norm_b)))


def _log_power_ratio(a: Spectrum, b: Spectrum) -> float:
    pa = float((a.mag ** 2).sum())
    pb = float((b.mag ** 2).sum())
    if pa <= 0 or pb <= 0:
        return float("nan")
    return float(np.log10(pa / pb))


@dataclass
class SpeedReport:
    speed: float
    joint: str
    f0_policy: float
    f0_mocap: float
    delta_f0_rel: float
    centroid_policy: float
    centroid_mocap: float
    centroid_abs_err: float
    spectral_mag_mae: float
    log_power_ratio: float


def _plot_overlay(
    freqs_p: np.ndarray, mag_p: np.ndarray,
    freqs_m: np.ndarray, mag_m: np.ndarray,
    title: str, path: Path, xlabel: str, ylabel: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(freqs_p, mag_p, label="PPO policy", color="#c44e52")
    ax.plot(freqs_m, mag_m, label="Mocap reference", color="#4c72b0", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_time_domain(
    t_p: np.ndarray, sig_p: np.ndarray,
    t_m: np.ndarray, sig_m: np.ndarray,
    title: str, path: Path,
) -> None:
    _plot_overlay(t_p, sig_p, t_m, sig_m, title, path,
                  xlabel="Time (s)", ylabel="Angle (rad)")


def _analyse_model(
    model_path: Path,
    cfg_path: Path,
    speeds: np.ndarray,
    out_root: Path,
    n_harmonics: int,
    episode_seconds: float,
    include_pad_dims: bool | None = None,
) -> list[SpeedReport]:
    cfg = load_run_config(cfg_path)
    if include_pad_dims is not None:
        cfg.env.include_pad_dims = bool(include_pad_dims)
    env = BipedEnv(config=cfg.env, demo_mode=True, demo_type="vel_diff")
    model = PPO.load(str(model_path), device="cpu")
    # Each env.step advances the simulator by _CONTROL_SUBSTEPS=10 physics
    # substeps of length ``cfg.env.dt``, so the recorded trajectory is sampled
    # at 10 * cfg.env.dt seconds per frame. Compare mocap at the same rate.
    effective_dt = cfg.env.dt * 10.0
    mocap = RawMocapGenerator(dt=effective_dt, tile_repeats=1)
    max_steps = int(episode_seconds / cfg.env.dt)
    joint_idx = _joint_indices(cfg.env.include_pad_dims)

    reports: list[SpeedReport] = []
    for spd in speeds:
        traces = _rollout_policy(env, model, float(spd), max_steps, joint_idx)
        if traces is None:
            print(f"[ppo-fft] {out_root.name} spd={spd}: policy fell, skipping")
            continue
        t_policy = np.arange(traces.shape[0]) * effective_dt
        mocap_trace = _mocap_reference_trace(
            mocap, float(spd), effective_dt, traces.shape[0]
        )
        t_mocap = np.arange(mocap_trace.shape[0]) * effective_dt

        for j, joint in enumerate(JOINT_NAMES):
            sp = _compute_spectrum(traces[:, j], effective_dt)
            sm = _compute_spectrum(mocap_trace[:, j], effective_dt)

            rep = SpeedReport(
                speed=float(spd),
                joint=joint,
                f0_policy=sp.fundamental,
                f0_mocap=sm.fundamental,
                delta_f0_rel=(abs(sp.fundamental - sm.fundamental) / sm.fundamental)
                if sm.fundamental > 1e-6 else float("nan"),
                centroid_policy=sp.centroid,
                centroid_mocap=sm.centroid,
                centroid_abs_err=abs(sp.centroid - sm.centroid),
                spectral_mag_mae=_spectral_mag_mae(sp, sm, n_harmonics=n_harmonics),
                log_power_ratio=_log_power_ratio(sp, sm),
            )
            reports.append(rep)

            _plot_overlay(
                sp.freqs, sp.mag, sm.freqs, sm.mag,
                title=f"{joint} spectrum @ speed={spd:.2f} m/s",
                path=out_root / "spectra" / f"{joint}_spd{spd:.2f}.png",
                xlabel="Frequency (Hz)", ylabel="Magnitude",
            )
            _plot_time_domain(
                t_policy, traces[:, j], t_mocap, mocap_trace[:, j],
                title=f"{joint} trajectory @ speed={spd:.2f} m/s",
                path=out_root / "time_domain" / f"{joint}_spd{spd:.2f}.png",
            )
    env.close()
    return reports


def _write_reports(reports: list[SpeedReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([
            "speed", "joint", "f0_policy", "f0_mocap", "delta_f0_rel",
            "centroid_policy", "centroid_mocap", "centroid_abs_err",
            "spectral_mag_mae", "log_power_ratio",
        ])
        for r in reports:
            w.writerow([
                r.speed, r.joint, r.f0_policy, r.f0_mocap, r.delta_f0_rel,
                r.centroid_policy, r.centroid_mocap, r.centroid_abs_err,
                r.spectral_mag_mae, r.log_power_ratio,
            ])


def _find_pairs(sweep_root: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for rd in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        model = rd / "final_model.zip"
        cfg = rd / "config.yaml"
        if model.exists() and cfg.exists():
            pairs.append((rd.name, model, cfg))
    return pairs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--sweep-root", type=Path, default=None,
                   help="If given, analyse every subfolder containing final_model.zip + config.yaml.")
    p.add_argument("--out", type=Path, default=Path("figs_demo/ppo_fft"))
    p.add_argument("--speeds", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    p.add_argument("--episode-seconds", type=float, default=4.0)
    p.add_argument("--n-harmonics", type=int, default=5)
    p.add_argument(
        "--include-pad-dims",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force 58-D padded obs (legacy snapshots) or 55-D (paper). None=use YAML value.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    speeds = np.asarray(args.speeds, dtype=np.float64)

    jobs: list[tuple[str, Path, Path]]
    if args.sweep_root is not None:
        jobs = _find_pairs(args.sweep_root)
        if not jobs:
            print(f"[ppo-fft] no (final_model.zip + config.yaml) pairs under {args.sweep_root}")
            return 1
    else:
        if args.model_path is None or args.config is None:
            print("[ppo-fft] supply either --sweep-root or (--model-path and --config).")
            return 2
        jobs = [(args.model_path.parent.name, args.model_path, args.config)]

    all_reports: list[SpeedReport] = []
    for name, model_path, cfg_path in jobs:
        out_root = args.out / name
        print(f"[ppo-fft] analysing {name} -> {out_root}")
        reports = _analyse_model(
            model_path=model_path,
            cfg_path=cfg_path,
            speeds=speeds,
            out_root=out_root,
            n_harmonics=args.n_harmonics,
            episode_seconds=args.episode_seconds,
            include_pad_dims=args.include_pad_dims,
        )
        _write_reports(reports, out_root / "ppo_fft_summary.csv")
        for rep in reports:
            all_reports.append(rep)
        print(f"[ppo-fft]  ... {len(reports)} rows")

    if args.sweep_root is not None:
        _write_reports(all_reports, args.out / "ppo_fft_summary_all.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
