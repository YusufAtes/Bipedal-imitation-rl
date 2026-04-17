"""Launcher for the C3 reward-component ablation sweep.

The four ``rgait_no_*`` YAMLs in ``configs/`` each zero out one component of
the reward function used by the paper's equation (5):

* ``rgait_no_alive``   -- drops the alive (posture) bonus
* ``rgait_no_contact`` -- drops the foot-contact consistency term
* ``rgait_no_speed``   -- drops the speed-tracking term
* ``rgait_no_torque``  -- drops the torque-minimisation penalty

This script trains each variant at three seeds using the same hyper-params
as the legacy Configuration 1 so reviewer-requested attribution ablations
can be run in one command.

Usage
-----
    # full-budget (paper): 4 ablations x 3 seeds x 15M steps
    python run_c3_sweep.py --output-root runs/c3

    # smoke / CI: 1 seed, 200k steps
    python run_c3_sweep.py --seeds 0 --total-timesteps 200000 \
        --output-root runs/c3_smoke

    # just materialise the per-seed configs
    python run_c3_sweep.py --plan-only --output-root runs/c3

Each run directory contains ``config.yaml``, logs and the trained model in
the same layout as :mod:`run_b3_sweep`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from biped_config import RunConfig
from sweep_utils import run_sweep


DEFAULT_ABLATIONS: tuple[str, ...] = (
    "rgait_no_alive",
    "rgait_no_contact",
    "rgait_no_speed",
    "rgait_no_torque",
)

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)


def _base_yaml(name: str) -> Path:
    path = Path("configs") / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Base YAML {path} missing.")
    return path


def _name_fn(_cfg: RunConfig, seed: int, yaml_path: Path) -> str:
    return f"{yaml_path.stem}_seed{seed}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C3 reward-component ablation sweep")
    p.add_argument(
        "--ablations", nargs="+", default=list(DEFAULT_ABLATIONS),
        choices=list(DEFAULT_ABLATIONS),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--output-root", type=Path, default=Path("runs/c3"))
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args()
    yaml_paths = [_base_yaml(n) for n in args.ablations]
    failures = run_sweep(
        base_yaml_paths=yaml_paths,
        seeds=args.seeds,
        output_root=args.output_root,
        total_timesteps=args.total_timesteps,
        name_fn=_name_fn,
        description_prefix="C3 reward ablation",
        plan_only=args.plan_only,
    )
    if failures:
        print(f"[c3] FAILED runs: {failures}")
        return 1
    print("[c3] all runs complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
