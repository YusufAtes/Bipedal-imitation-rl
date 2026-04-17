"""Launcher for the B3 gait-generator sweep.

The rebuttal table promised one PPO run per gait-generator variant at five
seeds with otherwise identical hyper-parameters. This script materialises
that sweep on top of the declarative configs in ``configs/gen_*.yaml`` using
the shared :mod:`sweep_utils` helper.

Usage
-----
    # full-budget (paper): 4 generators x 5 seeds x 15M steps
    python run_b3_sweep.py --output-root runs/b3

    # smoke / CI: same harness, 200k steps, 1 seed
    python run_b3_sweep.py --generators fft_mlp cubic_spline \
        --seeds 0 --total-timesteps 200000 \
        --output-root runs/b3_smoke

    # only generate the per-seed YAML configs, do not launch training
    python run_b3_sweep.py --plan-only --output-root runs/b3

Each generated run lands in ``<output_root>/<generator>_seed<N>/`` with:

* ``config.yaml``         effective RunConfig (base + overrides)
* ``<gen>_seed<N>.log``   stdout/stderr from ``train_mlp.py``
* ``rewards.csv``, ``reward_components.csv``
* ``final_model.zip``     final PPO policy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from biped_config import RunConfig
from sweep_utils import run_sweep


DEFAULT_GENERATORS: tuple[str, ...] = (
    "fft_mlp",
    "raw_mocap",
    "cubic_spline",
    "cpg_matsuoka",
)

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def _base_yaml(generator: str) -> Path:
    path = Path("configs") / f"gen_{generator}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Base YAML {path} missing.")
    return path


def _name_fn(_cfg: RunConfig, seed: int, yaml_path: Path) -> str:
    stem = yaml_path.stem  # e.g. "gen_fft_mlp"
    generator = stem.removeprefix("gen_")
    return f"{generator}_seed{seed}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B3 gait-generator sweep launcher")
    p.add_argument(
        "--generators",
        nargs="+",
        default=list(DEFAULT_GENERATORS),
        choices=[
            "fft_mlp", "raw_mocap", "cubic_spline",
            "cpg_matsuoka", "rnn", "amp_placeholder",
        ],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--output-root", type=Path, default=Path("runs/b3"))
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args()
    yaml_paths = [_base_yaml(g) for g in args.generators]
    failures = run_sweep(
        base_yaml_paths=yaml_paths,
        seeds=args.seeds,
        output_root=args.output_root,
        total_timesteps=args.total_timesteps,
        name_fn=_name_fn,
        description_prefix="B3 sweep",
        plan_only=args.plan_only,
    )
    if failures:
        print(f"[b3] FAILED runs: {failures}")
        return 1
    print("[b3] all runs complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
