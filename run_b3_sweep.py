"""Launcher for the B3 gait-generator sweep.

The rebuttal table promised one PPO run per gait-generator variant at five
seeds with otherwise identical hyper-parameters. This script materialises
that sweep on top of the declarative configs in ``configs/gen_*.yaml``.

Usage
-----
    # full-budget (paper): 4 generators x 5 seeds x 15M steps
    python run_b3_sweep.py --output-root runs/b3

    # smoke / CI: same harness, 200k steps, 1 seed, just to prove the
    # pipeline produces all outputs
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
import copy
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from biped_config import RunConfig, load_run_config, save_run_config


DEFAULT_GENERATORS: tuple[str, ...] = (
    "fft_mlp",
    "raw_mocap",
    "cubic_spline",
    "cpg_matsuoka",
)

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

BASE_CONFIG_TEMPLATE = "configs/gen_{name}.yaml"
TRAIN_SCRIPT = "train_mlp.py"


def _load_base(generator: str) -> RunConfig:
    path = Path(BASE_CONFIG_TEMPLATE.format(name=generator))
    if not path.exists():
        raise FileNotFoundError(
            f"Base YAML {path} missing. Add it to configs/ before sweeping."
        )
    return load_run_config(path)


def _materialise(
    generator: str,
    seed: int,
    total_timesteps: int | None,
    output_root: Path,
) -> Path:
    base = _load_base(generator)
    cfg = copy.deepcopy(base)
    cfg.name = f"{generator}_seed{seed}"
    cfg.description = f"B3 sweep: generator={generator}, seed={seed}."
    cfg.train.seed = seed
    if total_timesteps is not None:
        cfg.train.total_timesteps = int(total_timesteps)
        # Checkpoint at least twice per run so partial runs have a model to load.
        cfg.train.save_freq = max(1, min(cfg.train.save_freq, cfg.train.total_timesteps // 2))
    run_dir = output_root / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, run_dir / "config.yaml")
    return run_dir


def _launch(run_dir: Path, verbose: bool = True) -> int:
    log_path = run_dir / f"{run_dir.name}.log"
    cmd = [
        sys.executable,
        TRAIN_SCRIPT,
        "--config",
        str(run_dir / "config.yaml"),
        "--save-dir",
        str(run_dir),
    ]
    if verbose:
        print(f"[b3] launching {' '.join(cmd)}")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    if verbose:
        print(f"[b3] {run_dir.name}: {status} in {elapsed:.1f}s (log={log_path})")
    return proc.returncode


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B3 gait-generator sweep launcher")
    p.add_argument(
        "--generators",
        nargs="+",
        default=list(DEFAULT_GENERATORS),
        choices=[
            "fft_mlp",
            "raw_mocap",
            "cubic_spline",
            "cpg_matsuoka",
            "rnn",
            "amp_placeholder",
        ],
        help="Gait-generator variants to sweep.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds to sweep. Defaults to the paper-promised 5 seeds.",
    )
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help=(
            "Override cfg.train.total_timesteps for every run. Useful for "
            "smoke runs / CI. Omit to use the 15M value baked into configs/."
        ),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/b3"),
        help="Parent directory for all per-run folders.",
    )
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Materialise the YAML configs but do not call train_mlp.py.",
    )
    return p.parse_args()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args()
    root: Path = args.output_root
    root.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, int, Path]] = []
    for gen in args.generators:
        for seed in args.seeds:
            run_dir = _materialise(gen, seed, args.total_timesteps, root)
            pairs.append((gen, seed, run_dir))

    print(f"[b3] planned {len(pairs)} runs under {root}/")

    if args.plan_only:
        print("[b3] --plan-only set; YAML configs written. Exiting.")
        return 0

    failures: list[str] = []
    for gen, seed, run_dir in pairs:
        rc = _launch(run_dir)
        if rc != 0:
            failures.append(run_dir.name)

    if failures:
        print(f"[b3] FAILED runs: {failures}")
        return 1
    print("[b3] all runs complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
