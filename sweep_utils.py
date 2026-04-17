"""Shared helpers for per-seed sweep launchers (B3, C3, ...).

Both :mod:`run_b3_sweep` and :mod:`run_c3_sweep` follow the same pattern:

1. Load one or more base YAML configs.
2. For every (base, seed) pair, deepcopy the config, override the seed and
   (optionally) ``total_timesteps``, then write the resulting YAML into
   ``<output_root>/<run_name>/config.yaml``.
3. Invoke ``train_mlp.py --config <generated> --save-dir <run_dir>`` and
   capture stdout/stderr in ``<run_dir>/<run_name>.log``.

This module factors out steps (2) and (3) so individual sweep launchers are
short and declarative.
"""

from __future__ import annotations

import copy
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

from biped_config import RunConfig, load_run_config, save_run_config


TRAIN_SCRIPT = "train_mlp.py"


def materialise_run(
    base_cfg: RunConfig,
    run_name: str,
    description: str,
    seed: int,
    output_root: Path,
    total_timesteps: int | None,
) -> Path:
    cfg = copy.deepcopy(base_cfg)
    cfg.name = run_name
    cfg.description = description
    cfg.train.seed = seed
    if total_timesteps is not None:
        cfg.train.total_timesteps = int(total_timesteps)
        cfg.train.save_freq = max(
            1, min(cfg.train.save_freq, cfg.train.total_timesteps // 2)
        )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, run_dir / "config.yaml")
    return run_dir


def launch_training(run_dir: Path, verbose: bool = True) -> int:
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
        print(f"[sweep] launching {' '.join(cmd)}")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    if verbose:
        print(f"[sweep] {run_dir.name}: {status} in {elapsed:.1f}s (log={log_path})")
    return proc.returncode


def run_sweep(
    base_yaml_paths: Iterable[Path],
    seeds: Iterable[int],
    output_root: Path,
    total_timesteps: int | None,
    name_fn: Callable[[RunConfig, int, Path], str] | None = None,
    description_prefix: str = "sweep",
    plan_only: bool = False,
) -> list[str]:
    """Launch one run per (base_yaml, seed) pair. Returns the list of failed runs."""
    output_root.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, Path]] = []
    for yaml_path in base_yaml_paths:
        yaml_path = Path(yaml_path)
        base_cfg = load_run_config(yaml_path)
        for seed in seeds:
            run_name = (
                name_fn(base_cfg, seed, yaml_path)
                if name_fn is not None
                else f"{yaml_path.stem}_seed{seed}"
            )
            run_dir = materialise_run(
                base_cfg=base_cfg,
                run_name=run_name,
                description=f"{description_prefix}: {yaml_path.name} seed={seed}",
                seed=seed,
                output_root=output_root,
                total_timesteps=total_timesteps,
            )
            pairs.append((run_name, run_dir))

    print(f"[sweep] planned {len(pairs)} runs under {output_root}/")
    if plan_only:
        return []

    failures: list[str] = []
    for run_name, run_dir in pairs:
        rc = launch_training(run_dir)
        if rc != 0:
            failures.append(run_name)
    return failures


__all__ = ["materialise_run", "launch_training", "run_sweep"]
