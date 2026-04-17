"""Reproducibility regression test (A2).

For every ``configs/*.yaml`` we:

1. Instantiate :class:`biped_env.BipedEnv` from the YAML.
2. Reset deterministically and step for :data:`N_STEPS` with a fixed-seed
   random-torque policy.
3. Assert structural invariants:
   - Observation dim matches ``cfg.env.obs_dim()``.
   - ``info["reward_components"]`` contains all expected keys.
   - The trace is finite.

Optional strict mode (``--strict``) additionally compares each trace against
a golden ``.npy`` stored next to the YAML. PyBullet on Windows is not
bit-exact across processes, so strict mode should only be enabled in a
reproducible environment (Linux CI). Strict mode can be regenerated with
``--regenerate-golden``.

Running
-------

    python tests/test_config_reproducibility.py                # structural
    python tests/test_config_reproducibility.py --strict       # with golden
    python tests/test_config_reproducibility.py --regenerate-golden
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biped_config import load_run_config  # noqa: E402
from biped_env import BipedEnv  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
N_STEPS = 50
RNG_SEED = 123
STRICT_ATOL = 1.0  # loose; only catches observation-layout bugs
EXPECTED_COMPONENTS = {
    "alive",
    "contact",
    "speed",
    "torque",
    "im_hip_pos",
    "im_knee_pos",
    "im_ankle_pos",
    "im_hip_vel",
    "im_knee_vel",
    "im_ankle_vel",
}


def collect_trace(config_path: Path) -> tuple[np.ndarray, dict[str, object]]:
    cfg = load_run_config(config_path)
    env = BipedEnv(config=cfg.env, demo_mode=True)
    env.total_train_steps = cfg.train.total_timesteps
    obs, _ = env.reset(seed=RNG_SEED, test_speed=1.0, test_angle=0.0)
    rng = np.random.default_rng(RNG_SEED)
    obs_history = [obs.astype(np.float32).copy()]
    info: dict[str, object] = {}
    for _ in range(N_STEPS):
        torques = rng.uniform(-0.3, 0.3, size=env.action_space.shape).astype(np.float32)
        obs, _, done, truncated, info = env.step(torques)
        obs_history.append(obs.astype(np.float32).copy())
        if done or truncated:
            break
    expected_dim = cfg.env.obs_dim()
    env.close()
    return np.asarray(obs_history, dtype=np.float32), {
        "expected_dim": expected_dim,
        "last_info": info,
    }


def _golden_path(config_path: Path) -> Path:
    return config_path.with_suffix(".golden.npy")


def regenerate(config_dir: Path) -> None:
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        trace, _ = collect_trace(yaml_path)
        out = _golden_path(yaml_path)
        np.save(out, trace)
        print(f"[regen] {yaml_path.name} -> {out.name}  shape={trace.shape}")


def verify(config_dir: Path, strict: bool) -> int:
    failures = 0
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        trace, meta = collect_trace(yaml_path)
        name = yaml_path.name
        dim = trace.shape[1]
        if dim != meta["expected_dim"]:
            print(f"[FAIL] {name}: obs dim {dim} != expected {meta['expected_dim']}")
            failures += 1
            continue
        if not np.all(np.isfinite(trace)):
            print(f"[FAIL] {name}: non-finite values in trace")
            failures += 1
            continue
        comps = (meta["last_info"] or {}).get("reward_components", {})
        missing = EXPECTED_COMPONENTS - set(comps.keys())
        if missing:
            print(f"[FAIL] {name}: reward_components missing keys {missing}")
            failures += 1
            continue
        msg = f"[ok]   {name}: obs_dim={dim} steps={trace.shape[0]}"
        if strict:
            golden = _golden_path(yaml_path)
            if not golden.exists():
                msg += "  [strict: no golden, skipping]"
            else:
                expected = np.load(golden)
                common = min(expected.shape[0], trace.shape[0])
                max_diff = float(
                    np.max(np.abs(trace[:common] - expected[:common]))
                )
                if max_diff > STRICT_ATOL:
                    print(
                        f"[FAIL] {name}: strict |delta| = {max_diff:.2e} > {STRICT_ATOL}"
                    )
                    failures += 1
                    continue
                msg += f"  strict|d|={max_diff:.2e}"
        print(msg)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate-golden", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--configs-dir", type=str, default=str(REPO / "configs"),
    )
    args = parser.parse_args()
    config_dir = Path(args.configs_dir)
    if not config_dir.exists():
        print(f"configs dir does not exist: {config_dir}")
        return 2
    if args.regenerate_golden:
        regenerate(config_dir)
        return 0
    failures = verify(config_dir, strict=args.strict)
    if failures:
        print(f"\n{failures} config(s) failed the regression.")
        return 1
    print("\nAll configs pass structural invariants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
