# B3 gait-generator sweep

The full sweep is **4 generators x 5 seeds x 15M timesteps** of PPO on top of
`configs/gen_{fft_mlp,raw_mocap,cubic_spline,cpg_matsuoka}.yaml`.

## Full-budget launch (paper)

```
python run_b3_sweep.py --output-root runs/b3
python evaluate_b3_sweep.py --sweep-root runs/b3 --out figs_demo/b3
```

`run_b3_sweep.py` writes one subfolder `runs/b3/<generator>_seed<N>/` per run,
each containing the effective `config.yaml`, PPO logs, `rewards.csv`,
`reward_components.csv`, intermediate checkpoints and `final_model.zip`.

`evaluate_b3_sweep.py` then loads every `final_model.zip`, runs 3 rollouts at
9 commanded speeds (0.3..2.0 m/s), and aggregates:

| metric              | definition                                                            |
| ------------------- | --------------------------------------------------------------------- |
| `velocity_mse`      | MSE(cmd_speed, achieved_speed) averaged over successful rollouts     |
| `success_rate`      | fraction of rollouts that did not early-terminate                     |
| `cost_of_transport` | `info['cost_of_transport']` at episode end, averaged                  |
| `travel_range_m`    | mean forward distance covered by successful rollouts                  |
| `symmetry_index`    | Robinson's SI averaged over hip/knee/ankle joints                     |

Output lands in `figs_demo/b3/`:

- `per_seed.csv` - one row per (generator, seed) with the 5 metrics
- `summary.csv`  - mean and 95% bootstrap CI per generator (n=5 seeds)
- `comparison.png` - bar charts of the 5 headline metrics

## Smoke run (CI / sanity check)

The contents of this folder are the smoke-run artefacts from a **single** 16k-
timestep training run of `fft_mlp` at seed 0. It exists purely to prove the
harness is wired end-to-end (training -> evaluation -> CSV + PNG). A 16k-step
policy falls instantly, so `success_rate == 0` and the other metrics are NaN
by construction - expected behaviour, not a bug.

Reproduce with:

```
python run_b3_sweep.py --generators fft_mlp --seeds 0 \
    --total-timesteps 16384 --output-root runs/b3_smoke
python evaluate_b3_sweep.py --sweep-root runs/b3_smoke \
    --out figs_demo/b3_smoke --speeds 0.5 1.0 1.5 \
    --trials-per-speed 2 --episode-seconds 2.0
```
