# `review-suggestions` branch — summary & run-book

Structured, ordered list of what was added to the `review-suggestions`
branch and how to execute each piece to regenerate the results used in the
rebuttal. Every command assumes:

```powershell
git checkout review-suggestions
conda activate biped_env
```

and is run from the repo root. New YAMLs live in `configs/`; run artefacts
land under `runs/` or `figs_demo/` by convention.

---

## 1. Track A — Reproducibility & observation cleanup

### 1.1 Declarative config + parameterised `BipedEnv` (A1)

- `biped_config.py` — dataclass schemas (`BipedEnvConfig`, `PolicyConfig`,
  `TrainConfig`, `RunConfig`, `DomainRandomizationConfig`) + YAML I/O.
- `biped_env.py` — single canonical environment.
- `ppoenv_guide.py` — thin backward-compatible shim for legacy scripts.
- `configs/*.yaml` — one YAML per historical configuration.

Train any config:

```powershell
python train_mlp.py --config configs/config_025decay_mlp_rsi.yaml --save-dir runs/repro_025decay --seed 42
python train_lstm.py --config configs/config3_lstm_256_256.yaml --save-dir runs/repro_lstm_256 --seed 42
```

### 1.2 Reproducibility regression tests (A2, A3)

```powershell
python -m tests._run_dr_tests                           # DR plumbing (C5)
python tests\test_config_reproducibility.py             # structural invariants across all YAMLs
python tests\test_config_reproducibility.py --strict    # optional golden-trace mode
python tests\test_observation_dim.py                    # 55-D default / 58-D legacy pad
python tests\test_reward_components.py                  # per-component reward logger (A4)
python tests\test_gait_generators.py                    # gait generator smoke tests (B2)
```

### 1.3 Per-component reward logger (A4)

Automatically written as `reward_components.csv` alongside `rewards.csv`
inside every `--save-dir` when training through `train_mlp.py` /
`train_lstm.py`. No separate command — just inspect the CSV after training.

---

## 2. Track B — Gait generator suite

### 2.1 Stand-alone gait-generator evaluation (B1)

```powershell
python analyse_gait_generators.py --out figs_demo/gait_gen
```

Emits per-speed MSE, DTW, FFT-fidelity plots and `gait_generator_summary.csv`.

### 2.2 Pluggable gait generators (B2)

Selectable via `env.gait_generator:` in YAML. Available:
`fft_mlp`, `raw_mocap`, `cubic_spline`, `cpg_matsuoka`, `rnn`,
`amp_placeholder`. Pre-built YAMLs live in `configs/gen_*.yaml`.

### 2.3 5-seed PPO sweep over gait generators (B3)

```powershell
python run_b3_sweep.py --total-timesteps 15000000 --output-root runs/b3
python run_b3_sweep.py --total-timesteps 200000 --seeds 0 --plan-only    # dry-run
```

### 2.4 Evaluate sweep + build comparison tables/plots

```powershell
python evaluate_b3_sweep.py --sweep-root runs/b3 --out runs/b3_eval --group-by generator --title "B3 gait generators"
```

Produces `per_seed.csv`, `summary.csv` (mean + 95% bootstrap CI),
`comparison.png`.

### 2.5 FFT analysis of learned PPO joint trajectories (B4)

```powershell
# On a legacy 58-D model
python analyse_ppo_fft.py --model-path configurations/025decay_mlp_rsi/PPO_1/final_model.zip --config configs/config_025decay_mlp_rsi.yaml --include-pad-dims --out figs_demo/ppo_fft/alpha0.25

# On any new 55-D model
python analyse_ppo_fft.py --model-path runs/b3/fft_mlp_seed0/final_model.zip --config runs/b3/fft_mlp_seed0/config.yaml --out figs_demo/ppo_fft/fft_mlp_seed0
```

---

## 3. Track C — Rebuttal-ready evaluation

### 3.1 Per-demo tables with 95% bootstrap CI + LaTeX exporter (C1)

```powershell
python plot_demo_csv_comparison.py --tables --out-tables tables/
python plot_demo_csv_comparison.py --tables --skip-plots --out-tables tables/ --config-dirs configurations/025decay_mlp_rsi/PPO_1 configurations/nodecay_mlp_rsi/PPO_39
```

Writes CSV, Markdown, and `booktabs` LaTeX for the `vel_diff`, `rotation`,
`noisy`, and `track` demos.

### 3.2 Velocity-tracking demo (RMSE, MAE, Pearson r, phase lag) (C2)

```powershell
# Explicit configs
python track_demo.py --out figs_demo/track --inputs "configurations/025decay_mlp_rsi/PPO_1:alpha0.25:configs/config_025decay_mlp_rsi.yaml:58" "configurations/nodecay_mlp_rsi/PPO_39:nodecay:configs/config1_nodecay_mlp_rsi.yaml:58"

# Auto-discover from a sweep root
python track_demo.py --sweep-root runs/b3 --out figs_demo/track_b3
```

Emits per-config `track.csv`, `track.png` overlay, plus aggregate
`summary.csv`.

### 3.3 Reward-component ablations (C3)

```powershell
python run_c3_sweep.py --total-timesteps 15000000 --output-root runs/c3
python evaluate_b3_sweep.py --sweep-root runs/c3 --out runs/c3_eval --group-by variant --title "C3 reward ablations"
```

Four `rgait_no_*` YAMLs zero out alive / contact / speed / torque weights
respectively, now at 5 seeds by default (see C6).

### 3.4 Cost of Transport + per-joint Symmetry Index (C4)

```powershell
# Single legacy model
python analyse_cot_si.py --model-path configurations/025decay_mlp_rsi/PPO_1/final_model.zip --config configs/config_025decay_mlp_rsi.yaml --include-pad-dims --out figs_demo/cot_si/alpha0.25

# Entire sweep
python analyse_cot_si.py --sweep-root runs/b3 --out figs_demo/cot_si/b3
```

Emits `cot_si_per_speed.csv`, `cot_si_summary.csv`,
`cost_of_transport.png`, `symmetry_by_joint.png`.

### 3.5 Domain randomization + push-recovery demo (C5)

Train with DR:

```powershell
python train_mlp.py --config configs/config1_with_dr.yaml --save-dir runs/config1_dr --seed 42
```

Push-recovery rollouts:

```powershell
python push_recovery_demo.py --model-path runs/config1_dr/final_model.zip --config configs/config1_with_dr.yaml --out figs_demo/push_recovery/config1_dr

# On a legacy (no-DR) policy
python push_recovery_demo.py --model-path configurations/025decay_mlp_rsi/PPO_1/final_model.zip --config configs/config_025decay_mlp_rsi.yaml --include-pad-dims --out figs_demo/push_recovery/alpha0.25
```

Produces `push_recovery_trials.csv`, `push_recovery_summary.csv`,
`push_recovery.png` (success rate vs impulse magnitude per direction).

### 3.6 5 seeds + paired bootstrap / Welch tests (C6)

- Sweeps default to 5 seeds (`run_b3_sweep.py`, `run_c3_sweep.py`).
- `stats_utils.py` provides `mean_ci`, `paired_bootstrap`, `welch_ttest`,
  `pairwise_table` (no SciPy dependency; cross-checked against SciPy to
  machine precision).
- Two new notebook cells at the bottom of `analyse_csv.ipynb`:
  `summarise_per_seed` and `pairwise_from_per_seed`.

```powershell
python -m tests._check_stats                    # sanity-check helpers
jupyter notebook analyse_csv.ipynb              # run the last two cells after a sweep
```

Inside the notebook:

```python
from stats_utils import mean_ci, pairwise_table
summary = summarise_per_seed("runs/b3_eval/per_seed.csv")           # mean +- 95% CI
table = pairwise_from_per_seed("runs/b3_eval/per_seed.csv",
                               metric="vel_err_mse_mean",
                               baseline="fft_mlp")                  # boot + Welch
```

---

## 4. End-to-end pipeline to regenerate everything

```powershell
# 1. sanity checks
python -m tests._run_dr_tests
python -m tests._check_stats
python tests\test_gait_generators.py
python tests\test_observation_dim.py
python tests\test_reward_components.py

# 2. gait generator isolation study (B1)
python analyse_gait_generators.py --out figs_demo/gait_gen

# 3. train the big sweeps (B3 + C3) -- overnight / cluster
python run_b3_sweep.py --total-timesteps 15000000 --output-root runs/b3
python run_c3_sweep.py --total-timesteps 15000000 --output-root runs/c3
python train_mlp.py --config configs/config1_with_dr.yaml --save-dir runs/config1_dr --seed 42  # for C5

# 4. per-sweep evaluation (B3 / C3)
python evaluate_b3_sweep.py --sweep-root runs/b3 --out runs/b3_eval --group-by generator --title "B3 gait generators"
python evaluate_b3_sweep.py --sweep-root runs/c3 --out runs/c3_eval --group-by variant   --title "C3 reward ablations"

# 5. per-policy analyses
python analyse_ppo_fft.py --sweep-root runs/b3 --out figs_demo/ppo_fft_b3     # add --include-pad-dims for legacy 58-D models
python analyse_cot_si.py  --sweep-root runs/b3 --out figs_demo/cot_si/b3
python track_demo.py      --sweep-root runs/b3 --out figs_demo/track/b3
python push_recovery_demo.py --model-path runs/config1_dr/final_model.zip --config configs/config1_with_dr.yaml --out figs_demo/push_recovery/config1_dr

# 6. rebuttal tables (C1)
python plot_demo_csv_comparison.py --tables --out-tables tables/

# 7. stats (C6)
jupyter notebook analyse_csv.ipynb     # run the final two cells
```

Each step writes self-contained CSVs / PNGs / LaTeX under `runs/*` or
`figs_demo/*`, ready to drop into the rebuttal.
