## Bipedal Imitation RL

A research codebase for training and evaluating bipedal locomotion policies using imitation learning and reinforcement learning. It includes AMP (Adversarial Motion Prior) training, PPO and Recurrent PPO baselines, motion datasets, and visualization utilities built around PyBullet.

### Features
- AMP-based imitation learning utilities (`amp_implementation/`)
- PPO and RecurrentPPO training scripts and experiment outputs
- Motion capture dataset loaders (`dataset/`) and analysis notebooks
- PyBullet environments and URDF assets for a 2D biped (`assets/`)
- Demos for running trained policies and producing animations

### Requirements
- Python 3.9+ recommended
- See `requirements.txt` for the full list of dependencies

Install dependencies (preferably in a virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Quick Start
- Run a quick demo (visualize a trained policy if available):

```bash
python demo.py
```

- Visualize sample environments or utilities:

```bash
python demo_visualize.py
```

### Training
PPO/MLP or LSTM training entry points:

```bash
python train_mlp.py
python train_lstm.py
python train_mlp_vecenv.py
python train_lstm_vecenv.py
```

AMP training (adversarial motion prior):

```bash
python amp_implementation/train_amp.py
```

Experiment outputs are saved under folders like `ppo_*`, `ppo_lstm/`, `ppo_newreward/`, and `amp_runs_*`. You can inspect logs, checkpoints (`.zip`/`.pt`/`.pth`), and generated figures there.

### Data
- Motion datasets are under `dataset/` (C3D files) and various preprocessed `.npy` series under folders like `gait time series data/`.
- URDF and mesh assets for the biped and ground plane are in `assets/`.

### Notes
- Some notebooks (e.g., `analyse_csv.ipynb`, `fft_datacreate.ipynb`, `joint_comparison.ipynb`) provide analysis and plotting utilities.
- Configuration groups and experiment summaries are stored under `configurations/` and `arch_search/`.

### Troubleshooting
- Ensure a recent GPU-enabled PyTorch installation if you intend to train from scratch.
- For headless rendering on servers, configure EGL/OSMesa per your environment or disable GUI where supported.

### License
This repository is provided for research and educational purposes. If you plan to publish results based on it, please cite accordingly and include a link back to this repository.



### Reproducing the SIViP paper (revision branch `sivip-revision`)

Python: the conda base interpreter (`C:\Users\<you>\anaconda3\python.exe` on the
authors' machines). Run every command from the repository root.

| Paper item | Command | Output |
|---|---|---|
| Six-joint cycles with subject IDs (Section 3) | `python fft_extract_cycles.py --out-dir "gait reference 6joint"` | `gait reference 6joint/*.npy`, `meta.json` |
| Filter check, Section 3 numbers | `python analyse_cutoff.py` | `tables/cutoff_check.csv` |
| Cycle-duration table (Table 2) | `python analyse_cycle_durations.py` | `tables/cycle_durations.{csv,tex}` |
| Representation comparison (Table 3, Fig. 3) | `python analyse_gait_generators.py --out figs_demo/gait_gen_v2 --v2-variants residual` | `figs_demo/gait_gen_v2/summary.csv` |
| PCA and GRU baselines | `python train_latent_baselines.py` then `python analyse_gait_generators.py --out figs_demo/gait_gen_latent --v2-variants residual --extra pca_mlp gru_seq` | `kfold_results/latent_baselines/`, `figs_demo/gait_gen_latent/` |
| Representation ablation (cutoff, rate, bins) | `python ablate_representation.py` and `python ablate_representation.py K` | `kfold_results/ablation/*.csv` |
| Subject-grouped split | `python train_gaitgen_subject.py` and `python train_gaitgen_subject.py --seeds 0 1 2 3 4 5 6 7` | `kfold_results/subject_split/` |
| Closed-loop figure panels and Table 6 intervals | `python plot_paper_figures.py` | `sn-article-template/figs/fig_ramp_vel.pdf`, `tables/table5_ci.csv` |
| Manuscript | `latexmk -pdf main.tex` inside `sn-article-template/` | `main.pdf` |

The split indices behind Table 3 are `gait reference phase 2/split_indices.npz`
(897 / 135 / 89 cycles, seed 42). PPO configurations are the YAML files under
`configs/`; the per-episode demo logs used for Table 6 are under
`configurations/<config>/<run>/` and `ppo_newreward/`.
