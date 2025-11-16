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


