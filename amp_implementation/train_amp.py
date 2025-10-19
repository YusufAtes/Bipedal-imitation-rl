from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
from amp_biped import BipedEnv
from skrl.memories.torch import RandomMemory
from amp_models import PolicyMLP, ValueMLP, DiscriminatorMLP

import torch
import numpy as np


# ============================================================
# 1) ENVIRONMENT SETUP
# ============================================================
# BipedEnv must implement the standard Gym API:
#   obs, info = env.reset()
#   obs, reward, terminated, truncated, info = env.step(action)
#
# Additionally, for AMP it must provide:
#   - env.device : torch.device (e.g. "cuda")
#   - env.num_envs : int (1 for single-env training)
#   - env.observation_space : gym.spaces.Box(...) with shape=(56,)
#       -> the state used by policy/value networks
#   - env.action_space : gym.spaces.Box(...) with shape=(7,)
#   - env.amp_observation_space : tuple or Box(shape=(60,))
#       -> 10×6 joint-position history flattened (for AMP discriminator)
#
#   - env.collect_observation() : callable returning torch.Tensor (num_envs, 60)
#       -> current AMP observation (10×6 window flattened)
#       -> normalized using REF_MEAN / REF_STD below
#
#   - internally maintain:
#       env.amp_q_hist : torch.Tensor (num_envs, 10, 6)
#       -> rolling buffer updated every step with latest joint positions
#
# Example update in env.step():
#   self.amp_q_hist = torch.roll(self.amp_q_hist, shifts=-1, dims=1)
#   self.amp_q_hist[:, -1, :] = current_joint_positions
#
# These ensure AMP’s discriminator sees the right sequence data.
# ============================================================

env = BipedEnv()

# ============================================================
# 2) LOAD MOTION DATASET (.npy)
# ============================================================
np_data = np.load(
    "/home/baran/Bipedal-imitation-rl/gait time series data/window_data.npy"
).astype(np.float32)  # (2262, 10, 6)
N, T, J = np_data.shape
assert (T, J) == (50, 6), f"Expected (50,6), got {(T, J)}"
K = T * J

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REF_X = torch.from_numpy(np_data.reshape(N, K)).to(device)  # (2262, 60)
REF_MEAN = REF_X.mean(dim=0, keepdim=True)
REF_STD = REF_X.std(dim=0, keepdim=True).clamp_min(1e-6)
REF_X = (REF_X - REF_MEAN) / REF_STD

# sample random AMP reference windows when AMP requests them
def collect_reference_motions(num_samples: int) -> torch.Tensor:
    idx = torch.randint(0, REF_X.shape[0], (num_samples,), device=device)
    return REF_X.index_select(0, idx).contiguous()


from skrl.memories.torch import RandomMemory

# REF_X: (N, 300) normalized tensor already on device
motion_dataset = RandomMemory(memory_size=REF_X.shape[0], num_envs=1, device=env.device)
motion_dataset.create_tensor(name="states", size=300, dtype=torch.float32)

for chunk in REF_X.split(2048, dim=0):
    motion_dataset.add_samples(states=chunk)

# ============================================================
# 3) MEMORIES
# ============================================================
# On-policy rollout memory: used by PPO/AMP for trajectory storage
# Must match env.num_envs and reside on the same device
memory = RandomMemory(memory_size=2048, num_envs=1, device=env.device)

# Replay buffer of policy-generated AMP observations:
# used to stabilize discriminator training
reply_buffer = RandomMemory(memory_size=50000, num_envs=1, device=env.device)
reply_buffer.create_tensor(name="amp_states", size=K, dtype=torch.float32)

# Optional helper to push to buffer each step:
# (Call this in your training loop after collecting AMP obs)
def push_to_reply_buffer(amp_states: torch.Tensor):
    reply_buffer.add_samples(amp_states=amp_states.detach())


# ============================================================
# 4) MODELS
# ============================================================
models = {
    "policy":        PolicyMLP(observation_space=56, action_space=7, device=device),    
    "value":         ValueMLP(observation_space=56, action_space=None,device=device),
    "discriminator": DiscriminatorMLP(observation_space=300,action_space=None, device=device),  # <- was 60
}

# --- Sanity checks (run before agent = AMP(...)) ---
obs, _ = env.reset()
print("RL obs from reset:", type(obs), getattr(obs, "shape", None))     # must be (1, 56)
assert hasattr(obs, "shape") and obs.shape == (1, 56), f"reset returned {obs.shape}, expected (1, 56)"

amp_obs = env.collect_observation()
print("AMP obs from collect_observation:", amp_obs.shape)               # must be (num_envs, 300)
assert amp_obs.shape == (getattr(env, "num_envs", 1), 300), f"got {amp_obs.shape}"

# Try one policy forward with a fake batch of RL obs
import torch
obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)  # already (1,56)
m, ls, _ = models["policy"].compute({"states": obs_t}, role="policy")
print("Policy forward OK:", m.shape, ls.shape)  # (1,7), (7,)



# ============================================================
# 5) AGENT CONFIGURATION
# ============================================================
cfg_agent = AMP_DEFAULT_CONFIG.copy()

# --- make sure policy uses the 56-D RL observation, not AMP (300-D) ---
for k in [
    "use_amp_observation_as_state",
    "use_amp_observation_as_states",
    "amp_observation_as_state",
    "observe_amp_states",
    "use_amp_states_as_observations"
]:
    if k in cfg_agent:
        cfg_agent[k] = False

# --- AMP discriminator & dataset sampling ---
cfg_agent["discriminator_batch_size"] = 1024    # policy vs. reference mix
cfg_agent["amp_batch_size"] = 1024              # balanced for N≈2262
cfg_agent["discriminator_updates"] = 2          # 2 discriminator steps / iteration

# --- Regularization (prevents over-powerful discriminator) ---
cfg_agent["discriminator_logit_regularization_scale"] = 0.05
cfg_agent["discriminator_gradient_penalty_scale"] = 5.0
cfg_agent["discriminator_weight_decay_scale"] = 1e-4

# --- Reward weighting (style vs. task) ---
cfg_agent["style_reward_weight"] = 1.0          # strong imitation term
cfg_agent["task_reward_weight"]  = 0.3          # forward-speed / stability shaping

# --- optional debug print to confirm which key actually exists in your skrl version ---
print({k: cfg_agent.get(k, None) for k in [
    "use_amp_observation_as_state",
    "use_amp_observation_as_states",
    "amp_observation_as_state",
    "observe_amp_states",
    "use_amp_states_as_observations"
]})




# ============================================================
# 6) AGENT INSTANTIATION
# ============================================================
# === agent instantiation ===
agent = AMP(
    models=models,
    memory=memory,
    cfg=cfg_agent,
    observation_space=env.observation_space,      # (56,)
    action_space=env.action_space,                # (7,)
    device=env.device,
    amp_observation_space=env.amp_observation_space,  # (300,)

    # IMPORTANT:
    motion_dataset=motion_dataset,   # see B) below
    reply_buffer=reply_buffer,

    collect_reference_motions=collect_reference_motions,

    # DO NOT PASS collect_observation here! This is what was overriding your RL state.
    # collect_observation=env.collect_observation,
)

from skrl.trainers.torch import SequentialTrainer
from torch.utils.tensorboard import SummaryWriter
import os
import csv
import torch

# ============================================================
# 1. Output directories
# ============================================================
output_dir = "amp_runs_15m"
os.makedirs(output_dir, exist_ok=True)

checkpoints_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoints_dir, exist_ok=True)

tensorboard_dir = os.path.join(output_dir, "tensorboard")
os.makedirs(tensorboard_dir, exist_ok=True)

csv_log_path = os.path.join(output_dir, "train_log.csv")

# # ============================================================
# # 2. CSV + TensorBoard setup
# # ============================================================
# csv_file = open(csv_log_path, mode='w', newline='')
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(['timestep', 'episode', 'reward'])

# writer = SummaryWriter(log_dir=tensorboard_dir)

# ============================================================
# 3. Checkpoint saving is handled by CheckpointCallback class below
# ============================================================

# ============================================================
# 4. Configure StepTrainer (no 'log_dir' argument!)
# ============================================================
max_steps = 15_000_000
cfg_trainer = {
    "timesteps": max_steps,
    "headless": False,
    "print_every": 10_000,
}
cfg = {
    "timesteps": 15_000_000, 
    "headless": False,
    "print_every": 10_000
}
trainer = SequentialTrainer(
    env=env,
    agents=agent,
    cfg=cfg
)

# ============================================================
# 5. Training with callbacks
# ============================================================
checkpoint_steps = 500_000

# Custom callback for saving checkpoints
class CheckpointCallback:
    def __init__(self, agent, checkpoint_steps):
        self.agent = agent
        self.checkpoint_steps = checkpoint_steps
    
    def __call__(self, trainer):
        if trainer.total_timesteps > 0 and trainer.total_timesteps % self.checkpoint_steps == 0:
            ckpt_k = trainer.total_timesteps // 1000
            torch.save(self.agent.models["policy"].state_dict(), os.path.join(checkpoints_dir, f"policy_{ckpt_k}k.pt"))
            torch.save(self.agent.models["value"].state_dict(), os.path.join(checkpoints_dir, f"value_{ckpt_k}k.pt"))
            torch.save(self.agent.models["discriminator"].state_dict(), os.path.join(checkpoints_dir, f"discriminator_{ckpt_k}k.pt"))
            print(f"[Checkpoint] Saved weights at step {trainer.total_timesteps:,}")

callback = CheckpointCallback(agent, checkpoint_steps)
trainer.callbacks = [callback]

print(f"Starting training for {max_steps:,} steps...")
trainer.train()

# ============================================================
# 6. Clean up
# ============================================================
print(f"Training completed. Outputs in: {output_dir}")


