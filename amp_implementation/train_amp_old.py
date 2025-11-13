
from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
from amp_biped_old import BipedEnv
from skrl.memories.torch import RandomMemory
from amp_models import PolicyMLP, ValueMLP, DiscriminatorMLP

import torch
import numpy as np


env = BipedEnv()

savefolder = "amp_trial3"
max_steps = 3_000_000
save_interval = 500_000
# ============================================================
# 2) LOAD MOTION DATASET (.npy)
# ============================================================
np_data = np.load("gait time series data/window_data.npy").astype(np.float32)
assert np.isfinite(np_data).all(), "NaNs in motion dataset"
N, T, J = np_data.shape
assert (T, J) == (50, 6), f"Expected (50,6), got {(T, J)}"
K = T * J


device = env.device
# use `device` for REF_X, models, memories

REF_X = torch.from_numpy(np_data.reshape(N, K)).to(device)  # (2262, 300)
REF_MEAN = REF_X.mean(dim=0, keepdim=True)
REF_STD = REF_X.std(dim=0, keepdim=True).clamp_min(1e-6)
REF_X = (REF_X - REF_MEAN) / REF_STD

# Pass the mean and std to the env for normalization of live observations
# This is crucial for the discriminator
env.amp_mean = REF_MEAN
env.amp_std = REF_STD

# sample random AMP reference windows when AMP requests them
def collect_reference_motions(num_samples: int) -> torch.Tensor:
    idx = torch.randint(0, REF_X.shape[0], (num_samples,), device=device)
    return REF_X.index_select(0, idx).contiguous()


from skrl.memories.torch import RandomMemory

motion_dataset = RandomMemory(memory_size=REF_X.shape[0], num_envs=1, device=env.device)
motion_dataset.create_tensor(name="states",     size=K, dtype=torch.float32)
motion_dataset.create_tensor(name="amp_states", size=K, dtype=torch.float32)

for i in range(REF_X.shape[0]):
    row = REF_X[i].unsqueeze(0)
    motion_dataset.add_samples(states=row)
    motion_dataset.add_samples(amp_states=row)

# ============================================================
# 3) MEMORIES
# ============================================================
# On-policy rollout memory: used by PPO/AMP for trajectory storage
# Must match env.num_envs and reside on the same device

reply_buffer = RandomMemory(memory_size=50_000, num_envs=1, device=env.device)
reply_buffer.create_tensor(name="states", size=300, dtype=torch.float32)  # <-- not "amp_states"

# ============================================================
# 4) MODELS
# ============================================================
models = {
    "policy":        PolicyMLP(observation_space=56, action_space=7, device=device),    
    "value":         ValueMLP(observation_space=56, action_space=None,device=device),
    # === FIX: Set discriminator input to K (300) ===
    "discriminator": DiscriminatorMLP(observation_space=K, action_space=None, device=device),
    # === END FIX ===
}

# --- Sanity checks (run before agent = AMP(...)) ---
obs, _ = env.reset()
print("RL obs from reset:", type(obs), getattr(obs, "shape", None))     # must be (1, 56)
assert hasattr(obs, "shape") and obs.shape == (1, 56), f"reset returned {obs.shape}, expected (1, 56)"

amp_obs = env.collect_observation()
print("AMP obs from collect_observation:", amp_obs.shape)               # must be (num_envs, 300)
# === FIX: Assert the correct shape K (60) ===
assert amp_obs.shape == (getattr(env, "num_envs", 1), K), f"got {amp_obs.shape}, expected (1, {K})"
# === END FIX ===


# Try one policy forward with a fake batch of RL obs
import torch
obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)  # already (1,56)
m, ls, _ = models["policy"].compute({"states": obs_t}, role="policy")
print("Policy forward OK:", m.shape, ls.shape)  # (1,7), (7,)



# ============================================================
# 5) AGENT CONFIGURATION
# ============================================================
K = 300                     # 50 x 6 AMP observation
NUM_ENVS = 1                # your env is single-instance
MOTION_N = 5000             # you said 5k datapoints

# ==== 1) On-policy memory MUST match rollouts ====
cfg_agent = AMP_DEFAULT_CONFIG.copy()


cfg_agent.update({
    # rollout / update cadence
    "rollouts": 64,                 # collect 64 steps then update
    "learning_epochs": 5,           # PPO-style passes per update
    "mini_batches": 4,              # splits of the on-policy batch

    # optimization
    "learning_rate": 1e-4,          # AMP default is conservative; keep it
    "grad_norm_clip": 1.0,          # prevent spikes

    # style vs task
    "style_reward_weight": 0.25,     
    "task_reward_weight": 1.0,      

    # discriminator training
    "discriminator_batch_size": 512,           # logits on (on-policy + replay) vs motion
    "amp_batch_size": 512,                     # new motion windows per update
    "discriminator_updates": 1,                # one D step per PPO step (your class uses this key)

    # discriminator regularization (good defaults)
    "discriminator_logit_regularization_scale": 0.05,
    "discriminator_gradient_penalty_scale": 5.0,
    "discriminator_weight_decay_scale": 1e-4,

    # misc
    "random_timesteps": 0,
    "learning_starts": 0,
    "mixed_precision": False,                  # make debugging simpler & deterministic

    # logging
    "experiment": {
        "directory": "amp_runs_15m",
        "experiment_name": savefolder,
        "write_interval": "auto",
        "checkpoint_interval": save_interval,
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {}
    }
})

cfg_agent["entropy_loss_scale"] = 0.001   # small but non-zero
cfg_agent["grad_norm_clip"] = 1.0


# Keep batches modest; too big => too strong D early on
cfg_agent["discriminator_batch_size"] = 256
cfg_agent["amp_batch_size"] = 256
cfg_agent["discriminator_updates"] = 1

# Regularization
cfg_agent["discriminator_logit_regularization_scale"] = 0.05
cfg_agent["discriminator_gradient_penalty_scale"] = 1.0   # was 5.0, lower it
cfg_agent["discriminator_weight_decay_scale"] = 0.0



memory = RandomMemory(memory_size=cfg_agent["rollouts"], num_envs=1, device=env.device)

# ============================================================
# 6) AGENT INSTANTIATION
# ============================================================
# === agent instantiation ===
# before agent = AMP(...)

agent = AMP(
    models=models,
    memory=memory,
    cfg=cfg_agent,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
    amp_observation_space=env.amp_observation_space,

    motion_dataset=motion_dataset,
    reply_buffer=reply_buffer,     # <- RESTORE, plain empty buffer is fine

    collect_reference_motions=collect_reference_motions,
)

# ✅ ADD THIS: sanity check right after init
for name, p in agent.policy.named_parameters():
    if not torch.isfinite(p).all():
        raise RuntimeError(f"Policy param {name} is non-finite right after init!")

# Right before training, verify a sample is finite
batch = motion_dataset.sample(names=["amp_states"], batch_size=256)
# assert torch.isfinite(batch).all(), "NaNs in motion_dataset amp_states!"
print("-------------------------------- ")
print(batch)
print("--------------------------------")

print("Motion dataset tensors:", motion_dataset.tensors)   # must contain "states" -> (N, 300)
print("Replay buffer tensors:", reply_buffer.tensors)      # must contain "states" -> (0, 300) initially
print("AMP memory tensors:", agent.memory.tensors)         # must contain "amp_states" -> (rollouts, 300)

# After the first update, replay size should grow:
# (you can put this print inside a hook after a few steps)
print("Replay len (after 1st update):", len(reply_buffer))





from skrl.trainers.torch import SequentialTrainer
import os
import torch

# ============================================================
# 1. Configure trainer
# ============================================================
cfg = {
        "timesteps": max_steps,
        "headless": True,
        # "print_every": 100_000,
        # # === UPDATED tqdm_kwargs ===
        # # Set both miniters and a long mininterval
        # "tqdm_kwargs": {"miniters": 2000, "mininterval": 10.0},
        "experiment_name": savefolder,
        # === END UPDATE ===
    }

trainer = SequentialTrainer(
    env=env,
    agents=agent,
    cfg=cfg
)   

# ============================================================
# 2. Start training (checkpointing handled by agent config)
#============================================================
print(f"Starting training for {max_steps:,} steps...")
print(f"AMP observation dimension K = {K}")
print(f"Checkpoints will be saved every {save_interval:,} steps to amp_runs_15m/{savefolder}/")
trainer.train()

# print("Training completed!")
