from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
from amp_biped_old import BipedEnv
from skrl.memories.torch import RandomMemory
from amp_models import PolicyMLP, ValueMLP, DiscriminatorMLP

import torch
import numpy as np


env = BipedEnv()

# ============================================================
# 2) LOAD MOTION DATASET (.npy)
# ============================================================
np_data = np.load(
    "gait time series data/window_data.npy"
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

# --- Experiment configuration for built-in checkpointing ---
cfg_agent["experiment"] = {
    "directory": "amp_runs_15m",            # experiment's parent directory
    "experiment_name": "amp_trial0",        # experiment name
    "write_interval": "auto",               # TensorBoard writing interval (timesteps)
    "checkpoint_interval": 500_000,       # interval for checkpoints (timesteps)
    "store_separately": False,              # whether to store checkpoints separately
    "wandb": False,                         # whether to use Weights & Biases
    "wandb_kwargs": {}                      # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
}


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
agent.load("/home/baran/Bipedal-imitation-rl/agent_1500000.pt")
from skrl.trainers.torch import SequentialTrainer
import os
import torch

# ============================================================
# 1. Configure trainer
# ============================================================
max_steps = 15_000_000
cfg = {
        "timesteps": max_steps,
        "headless": True,
        "print_every": 100_000,
        # === UPDATED tqdm_kwargs ===
        # Set both miniters and a long mininterval
        "tqdm_kwargs": {"miniters": 2000, "mininterval": 10.0} 
        # === END UPDATE ===
    }

trainer = SequentialTrainer(
    env=env,
    agents=agent,
    cfg=cfg
)

# ============================================================
# 2. Start training (checkpointing handled by agent config)
# ============================================================
print(f"Starting training for {max_steps:,} steps...")
print("Checkpoints will be saved every 500,000 steps to amp_runs_15m/amp_trial0/")
trainer.train()

print("Training completed!")


