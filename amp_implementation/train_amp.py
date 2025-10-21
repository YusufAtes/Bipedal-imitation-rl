# --- begin patch ---
import numpy as np
import torch
from skrl.agents.torch.amp import AMP as AMP_Base

class AMP_Patched(AMP_Base):
    # keep signature compatible with skrl
    def record_transition(self,
                          states,
                          actions,
                          rewards,
                          next_states,
                          terminated,
                          truncated,
                          infos,
                          *args, **kwargs):
        # Vectorized envs may give infos as:
        #  (A) dict of arrays (gymnasium vectorized collate), or
        #  (B) list of dicts (per-env info)
        if isinstance(infos, dict) and "amp_obs" in infos:
            v = infos["amp_obs"]
            if isinstance(v, np.ndarray):
                infos = dict(infos)
                infos["amp_obs"] = torch.as_tensor(v, dtype=torch.float32, device=self.device)
        elif isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict) and "amp_obs" in infos[0]:
            # stack per-env arrays -> (num_envs, 300)
            v = np.stack([d["amp_obs"] for d in infos]).astype(np.float32, copy=False)
            infos = {"amp_obs": torch.as_tensor(v, dtype=torch.float32, device=self.device)}
        return super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, *args, **kwargs)
# --- end patch ---



# === FIX 1: SET MULTIPROCESSING START METHOD ===
import multiprocessing as mp
# Set the start method to 'spawn' BEFORE importing torch or creating env
# This is crucial for CUDA + multiprocessing on Linux
try:
    mp.set_start_method('spawn', force=True)
    print("Multiprocessing start method set to 'spawn'.")
except RuntimeError:
    print("Multiprocessing start method already set.")
# === END FIX 1 ===

from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
from amp_biped import BipedEnv
from skrl.memories.torch import RandomMemory
from amp_models import PolicyMLP, ValueMLP, DiscriminatorMLP

import torch
import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import os

# === Use the wrap_env import that works for your skrl version ===
# Assuming this is the correct path based on your working script
from skrl.envs.wrappers.torch import wrap_env

# Helper function to create a single env instance
def make_env():
    # Ensure you are using the amp_biped.py with lazy loading fix
    return BipedEnv()

# This guard is CRITICAL for multiprocessing
if __name__ == "__main__":

    # ============================================================
    # 1) ENVIRONMENT SETUP
    # ============================================================
    NUM_ENVS = 14

    print("Creating temporary env to get observation spaces...")
    temp_env = BipedEnv()
    amp_observation_space = temp_env.amp_observation_space
    observation_space = temp_env.observation_space
    action_space = temp_env.action_space
    temp_env.close()
    print("Observation spaces obtained.")

    env_fns = [make_env for _ in range(NUM_ENVS)]

    print("Starting AsyncVectorEnv... (This may take a moment)")
    env = AsyncVectorEnv(env_fns)
    print("AsyncVectorEnv created.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Use wrap_env WITHOUT device argument ===
    print("Wrapping environment...")
    env = wrap_env(env)
    # === END FIX ===
    print("Environment wrapped.")

    # ============================================================
    # 2) LOAD MOTION DATASET (.npy)
    # ============================================================
    print("Loading motion dataset...")
    motion_data_path = "gait time series data/window_data.npy"
    if not os.path.exists(motion_data_path):
        print(f"FATAL ERROR: Cannot find motion data at {motion_data_path}")
        print(f"Current working directory is: {os.getcwd()}")
        exit()

    np_data = np.load(motion_data_path).astype(np.float32)

    N, T, J = np_data.shape
    assert (T, J) == (50, 6), f"Expected (50,6), got {(T, J)}"
    K = T * J # 300

    REF_X = torch.from_numpy(np_data.reshape(N, K)).to(device)
    REF_MEAN = REF_X.mean(dim=0, keepdim=True)
    REF_STD = REF_X.std(dim=0, keepdim=True).clamp_min(1e-6)
    REF_X = (REF_X - REF_MEAN) / REF_STD
    print("Motion dataset loaded and normalized.")

    def collect_reference_motions(num_samples: int) -> torch.Tensor:
        idx = torch.randint(0, REF_X.shape[0], (num_samples,), device=device)
        return REF_X.index_select(0, idx).contiguous()

    # === FIX: Correct motion_dataset setup ===
    # Create motion dataset as a single pool (num_envs=1)
    # Use the device specified earlier
    motion_dataset = RandomMemory(memory_size=REF_X.shape[0], num_envs=1, device=device)
    motion_dataset.create_tensor(name="states", size=300, dtype=torch.float32)

    # Add the entire reference data at once
    motion_dataset.add_samples(states=REF_X)
    # === END FIX ===
    print(f"Filled motion dataset with {motion_dataset.memory_size} samples")

    # ============================================================
    # 3) MEMORIES
    # ============================================================
    print("Creating memories...")
    ROLLOUT_STEPS = 2048
    # These memories need the correct device from the wrapped env
    memory = RandomMemory(memory_size=ROLLOUT_STEPS, num_envs=NUM_ENVS, device=device)
    reply_buffer = RandomMemory(memory_size=50_000, num_envs=NUM_ENVS, device=device)
    reply_buffer.create_tensor(name="amp_states", size=K, dtype=torch.float32)
    print("Memories created.")

    # ============================================================
    # 4) MODELS
    # ============================================================
    print("Creating models...")
    models = {
        "policy":        PolicyMLP(observation_space=observation_space.shape[0], action_space=action_space.shape[0], device=device),
        "value":         ValueMLP(observation_space=observation_space.shape[0], action_space=None, device=device),
        "discriminator": DiscriminatorMLP(observation_space=amp_observation_space.shape[0], action_space=None, device=device),
    }
    print("Models created.")

    # --- Sanity checks ---
    print("Running sanity checks (this will call env.reset)...")
    obs, info = env.reset()
    print(f"  Reset obs shape: {obs.shape}") # (14, 56)
    assert obs.shape == (NUM_ENVS, observation_space.shape[0])

    m, ls, _ = models["policy"].compute({"states": obs}, role="policy")
    print(f"  Policy forward OK: {m.shape}, {ls.shape}")  # (14, 7), (7,)
    assert m.shape == (NUM_ENVS, action_space.shape[0])
    print("Sanity checks passed!")


    # ============================================================
    # 5) AGENT CONFIGURATION
    # ============================================================
    print("Configuring agent...")
    cfg_agent = AMP_DEFAULT_CONFIG.copy()

    for k in [
        "use_amp_observation_as_state", "use_amp_observation_as_states",
        "amp_observation_as_state", "observe_amp_states",
        "use_amp_states_as_observations"
    ]:
        if k in cfg_agent:
            cfg_agent[k] = False

    cfg_agent["rollouts"] = ROLLOUT_STEPS
    cfg_agent["learning_epochs"] = 10
    cfg_agent["mini_batches"] = 16

    cfg_agent["discriminator_batch_size"] = 1024
    cfg_agent["amp_batch_size"] = 512
    cfg_agent["discriminator_updates"] = 2

    cfg_agent["discriminator_logit_regularization_scale"] = 0.05
    cfg_agent["discriminator_gradient_penalty_scale"] = 5.0
    cfg_agent["discriminator_weight_decay_scale"] = 1e-4

    cfg_agent["style_reward_weight"] = 1.0
    cfg_agent["task_reward_weight"]  = 0.3

    cfg_agent["experiment"] = {
        "directory": "amp_runs_local_rtx3070",
        "experiment_name": "amp_trial_local",
        "write_interval": "auto",
        "checkpoint_interval": 500_000,
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {}
    }
    print("Agent configured.")

    # ============================================================
    # 6) AGENT INSTANTIATION
    # ============================================================
    print("Instantiating agent...")
    agent = AMP_Patched(
        models=models,
        memory=memory,
        cfg=cfg_agent,
        observation_space=observation_space,
        action_space=action_space,
        device=device, # Use the main device defined earlier
        amp_observation_space=amp_observation_space,
        motion_dataset=motion_dataset, # Now correctly configured
        reply_buffer=reply_buffer,
        collect_reference_motions=collect_reference_motions,
    )
    print("Agent instantiated.")

    # -- in train_amp.py (after you instantiate the agent) --
    import numpy as np
    import torch

    # after: agent = AMP(models=models, memory=memory, cfg=cfg_agent, ...)
    def _wrap_amp_record_transition(agent):
        orig = agent.record_transition
        def wrapped(*args, **kwargs):
            # skrl passes infos as a dict
            infos = kwargs.get("infos", None)
            if isinstance(infos, dict) and "amp_obs" in infos:
                v = infos["amp_obs"]
                # AsyncVectorEnv usually gives a (num_envs, 300) NumPy batch here
                if isinstance(v, np.ndarray):
                    infos = dict(infos)  # shallow copy
                    infos["amp_obs"] = torch.as_tensor(v, dtype=torch.float32, device=agent.device)
                    kwargs["infos"] = infos
            return orig(*args, **kwargs)
        agent.record_transition = wrapped.__get__(agent, agent.__class__)

    _wrap_amp_record_transition(agent)
    print("AMP record transition wrapped.")
    from skrl.trainers.torch import SequentialTrainer

    # ============================================================
    # 1. Configure trainer
    # ============================================================
    print("Configuring trainer...")
    max_steps = 15_000_000
    cfg = {
        "timesteps": max_steps,
        "headless": True,
        "print_every": 100_000
    }

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg=cfg
    )
    print("Trainer configured.")

    # ============================================================
    # 2. Start training
    # ============================================================
    print("="*80)
    print(f"STARTING TRAINING for {max_steps:,} steps...")
    print(f"Running with {NUM_ENVS} parallel environments on an RTX 3070.")
    print(f"Checkpoints: {cfg_agent['experiment']['directory']}/{cfg_agent['experiment']['experiment_name']}")
    print("="*80)

    trainer.train()

    print("Training completed!")