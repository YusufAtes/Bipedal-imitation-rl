# amp_models.py
import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


# =====================
# Policy (Gaussian)
# =====================

class PolicyMLP(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh(),  # keep if you want tanh-mean
        )

        # state-independent log std (float32 by default)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, dtype=torch.float32))

        # make sure model lives on the right device
        self.to(self.device)

    def compute(self, inputs, role):
        # SKRL expects {"states": tensor}, but be tolerant if we get nested dicts
        x = inputs["states"]
        if isinstance(x, dict):  # e.g. {"states": tensor}
            x = x["states"]

        # 1) move to same device
        if x.device != self.log_std_parameter.device:
            x = x.to(self.log_std_parameter.device)

        # 2) make dtype match (fixes: mat1 Double, mat2 Float)
        model_dtype = self.log_std_parameter.dtype
        if x.dtype != model_dtype:
            x = x.to(model_dtype)

        mean = self.net(x)
        return mean, self.log_std_parameter, {}



# =====================
# Value (Deterministic)
# =====================
class ValueMLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

        # >>> ensure all parameters/buffers are on the chosen device
        self.to(self.device)

    def compute(self, inputs, role):
        x = inputs["states"]
        
        # Get the device and dtype of the model's parameters
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        if x.device != model_device:
            x = x.to(model_device)
        
        # === ADDED THIS BLOCK FOR DTYPE ROBUSTNESS ===
        # Robustly cast input tensor to match model's dtype
        if x.dtype != model_dtype:
            x = x.to(model_dtype)
        # === END ADD ===
            
        return self.net(x), {}


# =====================
# Discriminator (Deterministic)
# =====================
class DiscriminatorMLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # === FIX ===
        # self.net now stops at the final Linear layer.
        # This is what SKRL's update loop expects.
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)  # <-- This is now the last module
        )
        
        # >>> ensure all parameters/buffers are on the chosen device
        self.to(self.device)

    def compute(self, inputs, role):
        # AMP sometimes passes "amp_states", sometimes "states"
        x = inputs.get("amp_states", inputs.get("states"))
        if x is None:
            raise RuntimeError("DiscriminatorMLP.compute expected 'amp_states' or 'states' in inputs")
        
        # Get the device and dtype of the model's parameters
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        if x.device != model_device:
            x = x.to(model_device)
        
        # === DTYPE ROBUSTNESS ===
        if x.dtype != model_dtype:
            x = x.to(model_dtype)
        # === END DTYPE ===
        
        # === FIX ===
        # 1. Pass input through the main network
        raw_logits = self.net(x)
        
        # 2. Apply the clipping activation manually
        clipped_logits = torch.clamp(raw_logits, -10.0, 10.0)
        return clipped_logits, {}
