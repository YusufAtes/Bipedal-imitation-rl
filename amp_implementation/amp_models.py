# # amp_models.py
# # Three SKRL-compatible networks:
# #  - PolicyNet: Gaussian policy (56 → 256 → 256 → 7)
# #  - ValueNet:  critic / value (56 → 256 → 256 → 1)
# #  - DiscriminatorNet: AMP discriminator (60 → 256 → 256 → 1)
# #
# # Hidden activations: ReLU
# # Policy output: linear (no tanh, matches SB3 default)
# # Set SB3_TANH_MEAN = True if you want tanh on the policy mean.

# from __future__ import annotations
# import torch
# import torch.nn as nn
# from typing import Any, Dict, Optional, Tuple, Union

# from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

# # ====== Config ======
# POLICY_INPUT_DIM = 56
# VALUE_INPUT_DIM = 56
# DISCRIM_INPUT_DIM = 300     # 50 × 6 AMP sequence flattened
# HIDDEN_DIM = 256
# ACTION_DIM = 7

# SB3_TANH_MEAN = False  # change to True if you want tanh on policy mean


# # =========================
# # Shared 2-layer MLP block
# # =========================
# class MLPBackbone(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int, hidden: int = HIDDEN_DIM):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, out_dim),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)


# # =========================
# # Policy network
# # =========================
# class PolicyNet(GaussianMixin, Model):
#     def __init__(
#         self,
#         observation_space: Union[int, Tuple[int, ...]] = POLICY_INPUT_DIM,
#         action_space: Union[int, Tuple[int, ...]] = ACTION_DIM,
#         device: Optional[torch.device] = None,
#         min_log_std: float = -5.0,
#         max_log_std: float = 2.0,
#         clip_actions: bool = True,
#         clip_log_std: bool = True,
#     ):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(
#             self,
#             clip_actions=clip_actions,
#             clip_log_std=clip_log_std,
#             min_log_std=min_log_std,
#             max_log_std=max_log_std,
#         )

#         self.backbone = MLPBackbone(self.num_observations, HIDDEN_DIM)
#         self.mean_head = nn.Linear(HIDDEN_DIM, self.num_actions)
#         self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

#         self._use_tanh = SB3_TANH_MEAN
#         self._tanh = nn.Tanh()

#     def compute(
#         self, inputs: Dict[str, torch.Tensor], role: str
#     ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
#         x = inputs["states"]
#         h = self.backbone(x)
#         mean = self.mean_head(h)
#         if self._use_tanh:
#             mean = self._tanh(mean)
#         return mean, self.log_std_parameter, {}


# # =========================
# # Value network
# # =========================
# class ValueNet(DeterministicMixin, Model):
#     def __init__(
#         self,
#         observation_space: Union[int, Tuple[int, ...]] = VALUE_INPUT_DIM,
#         device: Optional[torch.device] = None,
#     ):
#         Model.__init__(self, observation_space, action_space=None, device=device)
#         self.v_mlp = MLPBackbone(self.num_observations, 1)

#     def compute(
#         self, inputs: Dict[str, torch.Tensor], role: str
#     ) -> Tuple[torch.Tensor, Dict[str, Any]]:
#         v = self.v_mlp(inputs["states"])
#         return v, {}


# # =========================
# # Discriminator network
# # =========================
# class DiscriminatorNet(DeterministicMixin, Model):
#     """Discriminator for AMP: input = flattened (50×6) joint-pos window."""

#     def __init__(
#         self,
#         observation_space: Union[int, Tuple[int, ...]] = DISCRIM_INPUT_DIM,
#         device: Optional[torch.device] = None,
#     ):
#         Model.__init__(self, observation_space, action_space=None, device=device)
#         self.d_mlp = MLPBackbone(self.num_observations, 1)

#     def compute(
#         self, inputs: Dict[str, torch.Tensor], role: str
#     ) -> Tuple[torch.Tensor, Dict[str, Any]]:
#         # AMP sometimes passes "amp_states" instead of "states"
#         x = inputs.get("states", None)
#         if x is None:
#             x = inputs.get("amp_states")
#         logits = self.d_mlp(x)
#         return logits, {}



# amp_models.py
import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


# =====================
# Policy (Gaussian)
# =====================
class PolicyMLP(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh()  # keep if you want tanh at the mean; remove if not desired
        )

        # state-independent log std
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # >>> ensure all parameters/buffers are on the chosen device
        self.to(self.device)

    def compute(self, inputs, role):
        x = inputs["states"]
        # move inputs to the same device as the module (prevents CPU/CUDA mismatch)
        if x.device != self.log_std_parameter.device:
            x = x.to(self.log_std_parameter.device)
        return self.net(x), self.log_std_parameter, {}


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
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        return self.net(x), {}


# =====================
# Discriminator (Deterministic)
# =====================
class DiscriminatorMLP(DeterministicMixin, Model):
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
        # AMP sometimes passes "amp_states", sometimes "states"
        x = inputs.get("amp_states", inputs.get("states"))
        if x is None:
            raise RuntimeError("DiscriminatorMLP.compute expected 'amp_states' or 'states' in inputs")
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        return self.net(x), {}


# # instantiate the model (assumes there is a wrapped environment: env)
# critic = MLP(observation_space=env.observation_space,
#              action_space=env.action_space,
#              device=env.device,
#              clip_actions=False)


# # instantiate the model (assumes there is a wrapped environment: env)
# policy = MLP(observation_space=env.observation_space,
#              action_space=env.action_space,
#              device=env.device,
#              clip_actions=True,
#              clip_log_std=True,
#              min_log_std=-20,
#              max_log_std=2,
#              reduction="sum")