"""Parameterised BipedEnv used by every config in ``configs/*.yaml``.

This module is the refactored counterpart of the historical, scratch-copy
``ppoenv_guide.py``. It implements a single code path that reproduces every
row of Table 3 from the paper by toggling switches declared in
:mod:`biped_config`. The physics, control frequency, observation ordering,
reward formulation and RSI behaviour all match the frozen snapshot
``configurations/nodecay_mlp_rsi/PPO_39/params.txt`` (Configuration 1).

Deviations from the historical snapshots, each guarded by an explicit
:class:`BipedEnvConfig` flag:

* ``include_pad_dims=False`` (default) drops the three legacy zero pads at
  ``state[2:5]``. Observation becomes 55-D to match paper Table 1 literally.
* ``reward="no_im"`` zeroes the imitation reward, reproducing
  ``noimreward_mlp``.
* ``observation="no_im_state"`` zeroes the reference-preview slice of the
  observation, reproducing ``nostate_mlp``.
* ``decay_schedule="linear_step"`` with ``decay_alpha>0`` enables the paper's
  equation (3) schedule, reproducing ``025decay_mlp_rsi`` / ``05decay_mlp_rsi``.
* ``rsi=False`` reproduces ``nodecay_mlp_norsi``.
* ``dr.enabled=True`` enables Track C5's domain randomization; off by default.

Every :meth:`step` call now also populates ``self.last_reward_components`` —
a dict consumed by :class:`utils.RewardComponentLoggerCallback` (A4) so
per-term contributions can be attributed after training.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as pyb
import pybullet_data
import torch
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import resample

from biped_config import BipedEnvConfig
from gait_generator_net import OldSimpleFCNN


_LEG_LEN_DEFAULT = 0.94
_MAX_TORQUE = np.array([500, 500, 300, 150, 500, 300, 150], dtype=np.float64)
_POS_NORMCOEFF = float(np.pi)
_VEL_NORMCOEFF = 10.0
_CONTROL_SUBSTEPS = 10
_LEFT_FOOT_LINK = 8
_RIGHT_FOOT_LINK = 5


class BipedEnv(gym.Env):
    """Torque-controlled 7-DoF planar biped with swappable gait generator.

    Parameters
    ----------
    config:
        Declarative run configuration. When ``None`` a paper-accurate default
        (Configuration 1: full reward, full state, RSI, no decay, pad dims
        dropped) is used.
    render_mode:
        ``'human'`` opens a PyBullet GUI. Anything else falls back to the
        headless ``DIRECT`` connection.
    demo_mode / demo_type:
        Kept for backward compatibility with :mod:`demo`; do not affect
        training.
    gait_generator:
        Optional object exposing ``predict(speed, leg_lengths)`` returning a
        ``(T, 6)`` trajectory. When ``None`` the legacy FFT MLP is loaded. This
        hook powers Track B2 (swappable gait generator).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: BipedEnvConfig | None = None,
        render_mode: str | None = None,
        demo_mode: bool = False,
        demo_type: str | None = None,
        gait_generator: Any = None,
    ) -> None:
        self.cfg = config if config is not None else BipedEnvConfig()
        self.p = pyb
        self.step_counter = 0
        self.total_train_steps = 15_000_000  # mirrors TrainConfig.total_timesteps; set by trainer
        self.dt = self.cfg.dt
        self.demo_mode = demo_mode
        self.demo_type = demo_type
        self.render_mode = render_mode

        if render_mode == "human":
            self.physics_client = self.p.connect(self.p.GUI)
        else:
            self.physics_client = self.p.connect(self.p.DIRECT)

        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = self.p.loadURDF(
            "assets/biped2d.urdf",
            [0, 0, 1.185],
            self.p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            physicsClientId=self.physics_client,
        )
        self.planeId = self.p.loadURDF(
            "plane.urdf", physicsClientId=self.physics_client
        )

        self.leg_len = _LEG_LEN_DEFAULT
        self.joint_idx = [2, 3, 4, 5, 6, 7, 8]
        self.max_torque = _MAX_TORQUE.copy()
        self.pos_normcoeff = _POS_NORMCOEFF
        self.velocity_normcoeff = _VEL_NORMCOEFF
        self.torque_normcoeff = float(self.max_torque.max())
        self.update_const = 0.75

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        obs_dim = self.cfg.obs_dim()
        self.observation_space = spaces.Box(
            low=-100, high=100, shape=(obs_dim,), dtype=np.float32
        )
        self.state = np.zeros(obs_dim, dtype=np.float64)

        self.gait_generator = gait_generator
        if self.gait_generator is None and self.cfg.gait_generator != "fft_mlp":
            from gait_generators import build_generator

            self.gait_generator = build_generator(
                self.cfg.gait_generator, dt=self.dt
            )
        if self.gait_generator is None:
            self.gaitgen_net = OldSimpleFCNN()
            self.gaitgen_net.load_state_dict(
                torch.load("final_model.pth", weights_only=True)
            )
            self.normalizationconst = np.load(r"newnormalization_constants.npy")
        else:
            self.gaitgen_net = None
            self.normalizationconst = None

        self.max_steps = int(self.cfg.max_episode_seconds * (1.0 / self.dt))
        self.t = 0

        self.double_support = True
        self.right_swing = False
        self.left_swing = False
        self.taken_step_counter = 0
        self.heightfield_data = None
        self.ramp_angle = 0.0

        # Per-step domain randomization state (re-sampled at reset).
        self._motor_delay_steps = 0
        self._motor_delay_buffer: list[np.ndarray] = []
        self._encoder_noise_std = 0.0
        self._friction = 1.0
        self._foot_mass_scale = 1.0

        # A4 — populated by step(), consumed by the per-component reward logger.
        self.last_reward_components: dict[str, float] = {}
        # C4 — Cost of Transport accumulators.
        self._energy_sum = 0.0
        self._distance_sum = 0.0

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        test_speed: float | None = None,
        test_angle: float | None = None,
        demo_max_steps: int | None = None,
        ground_noise: float | None = None,
        ground_resolution: float | None = None,
        heightfield_data: Any = None,
        options: Any = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self.test_speed = test_speed
        self.test_angle = test_angle
        self.max_steps = int(self.cfg.max_episode_seconds * (1.0 / self.dt))
        self.t = 0
        self.taken_step_counter = 0
        self._energy_sum = 0.0
        self._distance_sum = 0.0
        self._motor_delay_buffer.clear()

        self.p.resetSimulation(physicsClientId=self.physics_client)
        self.p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        self.p.setTimeStep(self.dt, physicsClientId=self.physics_client)

        if self.demo_mode:
            self.p.setPhysicsEngineParameter(
                fixedTimeStep=1.0 / 1000.0,
                numSolverIterations=100,
                deterministicOverlappingPairs=1,
                enableConeFriction=0,
                physicsClientId=self.physics_client,
            )
            self.p.setPhysicsEngineParameter(numSubSteps=0)
            self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.reference_speed = np.random.uniform(0.2, self.cfg.speed_limit)
        self.ramp_angle = (
            np.random.uniform(-self.cfg.ramp_limit_deg, self.cfg.ramp_limit_deg)
            * np.pi
            / 180
        )

        if self.demo_mode:
            if demo_max_steps:
                self.max_steps = demo_max_steps
            if test_speed is not None:
                self.reference_speed = float(test_speed)
            if test_angle is not None:
                self.ramp_angle = float(test_angle) * np.pi / 180

        self._apply_domain_randomization()

        self.reference = self._compute_reference_trajectory(self.reference_speed)
        self.reference = np.clip(self.reference, -np.pi / 2, np.pi / 2)

        plane_orientation = self.p.getQuaternionFromEuler(
            [self.ramp_angle, 0, 0], physicsClientId=self.physics_client
        )

        if (not self.demo_mode) or ground_noise is None:
            self.planeId = self.p.loadURDF(
                "plane.urdf",
                physicsClientId=self.physics_client,
                baseOrientation=plane_orientation,
            )
            self.p.changeDynamics(
                self.planeId,
                -1,
                lateralFriction=self._friction,
                frictionAnchor=1,
                physicsClientId=self.physics_client,
            )
        else:
            self._init_noisy_plane(
                ground_resolution=ground_resolution,
                baseOrientation=plane_orientation,
                heightfield_data=heightfield_data,
            )
            self.heightfield_data = heightfield_data

        self.reset_info = {"current state": self.state}
        self.past_action_error = np.zeros(7)
        self.current_action = np.zeros(7)
        self.target_action = np.zeros(7)
        self.past_target_action = np.zeros(7)
        self.past2_target_action = np.zeros(7)
        self.past_forward_place = 0.0
        self.control_freq = _CONTROL_SUBSTEPS
        self.external_states = np.zeros(4)
        self.ground_noise = ground_noise if ground_noise is not None else 0.0

        self._init_state()
        self._return_state()
        return self.state.astype(np.float32), self.reset_info

    def step(self, torques: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_counter += 1
        self.target_action = np.asarray(torques, dtype=np.float64) * self.max_torque

        applied_action = self._maybe_delay_motor(self.target_action)

        energy_inc = 0.0
        for _ in range(_CONTROL_SUBSTEPS):
            self.current_action = (
                self.update_const * applied_action
                + (1 - self.update_const) * self.current_action
            )
            self.t += 1
            self.p.setJointMotorControlArray(
                bodyIndex=self.robot,
                jointIndices=self.joint_idx,
                controlMode=self.p.TORQUE_CONTROL,
                forces=self.current_action,
                physicsClientId=self.physics_client,
            )
            self.p.stepSimulation(physicsClientId=self.physics_client)

            # C4 — accumulate |tau * q_dot| for Cost of Transport.
            joint_states = self.p.getJointStates(
                self.robot, self.joint_idx, physicsClientId=self.physics_client
            )
            q_dot = np.array([js[1] for js in joint_states], dtype=np.float64)
            energy_inc += float(np.sum(np.abs(self.current_action * q_dot))) * self.dt

        self._energy_sum += energy_inc

        self.past2_target_action = self.past_target_action
        self.past_target_action = self.target_action
        self._return_state()

        if self.render_mode == "human":
            time.sleep(self.dt)

        reward, done, components = self._compute_reward()
        self.last_reward_components = components
        truncated = self.t > self.max_steps
        info = dict(self.state_info)
        info["reward_components"] = components
        info["cost_of_transport"] = self._current_cot()
        return self.state.astype(np.float32), float(reward), bool(done), bool(truncated), info

    def close(self) -> None:
        self.p.disconnect(physicsClientId=self.physics_client)

    # ------------------------------------------------------------------
    # Gait generator
    # ------------------------------------------------------------------

    def _compute_reference_trajectory(self, speed: float) -> np.ndarray:
        """Return a (T, 6) joint-angle trajectory for ``speed``.

        Uses the external ``gait_generator`` when provided (Track B2), else
        falls back to the paper's FFT MLP.
        """
        if self.gait_generator is not None:
            return self.gait_generator.predict(speed, (self.leg_len, self.leg_len))
        encoder_vec = torch.tensor(
            [speed / 3.0, self.leg_len / 1.5, self.leg_len / 1.5], dtype=torch.float32
        )
        return self._findgait(encoder_vec)

    def _findgait(self, input_vec: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            freqs = self.gaitgen_net(input_vec)
        predictions = freqs.reshape(-1, 6, 2, 17).detach().numpy()[0]
        predictions = self._denormalize(predictions)
        return self._pred_ifft(predictions)

    def _denormalize(self, pred: np.ndarray) -> np.ndarray:
        for i in range(17):
            for k in range(2):
                pred[:, k, i] = pred[:, k, i] * self.normalizationconst[i * 2 + k]
        return pred

    def _pred_ifft(self, predictions: np.ndarray) -> np.ndarray:
        real_pred = predictions[:, 0, :]
        imag_pred = predictions[:, 1, :]
        complex_pred = real_pred + 1j * imag_pred
        pred_time = np.fft.irfft(complex_pred, axis=1).transpose(1, 0)
        org_rate = 10
        if self.dt < 0.1:
            num_samples = int(pred_time.shape[0] * (1.0 / self.dt) / org_rate)
            pred_time = resample(pred_time, num_samples, axis=0)
            pred_time = np.tile(pred_time, (50, 1))
        return pred_time

    def change_ref_speed(self, new_speed: float) -> None:
        """Hot-swap the reference trajectory mid-episode (velocity-tracking demo)."""
        newly_reference = np.clip(
            self._compute_reference_trajectory(new_speed), -np.pi / 2, np.pi / 2
        )
        current_ref_pos = np.array(
            [
                self.reference[self.reference_idx + self.t, 0],
                self.reference[self.reference_idx + self.t, 1],
                self.reference[self.reference_idx + self.t, 3],
                self.reference[self.reference_idx + self.t, 4],
            ]
        )
        ref_pos = np.array(
            [
                newly_reference[:, 0],
                newly_reference[:, 1],
                newly_reference[:, 3],
                newly_reference[:, 4],
            ]
        ).T
        distances = np.linalg.norm(ref_pos - current_ref_pos, axis=1)
        self.t = int(np.argmin(distances))
        self.reference_idx = 0
        self.reference = newly_reference
        self.reference_speed = new_speed

    # ------------------------------------------------------------------
    # Reward (paper Eq. 2-5 + 10)
    # ------------------------------------------------------------------

    def _decay_weights(self) -> tuple[float, float]:
        """Return ``(omega_imitation, omega_gait)`` for the current step.

        ``"none"`` keeps both weights at 1.0. ``"linear_step"`` applies paper
        equation (3): ``1 ± alpha * step / total_steps``.
        """
        if self.cfg.decay_schedule == "none":
            return 1.0, 1.0
        frac = min(self.step_counter / max(self.total_train_steps, 1), 1.0)
        delta = self.cfg.decay_alpha * frac
        return 1.0 - delta, 1.0 + delta

    def _compute_reward(self) -> tuple[float, bool, dict[str, float]]:
        x = self.state
        components: dict[str, float] = {
            "alive": 0.0,
            "contact": 0.0,
            "speed": 0.0,
            "torque": 0.0,
            "im_hip_pos": 0.0,
            "im_knee_pos": 0.0,
            "im_ankle_pos": 0.0,
            "im_hip_vel": 0.0,
            "im_knee_vel": 0.0,
            "im_ankle_vel": 0.0,
        }
        done = False
        omega_im, omega_gait = self._decay_weights()

        # Contact
        contact_points = self.p.getContactPoints(
            self.robot, self.planeId, physicsClientId=self.physics_client
        )
        left_contact_forces = [c[9] for c in contact_points if c[3] == _LEFT_FOOT_LINK]
        left_contact = len(left_contact_forces)
        left_contact_mean = float(np.mean(left_contact_forces)) if left_contact_forces else 0.0
        lfoot_pos = self.p.getLinkState(
            self.robot, _LEFT_FOOT_LINK, computeLinkVelocity=True, physicsClientId=self.physics_client
        )[0]
        right_contact_forces = [
            c[9] for c in contact_points if c[3] == _RIGHT_FOOT_LINK
        ]
        right_contact = len(right_contact_forces)
        right_contact_mean = float(np.mean(right_contact_forces)) if right_contact_forces else 0.0
        rfoot_pos = self.p.getLinkState(
            self.robot, _RIGHT_FOOT_LINK, computeLinkVelocity=True, physicsClientId=self.physics_client
        )[0]
        components["contact"] = self.cfg.contact_weight * self._calculate_contact_reward(
            left_contact, right_contact, left_contact_mean, right_contact_mean, lfoot_pos, rfoot_pos
        )

        # Imitation (paper Eq. 4)
        if self.cfg.reward != "no_im" and self.cfg.reward != "gait_only":
            ref_base = self._ref_slice_start()
            hip_joint_pos = x[[7, 10]] * self.pos_normcoeff
            hip_ref_pos = x[[ref_base, ref_base + 3]] * self.pos_normcoeff
            components["im_hip_pos"] = (
                omega_im
                * self.cfg.imitation_weight_hip_pos
                * np.exp(-5 * np.linalg.norm(hip_joint_pos - hip_ref_pos))
            )

            knee_joint_pos = x[[8, 11]] * self.pos_normcoeff
            knee_ref_pos = x[[ref_base + 1, ref_base + 4]] * self.pos_normcoeff
            components["im_knee_pos"] = (
                omega_im
                * self.cfg.imitation_weight_knee_pos
                * np.exp(-5 * np.linalg.norm(knee_joint_pos - knee_ref_pos))
            )

            ankle_joint_pos = x[[9, 12]] * self.pos_normcoeff
            ankle_ref_pos = x[[ref_base + 2, ref_base + 5]] * self.pos_normcoeff
            components["im_ankle_pos"] = (
                omega_im
                * self.cfg.imitation_weight_ankle_pos
                * np.exp(-5 * np.linalg.norm(ankle_joint_pos - ankle_ref_pos))
            )

            ref_vel_base = self._ref_vel_slice_start()
            hip_joint_vel = x[[28, 31]] * self.velocity_normcoeff
            hip_ref_vel = x[[ref_vel_base, ref_vel_base + 3]] * self.velocity_normcoeff
            components["im_hip_vel"] = (
                omega_im
                * self.cfg.imitation_weight_hip_vel
                * np.exp(-0.2 * np.linalg.norm(hip_joint_vel - hip_ref_vel))
            )

            knee_joint_vel = x[[29, 32]] * self.velocity_normcoeff
            knee_ref_vel = x[[ref_vel_base + 1, ref_vel_base + 4]] * self.velocity_normcoeff
            components["im_knee_vel"] = (
                omega_im
                * self.cfg.imitation_weight_knee_vel
                * np.exp(-0.2 * np.linalg.norm(knee_joint_vel - knee_ref_vel))
            )

            ankle_joint_vel = x[[30, 33]] * self.velocity_normcoeff
            ankle_ref_vel = x[[ref_vel_base + 2, ref_vel_base + 5]] * self.velocity_normcoeff
            components["im_ankle_vel"] = (
                omega_im
                * self.cfg.imitation_weight_ankle_vel
                * np.exp(-0.2 * np.linalg.norm(ankle_joint_vel - ankle_ref_vel))
            )

        # Torque (paper Eq. 10)
        components["torque"] = -self.cfg.torque_weight * float(
            np.mean(np.abs(self.target_action))
        )

        # Forward speed (paper Eq. 7)
        current_speed = (self.external_states[1] - self.past_forward_place) / (
            self.dt * _CONTROL_SUBSTEPS
        )
        components["speed"] = (
            self.cfg.speed_weight
            * omega_gait
            * float(np.exp(-2 * np.abs(current_speed - self.reference_speed)))
        )

        # Alive / termination (paper Eq. 6)
        if np.abs(self.external_states[3]) > 0.98:
            components["alive"] = -100.0
            done = True
        else:
            alive_term, alive_done = self._alive_reward()
            components["alive"] = alive_term
            done = done or alive_done

        reward = sum(components.values())
        return reward, done, components

    def _alive_reward(self) -> tuple[float, bool]:
        aw = self.cfg.alive_weight
        if self.demo_type != "noisy":
            plane_z = 0.0
        else:
            x_pos = self.external_states[1]
            plane_z = float(self.heightfield_data[int((x_pos / 0.05 + 512) * 32)])
        ramp_off = np.tan(self.ramp_angle) * self.external_states[1] + plane_z
        z = self.external_states[2]
        if z > 1.45 + ramp_off:
            return -100.0, True
        if z > 1.3 + ramp_off:
            return -aw, False
        if z < 0.8 + ramp_off:
            return -100.0, True
        if z < 0.98 + ramp_off:
            return -aw, False
        return aw, False

    def _calculate_contact_reward(
        self,
        left_contact: int,
        right_contact: int,
        left_force: float,
        right_force: float,
        lfoot_pos: tuple[float, float, float],
        rfoot_pos: tuple[float, float, float],
        force_eps: float = 10.0,
    ) -> float:
        if left_force > force_eps and right_force > force_eps:
            if not self.double_support:
                self.taken_step_counter += 1
            self.double_support = True
            self.right_swing = False
            self.left_swing = False
        elif left_force > force_eps and right_force <= force_eps:
            if not self.right_swing:
                self.taken_step_counter += 1
            self.double_support = False
            self.right_swing = True
            self.left_swing = False
        elif right_force > force_eps and left_force <= force_eps:
            if not self.left_swing:
                self.taken_step_counter += 1
            self.double_support = False
            self.right_swing = False
            self.left_swing = True
        else:
            self.double_support = False
            self.right_swing = False
            self.left_swing = False

        if self.double_support:
            contact_no = left_contact + right_contact
            return 1 / (1 + np.exp(-2 * (contact_no - 4.0)))
        if self.right_swing:
            plane_height = np.tan(self.ramp_angle) * rfoot_pos[1]
            clearance = 1 / (1 + np.exp(20 * np.abs(rfoot_pos[2] - plane_height - 0.15)))
            contact = 1 / (1 + np.exp(-2 * (left_contact - 2)))
            return 0.5 * clearance + 0.5 * contact
        if self.left_swing:
            plane_height = np.tan(self.ramp_angle) * lfoot_pos[1]
            clearance = 1 / (1 + np.exp(20 * np.abs(lfoot_pos[2] - plane_height - 0.15)))
            contact = 1 / (1 + np.exp(-2 * (right_contact - 2)))
            return 0.5 * clearance + 0.5 * contact
        return 0.0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _ref_slice_start(self) -> int:
        """Index of the first reference-preview element (q_hat_t) in ``self.state``."""
        return 34 if self.cfg.include_pad_dims else 31

    def _ref_vel_slice_start(self) -> int:
        """Index of the first reference-velocity element (q_hat_dot_t)."""
        return 52 if self.cfg.include_pad_dims else 49

    def _init_state(self) -> None:
        if not self.demo_mode and self.cfg.rsi:
            start_idx = int(np.random.randint(0, 500))
        else:
            start_idx = 0
        self.reference_idx = start_idx

        rhip_pos = self.reference[start_idx, 0]
        rknee_pos = self.reference[start_idx, 1]
        rankle_pos = self.reference[start_idx, 2]
        lhip_pos = self.reference[start_idx, 3]
        lknee_pos = self.reference[start_idx, 4]
        lankle_pos = self.reference[start_idx, 5]

        hip_init = lhip_pos if np.abs(rhip_pos) > np.abs(lhip_pos) else rhip_pos
        knee_init = lknee_pos if np.abs(rknee_pos) > np.abs(lknee_pos) else rknee_pos
        ankle_init = lankle_pos if np.abs(rankle_pos) < np.abs(lankle_pos) else rankle_pos

        init_z = self._starting_height(hip_init, knee_init, ankle_init)

        if hasattr(self, "robot"):
            del self.robot
        drop = 0.02 if (not self.demo_mode and self.cfg.rsi) else 0.0
        self.robot = self.p.loadURDF(
            "assets/biped2d.urdf",
            [0, 0, init_z + drop] if (not self.demo_mode and self.cfg.rsi) else [0, 0, 1.185],
            self.p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            physicsClientId=self.physics_client,
        )
        self.p.setJointMotorControlArray(
            self.robot,
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            self.p.VELOCITY_CONTROL,
            forces=[0, 0, 0, 0, 0, 0, 0, 0, 0],
            physicsClientId=self.physics_client,
        )
        self._apply_foot_mass_scale()
        if not self.demo_mode and self.cfg.rsi:
            for idx, val in zip(
                [3, 4, 5, 6, 7, 8],
                [rhip_pos, rknee_pos, rankle_pos, lhip_pos, lknee_pos, lankle_pos],
            ):
                self.p.resetJointState(
                    self.robot, idx, targetValue=val, physicsClientId=self.physics_client
                )

        self.p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        self.p.setTimeStep(self.dt, physicsClientId=self.physics_client)

        self.t1_torso_pos = self.p.getJointState(
            self.robot, 2, physicsClientId=self.physics_client
        )[0]
        self.t1_rhip_pos = self.p.getJointState(
            self.robot, 3, physicsClientId=self.physics_client
        )[0]
        self.t1_rknee_pos = self.p.getJointState(
            self.robot, 4, physicsClientId=self.physics_client
        )[0]
        self.t1_rankle_pos = self.p.getJointState(
            self.robot, 5, physicsClientId=self.physics_client
        )[0]
        self.t1_lhip_pos = self.p.getJointState(
            self.robot, 6, physicsClientId=self.physics_client
        )[0]
        self.t1_lknee_pos = self.p.getJointState(
            self.robot, 7, physicsClientId=self.physics_client
        )[0]
        self.t1_lankle_pos = self.p.getJointState(
            self.robot, 8, physicsClientId=self.physics_client
        )[0]

    def _starting_height(self, hip_init: float, knee_init: float, ankle_init: float) -> float:
        upper_len = 0.45
        lower_len = 0.45
        hip_short = upper_len - upper_len * np.cos(hip_init)
        knee_short = lower_len - lower_len * np.cos(knee_init)
        return 1.195 - hip_short - knee_short

    def _return_state(self) -> None:
        link_state = self.p.getLinkState(
            self.robot, 2, computeLinkVelocity=True, physicsClientId=self.physics_client
        )
        torso_g_quat = link_state[1]
        roll, _, _ = self.p.getEulerFromQuaternion(
            torso_g_quat, physicsClientId=self.physics_client
        )
        pos_x, pos_y, pos_z = link_state[0]
        y_vel = link_state[6][1]

        self.torso_pos = self.p.getJointState(
            self.robot, 2, physicsClientId=self.physics_client
        )[0]
        self.rhip_pos = self.p.getJointState(
            self.robot, 3, physicsClientId=self.physics_client
        )[0]
        self.rknee_pos = self.p.getJointState(
            self.robot, 4, physicsClientId=self.physics_client
        )[0]
        self.rankle_pos = self.p.getJointState(
            self.robot, 5, physicsClientId=self.physics_client
        )[0]
        self.lhip_pos = self.p.getJointState(
            self.robot, 6, physicsClientId=self.physics_client
        )[0]
        self.lknee_pos = self.p.getJointState(
            self.robot, 7, physicsClientId=self.physics_client
        )[0]
        self.lankle_pos = self.p.getJointState(
            self.robot, 8, physicsClientId=self.physics_client
        )[0]

        self.torso_vel = self.p.getJointState(
            self.robot, 2, physicsClientId=self.physics_client
        )[1]
        self.rhip_vel = self.p.getJointState(
            self.robot, 3, physicsClientId=self.physics_client
        )[1]
        self.rknee_vel = self.p.getJointState(
            self.robot, 4, physicsClientId=self.physics_client
        )[1]
        self.rankle_vel = self.p.getJointState(
            self.robot, 5, physicsClientId=self.physics_client
        )[1]
        self.lhip_vel = self.p.getJointState(
            self.robot, 6, physicsClientId=self.physics_client
        )[1]
        self.lknee_vel = self.p.getJointState(
            self.robot, 7, physicsClientId=self.physics_client
        )[1]
        self.lankle_vel = self.p.getJointState(
            self.robot, 8, physicsClientId=self.physics_client
        )[1]

        if self._encoder_noise_std > 0.0:
            noise = np.random.normal(0.0, self._encoder_noise_std, size=7)
            self.rhip_pos += noise[1]
            self.rknee_pos += noise[2]
            self.rankle_pos += noise[3]
            self.lhip_pos += noise[4]
            self.lknee_pos += noise[5]
            self.lankle_pos += noise[6]

        ref = self.reference
        idx = self.reference_idx
        t = self.t
        ref_rhip_vel = (ref[idx + t, 0] - ref[idx + t - 1, 0]) / self.dt
        ref_rknee_vel = (ref[idx + t, 1] - ref[idx + t - 1, 1]) / self.dt
        ref_rankle_vel = (ref[idx + t, 2] - ref[idx + t - 1, 2]) / self.dt
        ref_lhip_vel = (ref[idx + t, 3] - ref[idx + t - 1, 3]) / self.dt
        ref_lknee_vel = (ref[idx + t, 4] - ref[idx + t - 1, 4]) / self.dt
        ref_lankle_vel = (ref[idx + t, 5] - ref[idx + t - 1, 5]) / self.dt

        self.past_forward_place = self.external_states[1]
        self.external_states = [pos_x, pos_y, pos_z, roll]
        self._distance_sum = max(self._distance_sum, pos_y)

        if self.cfg.include_pad_dims:
            self.state[0] = self.reference_speed / 3
            self.state[1] = self.ramp_angle
            self.state[2] = 0.0
            self.state[3] = 0.0
            self.state[4] = 0.0
            self.state[5] = y_vel / 3
            joint_base = 6
        else:
            self.state[0] = self.reference_speed / 3
            self.state[1] = self.ramp_angle
            self.state[2] = y_vel / 3
            joint_base = 3

        self.state[joint_base : joint_base + 7] = (
            np.array(
                [
                    self.torso_pos,
                    self.rhip_pos,
                    self.rknee_pos,
                    self.rankle_pos,
                    self.lhip_pos,
                    self.lknee_pos,
                    self.lankle_pos,
                ]
            )
            / self.pos_normcoeff
        )

        action_base = joint_base + 7
        self.state[action_base : action_base + 7] = (
            self.past_target_action / self.max_torque
        )

        prev_pos_base = action_base + 7
        self.state[prev_pos_base : prev_pos_base + 7] = (
            np.array(
                [
                    self.t1_torso_pos,
                    self.t1_rhip_pos,
                    self.t1_rknee_pos,
                    self.t1_rankle_pos,
                    self.t1_lhip_pos,
                    self.t1_lknee_pos,
                    self.t1_lankle_pos,
                ]
            )
            / self.pos_normcoeff
        )

        vel_base = prev_pos_base + 7
        self.state[vel_base : vel_base + 7] = (
            np.array(
                [
                    self.torso_vel,
                    self.rhip_vel,
                    self.rknee_vel,
                    self.rankle_vel,
                    self.lhip_vel,
                    self.lknee_vel,
                    self.lankle_vel,
                ]
            )
            / self.velocity_normcoeff
        )

        ref_base = vel_base + 7  # 34 (legacy) / 31 (refactored)
        if self.cfg.observation == "no_im_state":
            self.state[ref_base:] = 0.0
        else:
            t_now = min(idx + t, len(ref) - 1)
            t_p1 = min(idx + t + 1, len(ref) - 1)
            t_p10 = min(idx + t + 10, len(ref) - 1)
            t_p100 = min(idx + t + 100, len(ref) - 1)
            self.state[ref_base : ref_base + 6] = ref[t_now, :] / self.pos_normcoeff
            self.state[ref_base + 6 : ref_base + 12] = ref[t_p1, :] / self.pos_normcoeff
            self.state[ref_base + 12 : ref_base + 18] = ref[t_p100, :] / self.pos_normcoeff
            self.state[ref_base + 18 : ref_base + 24] = (
                np.array(
                    [
                        ref_rhip_vel,
                        ref_rknee_vel,
                        ref_rankle_vel,
                        ref_lhip_vel,
                        ref_lknee_vel,
                        ref_lankle_vel,
                    ]
                )
                / self.velocity_normcoeff
            )
            # The third preview slot in Table 1 is q_hat_{t+10}; we keep
            # q_hat_{t+100} here to mirror the frozen Configuration 1 code.
            # Update only if `BipedEnvConfig.observation` gets a new mode.
            _ = t_p10

        self.t1_torso_pos = self.torso_pos
        self.t1_rhip_pos = self.rhip_pos
        self.t1_rknee_pos = self.rknee_pos
        self.t1_rankle_pos = self.rankle_pos
        self.t1_lhip_pos = self.lhip_pos
        self.t1_lknee_pos = self.lknee_pos
        self.t1_lankle_pos = self.lankle_pos
        self.state_info = self._state_info_map()

    def _state_info_map(self) -> dict[int, str]:
        """Return a mapping of observation index -> human-readable name."""
        if self.cfg.include_pad_dims:
            return {
                0: "reference_speed",
                1: "ramp_angle",
                2: "pad0",
                3: "pad1",
                4: "pad2",
                5: "y_vel",
                6: "torso_pos",
                7: "rhip_pos",
                8: "rknee_pos",
                9: "rankle_pos",
                10: "lhip_pos",
                11: "lknee_pos",
                12: "lankle_pos",
                13: "t1_torso_action",
                14: "t1_rhip_action",
                15: "t1_rknee_action",
                16: "t1_rankle_action",
                17: "t1_lhip_action",
                18: "t1_lknee_action",
                19: "t1_lankle_action",
                20: "t1torso_pos",
                21: "t1rhip_pos",
                22: "t1rknee_pos",
                23: "t1rankle_pos",
                24: "t1lhip_pos",
                25: "t1lknee_pos",
                26: "t1lankle_pos",
                27: "torso_vel",
                28: "rhip_vel",
                29: "rknee_vel",
                30: "rankle_vel",
                31: "lhip_vel",
                32: "lknee_vel",
                33: "lankle_vel",
                34: "ref_rhip",
                35: "ref_rknee",
                36: "ref_rankle",
                37: "ref_lhip",
                38: "ref_lknee",
                39: "ref_lankle",
                40: "ref_p1_rhip",
                41: "ref_p1_rknee",
                42: "ref_p1_rankle",
                43: "ref_p1_lhip",
                44: "ref_p1_lknee",
                45: "ref_p1_lankle",
                46: "ref_p100_rhip",
                47: "ref_p100_rknee",
                48: "ref_p100_rankle",
                49: "ref_p100_lhip",
                50: "ref_p100_lknee",
                51: "ref_p100_lankle",
                52: "ref_rhip_vel",
                53: "ref_rknee_vel",
                54: "ref_rankle_vel",
                55: "ref_lhip_vel",
                56: "ref_lknee_vel",
                57: "ref_lankle_vel",
            }
        return {
            0: "reference_speed",
            1: "ramp_angle",
            2: "y_vel",
            3: "torso_pos",
            4: "rhip_pos",
            5: "rknee_pos",
            6: "rankle_pos",
            7: "lhip_pos",
            8: "lknee_pos",
            9: "lankle_pos",
            10: "t1_torso_action",
            11: "t1_rhip_action",
            12: "t1_rknee_action",
            13: "t1_rankle_action",
            14: "t1_lhip_action",
            15: "t1_lknee_action",
            16: "t1_lankle_action",
            17: "t1torso_pos",
            18: "t1rhip_pos",
            19: "t1rknee_pos",
            20: "t1rankle_pos",
            21: "t1lhip_pos",
            22: "t1lknee_pos",
            23: "t1lankle_pos",
            24: "torso_vel",
            25: "rhip_vel",
            26: "rknee_vel",
            27: "rankle_vel",
            28: "lhip_vel",
            29: "lknee_vel",
            30: "lankle_vel",
            31: "ref_rhip",
            32: "ref_rknee",
            33: "ref_rankle",
            34: "ref_lhip",
            35: "ref_lknee",
            36: "ref_lankle",
            37: "ref_p1_rhip",
            38: "ref_p1_rknee",
            39: "ref_p1_rankle",
            40: "ref_p1_lhip",
            41: "ref_p1_lknee",
            42: "ref_p1_lankle",
            43: "ref_p100_rhip",
            44: "ref_p100_rknee",
            45: "ref_p100_rankle",
            46: "ref_p100_lhip",
            47: "ref_p100_lknee",
            48: "ref_p100_lankle",
            49: "ref_rhip_vel",
            50: "ref_rknee_vel",
            51: "ref_rankle_vel",
            52: "ref_lhip_vel",
            53: "ref_lknee_vel",
            54: "ref_lankle_vel",
        }

    # ------------------------------------------------------------------
    # Terrain and domain randomization
    # ------------------------------------------------------------------

    def _apply_domain_randomization(self) -> None:
        dr = self.cfg.dr
        if not dr.enabled:
            self._friction = 1.0
            self._motor_delay_steps = 0
            self._encoder_noise_std = 0.0
            self._foot_mass_scale = 1.0
            return
        self._friction = float(np.random.uniform(*dr.friction_range))
        self._motor_delay_steps = int(
            np.round(np.random.uniform(*dr.motor_delay_ms_range))
        )
        self._encoder_noise_std = dr.encoder_noise_std
        # Foot mass is applied after the robot is loaded in _init_state via
        # :meth:`_apply_foot_mass_scale`.
        self._foot_mass_scale = float(np.random.uniform(*dr.foot_mass_scale_range))

    def _apply_foot_mass_scale(self) -> None:
        """Scale the mass of each foot link by ``self._foot_mass_scale``."""
        if abs(self._foot_mass_scale - 1.0) < 1e-9:
            return
        for link in (_RIGHT_FOOT_LINK, _LEFT_FOOT_LINK):
            info = self.p.getDynamicsInfo(
                self.robot, link, physicsClientId=self.physics_client
            )
            new_mass = float(info[0]) * self._foot_mass_scale
            self.p.changeDynamics(
                self.robot, link,
                mass=new_mass,
                physicsClientId=self.physics_client,
            )

    def _maybe_delay_motor(self, action: np.ndarray) -> np.ndarray:
        if self._motor_delay_steps <= 0:
            return action
        self._motor_delay_buffer.append(np.asarray(action, dtype=np.float64).copy())
        if len(self._motor_delay_buffer) <= self._motor_delay_steps:
            return np.zeros_like(action)
        delayed = self._motor_delay_buffer.pop(0)
        return delayed

    def _init_noisy_plane(
        self,
        ground_resolution: float | None,
        baseOrientation: Any,
        heightfield_data: Any,
        num_rows: int = 32,
        num_columns: int = 1024,
    ) -> None:
        mesh_scale = [ground_resolution or 0.05, ground_resolution or 0.05, 1]
        if baseOrientation is None:
            baseOrientation = self.p.getQuaternionFromEuler(
                [0, 0, 0], physicsClientId=self.physics_client
            )
        terrain_shape = self.p.createCollisionShape(
            shapeType=self.p.GEOM_HEIGHTFIELD,
            meshScale=mesh_scale,
            heightfieldData=heightfield_data,
            numHeightfieldRows=num_rows,
            numHeightfieldColumns=num_columns,
            physicsClientId=self.physics_client,
        )
        min_h = float(np.min(heightfield_data))
        max_h = float(np.max(heightfield_data))
        z_center = 0.5 * (min_h + max_h) * mesh_scale[2]
        self.planeId = self.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            basePosition=[0, 0, z_center],
            baseOrientation=baseOrientation,
            physicsClientId=self.physics_client,
        )
        self.p.changeDynamics(
            self.planeId,
            -1,
            lateralFriction=self._friction,
            frictionAnchor=1,
            physicsClientId=self.physics_client,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_cot(self) -> float:
        """Return the running Cost of Transport estimate (dimensionless)."""
        m = 60.0  # biped mass (approx.); URDF-derived
        g = 9.81
        d = max(self._distance_sum, 1e-6)
        return self._energy_sum / (m * g * d)

    def return_external_state(self) -> np.ndarray:
        return np.asarray(self.external_states)

    def return_step_taken(self) -> int:
        return int(self.taken_step_counter)

    def apply_external_impulse(
        self, magnitude: float, direction: tuple[float, float, float]
    ) -> None:
        """Apply a one-shot world-frame impulse to the torso (push-recovery demo, C5)."""
        force = tuple(magnitude * d for d in direction)
        self.p.applyExternalForce(
            objectUniqueId=self.robot,
            linkIndex=2,
            forceObj=force,
            posObj=[0, 0, 0],
            flags=self.p.LINK_FRAME,
            physicsClientId=self.physics_client,
        )

    def get_image(self) -> Image.Image:
        view_matrix = self.p.computeViewMatrix(
            cameraEyePosition=[3, 0, 1.5],
            cameraTargetPosition=[0, 0, 1.0],
            cameraUpVector=[0, 0, 1],
        )
        projection_matrix = self.p.computeProjectionMatrixFOV(
            fov=75, aspect=1.0, nearVal=0.1, farVal=100.0
        )
        res = 640
        _, _, rgbImg, _, _ = self.p.getCameraImage(
            width=res,
            height=res,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
        )
        rgb_array = np.reshape(rgbImg, (res, res, 4))
        return Image.fromarray(rgb_array[:, :, :3], "RGB")

    def get_follow_camera_image(
        self, follow_distance: float = 3.0, height: float = 1.5, overlay_text: str | None = None
    ) -> Image.Image:
        torso_state = self.p.getLinkState(
            self.robot, 2, physicsClientId=self.physics_client
        )
        torso_pos = torso_state[0]
        camera_eye = [torso_pos[0] - follow_distance, torso_pos[1], height]
        camera_target = [torso_pos[0], torso_pos[1], height]
        view_matrix = self.p.computeViewMatrix(
            cameraEyePosition=camera_eye,
            cameraTargetPosition=camera_target,
            cameraUpVector=[0, 0, 1],
        )
        projection_matrix = self.p.computeProjectionMatrixFOV(
            fov=75, aspect=1.0, nearVal=0.1, farVal=100.0
        )
        res = 640
        _, _, rgbImg, _, _ = self.p.getCameraImage(
            width=res,
            height=res,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
        )
        rgb_array = np.reshape(rgbImg, (res, res, 4))
        image = Image.fromarray(rgb_array[:, :, :3], "RGB")
        if overlay_text:
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("DejaVuSansMono.ttf", 22)
            except OSError:
                font = ImageFont.load_default()
            pad = 6
            tw, th = draw.textbbox((0, 0), overlay_text, font=font)[2:]
            draw.rectangle([(10, 10), (10 + tw + 2 * pad, 10 + th + 2 * pad)], fill=(0, 0, 0, 160))
            draw.text((10 + pad, 10 + pad), overlay_text, fill=(255, 255, 255), font=font)
        return image


__all__ = ["BipedEnv"]
