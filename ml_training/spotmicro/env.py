from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from commands.device import Device
from spotmicro.agent import Agent, AgentParams
from spotmicro.sim import MujocoSim

if TYPE_CHECKING:
    from training.rewards import Reward


@dataclass
class EnvParams:
    """Environment settings (the ``env:`` section of a run config)."""

    max_episode_len: int = 3000  # control steps before the episode is truncated
    sim_frequency: int = 240  # physics steps per second
    control_frequency: int = 60  # policy steps per second
    min_height: float = 0.15  # episode ends below this base height [m]
    max_height: float = 0.4  # episode ends above this base height [m]
    max_pitchroll: float = 0.96  # episode ends beyond this roll/pitch [rad]
    spawn_height: float = 0.230
    target_body_to_feet_height: float = 0.2
    model_path: str | None = None  # MuJoCo XML; None = robot_models/spotmicroai.mujoco.xml


class SpotmicroEnv(gym.Env):
    """
    SpotMicro environment simulated in MuJoCo.

    SB3 <-> Env <-> Agent <-> MujocoSim (available as ``env.sim``)

    The policy follows a velocity command from ``command`` and is scored by ``reward``.
    Curricula can modify actions and observations through ``action_filters`` and
    ``observation_filters`` (functions applied in order).
    """

    OBS_SIZE = 97
    ACTION_SIZE = 12

    observation_space: gym.spaces.Box
    action_space: gym.spaces.Box

    @classmethod
    def spaces(cls) -> tuple[gym.spaces.Box, gym.spaces.Box]:
        """(observation space, action space), without creating an environment."""
        return (
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(cls.OBS_SIZE,), dtype=np.float32),
            gym.spaces.Box(low=-1.0, high=1.0, shape=(cls.ACTION_SIZE,), dtype=np.float32),
        )

    def __init__(
        self,
        command: Device,
        reward: "Reward",
        params: EnvParams | None = None,
        agent_params: AgentParams | None = None,
        use_gui: bool = False,
    ) -> None:
        super().__init__()
        self.params = params if params is not None else EnvParams()
        p = self.params
        self.sim = MujocoSim(use_gui=use_gui)
        self.reward = reward
        self.action_filters: list[Callable[[np.ndarray], np.ndarray]] = []
        self.observation_filters: list[Callable[[np.ndarray], np.ndarray]] = []

        self._episode_reward_info: list[dict[str, float]] = []
        # The previous episode, kept at reset: SB3 resets the env before callbacks see `done`
        self.last_episode_reward_info: list[dict[str, float]] = []
        self.last_episode_length = 0
        self._episode_step_counter = 0
        self._total_steps_counter = 0
        # How the current step ended the episode, read by the reward (see _termination)
        self.termination: str | None = None
        self.truncated = False
        # Added to num_steps per step: with N environments in parallel, set to N so that each
        # one's num_steps counts the steps of all of them (rewards schedule weights on it)
        self.steps_per_step = 1

        self.observation_space, self.action_space = self.spaces()

        # Read by rewards and curricula
        self.max_episode_len = p.max_episode_len
        self.sim_frequency = p.sim_frequency
        self.control_frequency = p.control_frequency
        self.target_body_to_feet_height = p.target_body_to_feet_height

        if p.sim_frequency % p.control_frequency != 0:
            raise ValueError(
                f"sim_frequency ({p.sim_frequency}) must be a multiple of "
                f"control_frequency ({p.control_frequency})"
            )
        model_path = p.model_path or str(files("robot_models").joinpath("spotmicroai.mujoco.xml"))
        # Create agent — it loads the model into the simulation
        self._agent = Agent(
            sim=self.sim,
            model_path=model_path,
            spawn_height=p.spawn_height,
            command=command,
            action_space_size=self.ACTION_SIZE,
            params=agent_params,
        )
        # The physics timestep follows sim_frequency, whatever the XML says
        self.sim.set_timestep(1.0 / p.sim_frequency)
        self._substeps = p.sim_frequency // p.control_frequency
        # Joint limits as arrays (joint order), to normalize joint positions
        self._joint_low = np.array([j.limits[0] for j in self._agent.motor_joints])
        self._joint_range = np.array([j.limits[1] - j.limits[0] for j in self._agent.motor_joints])

    def close(self) -> None:
        self.sim.close()

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.last_episode_reward_info = self._episode_reward_info
        self.last_episode_length = self._episode_step_counter
        self._episode_step_counter = 0
        self._episode_reward_info = []
        self.termination = None
        self.truncated = False

        self._agent.reset(self.params.spawn_height)

        # Let physics settle with homing applied
        for _ in range(5):
            self._agent.apply_action(self._agent.default_actions)
            self.sim.step()
            self._agent.sync_state()

        self.reward.reset(self)
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        for action_filter in self.action_filters:
            action = action_filter(action)
        self._agent.command.update()
        observation = self._step_simulation(action)
        self._episode_step_counter += 1
        # Decided before the reward, which adds its own penalties or bonus for the ending
        self.termination = self._termination()
        self.truncated = self._episode_step_counter >= self.max_episode_len
        reward, reward_info = self.reward(self, action)
        self._episode_reward_info.append(reward_info)

        self._total_steps_counter += self.steps_per_step
        terminated = self.termination is not None
        return observation, reward, terminated, self.truncated, self._get_info()

    def _step_simulation(self, action: np.ndarray) -> np.ndarray:
        self._agent.apply_action(action)
        for _ in range(self._substeps):
            self.sim.step()

        # External pushes last for one control step
        self.sim.clear_external_forces()
        self.sim.sync_viewer()
        self._agent.sync_state()
        return self._get_observation()

    def _get_gravity_vector(self) -> np.ndarray:
        return self._agent.state.rotation.T @ np.array([0.0, 0.0, -1.0])

    def _joint_positions_norm(self, pos: np.ndarray) -> np.ndarray:
        """Joint positions mapped to [-1, 1] (lower limit -> -1, upper limit -> 1)."""
        return ((2 * (pos - self._joint_low)) / self._joint_range) - 1

    def _joint_velocities_norm(self, vels: np.ndarray) -> np.ndarray:
        return np.tanh(vels / self._agent.max_joint_velocity)

    def _get_observation(self) -> np.ndarray:
        """
        Vectors are in the robot frame (x towards the head, y left, z up).

        - 0-2: gravity direction
        - 3: height of the base above target_body_to_feet_height, normalized
        - 4-6: linear velocity of the base
        - 7-9: angular velocity of the base
        - 10-21: positions of the joints
        - 22-33: velocities of the joints
        - 34-57: joint positions and velocities one control step ago
        - 58-81: joint positions and velocities four control steps ago
        - 82-93: the action applied at the latest step
        - 94-95: commanded linear velocity (vx, vy)
        - 96: commanded yaw rate
        """
        agent, state = self._agent, self._agent.state
        history = agent.joint_history  # history[-1] is the current state
        height = (state.base_position[2] - self.target_body_to_feet_height) / agent.max_norm_height
        observation = np.concatenate(
            [
                self._get_gravity_vector(),
                [height],
                state.linear_velocity / agent.max_linear_velocity,
                state.angular_velocity / agent.max_angular_velocity,
                self._joint_positions_norm(state.joint_positions),
                self._joint_velocities_norm(state.joint_velocities),
                self._joint_positions_norm(history[-2][0]),
                self._joint_velocities_norm(history[-2][1]),
                self._joint_positions_norm(history[-5][0]),
                self._joint_velocities_norm(history[-5][1]),
                agent.last_action,
                agent.command_input.as_array,
            ]
        ).astype(np.float32)
        assert observation.shape == (self.OBS_SIZE,), observation.shape
        for observation_filter in self.observation_filters:
            observation = observation_filter(observation)
        return observation

    def _termination(self) -> str | None:
        """Why the episode ends at this step: "fall" (base height out of range), "tipping"
        (roll or pitch too large), or None if it goes on."""
        p = self.params
        height = self._agent.state.base_position[2]
        roll, pitch, _ = self._agent.state.roll_pitch_yaw
        if height <= p.min_height or height > p.max_height:
            return "fall"
        if abs(roll) > p.max_pitchroll or abs(pitch) > p.max_pitchroll:
            return "tipping"
        return None

    def _get_info(self) -> dict[str, Any]:
        return {
            "height": self._agent.state.base_position[2],
            "pitch": self._agent.state.roll_pitch_yaw[1],
            "episode_step": self._episode_step_counter,
        }

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def num_steps(self) -> int:
        """Environment steps taken since creation (carried over when a run is resumed)."""
        return self._total_steps_counter

    @num_steps.setter
    def num_steps(self, value: int) -> None:
        self._total_steps_counter = int(value)

    def set_num_steps(self, value: int) -> None:
        """Same as setting ``num_steps``; callable through wrappers and ``VecEnv.env_method``."""
        self.num_steps = value

    @property
    def episode_reward_info(self) -> list[dict[str, float]]:
        """Reward components of every step of the current episode."""
        return self._episode_reward_info

    @property
    def episode_length(self) -> int:
        """Steps taken in the current episode."""
        return self._episode_step_counter
