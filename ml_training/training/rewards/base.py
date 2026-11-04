"""Base class for reward functions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from spotmicro.env import SpotmicroEnv


@dataclass
class RewardConfig:
    """The parameters of a reward: the base of every reward's nested ``Params`` dataclass.

    The fields here are shared by every reward: they are added once, on the step the
    episode ends (see ``SpotmicroEnv.termination`` and ``SpotmicroEnv.truncated``).
    """

    name: ClassVar[str] = ""
    reward_class: ClassVar[type["Reward"]]

    fall_penalty: float = -100.0  # the base height left [min_height, max_height]
    tipping_penalty: float = -2.0  # roll or pitch went beyond max_pitchroll
    survival_bonus: float = 3.0  # the episode reached max_episode_len

    def make(self) -> "Reward":
        return self.reward_class(self)


class Reward(ABC):
    """A reward function with its own tunable parameters and per-episode state.

    Subclasses implement ``compute`` (the per-step components); calling the reward adds
    the episode-end components and returns (total, components).
    """

    name: ClassVar[str] = ""

    @dataclass
    class Params(RewardConfig):
        """No parameters by default."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "Params" in vars(cls):  # link the parameters to their reward
            cls.Params.name = cls.name
            cls.Params.reward_class = cls

    def __init__(self, params: Any = None) -> None:
        self.params = params if params is not None else self.Params()

    def reset(self, env: "SpotmicroEnv") -> None:
        """Called by the environment at the start of every episode, after the robot settles."""

    @abstractmethod
    def compute(self, env: "SpotmicroEnv", action: np.ndarray) -> dict[str, float]:
        """The reward components of the step that just happened (their sum is the reward)."""

    def __call__(self, env: "SpotmicroEnv", action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Return (total reward, per-component breakdown) for the step that just happened."""
        components = self.compute(env, action)
        if not isinstance(components, dict):
            raise TypeError(f"{type(self).__name__}.compute must return a dict of components")
        components.update(self.episode_end(env))
        return float(sum(components.values())), components

    def episode_end(self, env: "SpotmicroEnv") -> dict[str, float]:
        """Penalties for how the episode ended, and the bonus for lasting until the time limit."""
        p = self.params
        return {
            "fall_penalty": p.fall_penalty if env.termination == "fall" else 0.0,
            "tipping_penalty": p.tipping_penalty if env.termination == "tipping" else 0.0,
            "survival_bonus": p.survival_bonus if env.truncated else 0.0,
        }


def normalized_effort(env: "SpotmicroEnv") -> float:
    """Mean squared joint effort, as a fraction of each joint's max torque."""
    efforts = env.agent.state.joint_efforts
    max_torque = np.array([j.max_torque for j in env.agent.motor_joints])
    return float(np.mean((efforts / max_torque) ** 2))
