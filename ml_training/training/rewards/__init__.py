"""Reward functions, by the name a run config uses under ``reward:``.

To add a reward: write a ``Reward`` subclass in this package with a ``name`` and a nested
``Params(RewardConfig)`` dataclass, then add it to ``REWARDS``.
"""

from typing import Any

from training.rewards.base import Reward, RewardConfig
from training.rewards.jumping import JumpingReward
from training.rewards.robust_walking import RobustWalkingReward
from training.rewards.standing import StandingReward
from training.rewards.trotting import TrottingReward
from training.rewards.walking import WalkingReward

REWARDS: dict[str, type[Reward]] = {
    reward.name: reward
    for reward in (
        RobustWalkingReward,
        WalkingReward,
        TrottingReward,
        StandingReward,
        JumpingReward,
    )
}


def make_reward(section: dict[str, Any]) -> Reward:
    """Build a reward from a one-key config section, e.g. ``{"standing": {...params}}``."""
    from training.config import build_component

    return build_component(section, REWARDS, "reward").make()


__all__ = ["REWARDS", "Reward", "RewardConfig", "make_reward"]
