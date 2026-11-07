"""SharedMemoryVecEnv against SB3's DummyVecEnv, on a tiny deterministic environment."""

import multiprocessing
import sys
import time
import unittest
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3.common.vec_env import DummyVecEnv

from training.vec_env import SharedMemoryVecEnv, available

OBS_SPACE = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
ACT_SPACE = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


class CountingEnv(gym.Env):
    """obs = (rank, step in episode, last action sum); episodes last ``rank + 3`` steps."""

    observation_space = OBS_SPACE
    action_space = ACT_SPACE

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.t = 0
        self.scale = 1.0

    def _obs(self, action_sum: float = 0.0) -> np.ndarray:
        return np.array([self.rank, self.t, action_sum], dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {"seed": seed}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if np.isnan(action).any():
            raise ValueError("NaN action")
        self.t += 1
        total = float(np.sum(action))
        truncated = self.t >= self.rank + 3
        return self._obs(total), self.scale * total, False, truncated, {"t": self.t}

    def double(self, x: float) -> float:
        return 2 * x


def factories(n: int) -> list:
    return [lambda rank=rank: CountingEnv(rank) for rank in range(n)]


@unittest.skipUnless(available(), "needs fork and x86-64")
class TestSharedMemoryVecEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SharedMemoryVecEnv(factories(3), OBS_SPACE, ACT_SPACE)

    def tearDown(self) -> None:
        self.env.close()

    def test_same_results_as_dummy_vec_env(self) -> None:
        reference = DummyVecEnv(factories(3))
        rng = np.random.default_rng(0)
        np.testing.assert_array_equal(self.env.reset(), reference.reset())
        for _ in range(20):
            actions = rng.uniform(-1, 1, (3, 2)).astype(np.float32)
            obs, rewards, dones, infos = self.env.step(actions)
            ref_obs, ref_rewards, ref_dones, ref_infos = reference.step(actions)
            np.testing.assert_array_equal(obs, ref_obs)
            np.testing.assert_allclose(rewards, ref_rewards)
            np.testing.assert_array_equal(dones, ref_dones)
            for info, ref_info in zip(infos, ref_infos):
                self.assertEqual(info["TimeLimit.truncated"], ref_info["TimeLimit.truncated"])
                if "terminal_observation" in ref_info:
                    np.testing.assert_array_equal(
                        info["terminal_observation"], ref_info["terminal_observation"]
                    )
                    self.assertEqual(info["t"], ref_info["t"])  # full info at episode end

    def test_commands(self) -> None:
        self.assertEqual(self.env.get_attr("rank"), [0, 1, 2])
        self.assertEqual(self.env.env_method("double", 4, indices=[1]), [8])
        self.env.set_attr("scale", 10.0, indices=[2])
        self.env.reset()
        _, rewards, _, _ = self.env.step(np.ones((3, 2), dtype=np.float32))
        np.testing.assert_allclose(rewards, [2.0, 2.0, 20.0])
        self.assertEqual(self.env.env_is_wrapped(gym.Wrapper), [False, False, False])

    def test_seeds_reach_reset(self) -> None:
        self.env.seed(7)
        self.env.reset()
        self.assertEqual([info["seed"] for info in self.env.reset_infos], [7, 8, 9])

    def test_errors_are_raised_and_the_env_stays_usable(self) -> None:
        self.env.reset()
        actions = np.zeros((3, 2), dtype=np.float32)
        actions[1, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "environment 1"):
            self.env.step(actions)
        with self.assertRaisesRegex(RuntimeError, "no_such_method"):
            self.env.env_method("no_such_method")
        obs, _, _, _ = self.env.step(np.zeros((3, 2), dtype=np.float32))
        self.assertEqual(obs.shape, (3, 3))

    def test_interrupted_step_does_not_desynchronize(self) -> None:
        # As if Ctrl+C hit between step_async and step_wait, at an episode end (pending info)
        self.env.reset()
        for _ in range(2):
            self.env.step(np.zeros((3, 2), dtype=np.float32))
        self.env.step_async(np.zeros((3, 2), dtype=np.float32))  # env 0 finishes its episode
        self.assertEqual(self.env.get_attr("rank"), [0, 1, 2])
        obs, _, _, _ = self.env.step(np.zeros((3, 2), dtype=np.float32))
        self.assertEqual(obs[0, 1], 1)  # env 0 is one step into its next episode

    def test_failing_factory(self) -> None:
        def broken() -> gym.Env:
            raise OSError("no model file")

        before = set(multiprocessing.active_children())  # setUp's environment
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "no model file"):
            SharedMemoryVecEnv([broken, *factories(1)], OBS_SPACE, ACT_SPACE)
        # Every worker obeyed "close", the broken one too: no waiting to kill it, none left
        self.assertLess(time.monotonic() - start, 3.0)
        self.assertEqual(set(multiprocessing.active_children()), before)


if __name__ == "__main__":
    unittest.main()
