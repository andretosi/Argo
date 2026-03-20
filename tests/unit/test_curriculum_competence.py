import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spotmicro.tools.config import Config
from training.callbacks import CompetenceTrackerCallback


class DummyRewardConfig:
    tracking_lin_vel = 1.25
    tracking_ang_vel = 0.75


class DummyRewardState:
    config = DummyRewardConfig()


class DummyEnv:
    def __init__(self):
        self.reward_state = DummyRewardState()
        self._episode_step_counter = 1500
        self.max_episode_len = 3000
        self._episode_reward_info = [
            {
                "tracking_lin_vel": 1.25,
                "tracking_ang_vel": 0.75,
                "orientation": -0.1,
                "base_height": -0.2,
            },
            {
                "tracking_lin_vel": 1.00,
                "tracking_ang_vel": 0.60,
                "orientation": -0.2,
                "base_height": -0.1,
            },
        ]


class DummyModel:
    def __init__(self, num_timesteps=0):
        self.num_timesteps = num_timesteps


class CompetenceTrackerTestCase(unittest.TestCase):
    def test_compute_episode_scores(self):
        cfg = Config()
        env = DummyEnv()
        callback = CompetenceTrackerCallback(config=cfg, env=env, verbose=False)

        tracking, survival, stability, raw = callback._compute_episode_scores(env)

        self.assertGreater(tracking, 0.0)
        self.assertLessEqual(tracking, 1.0)
        self.assertAlmostEqual(survival, 0.5)
        self.assertGreater(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
        self.assertGreater(raw, 0.0)
        self.assertLessEqual(raw, 1.0)

    def test_update_competence_advances_only_above_threshold(self):
        cfg = Config()
        env = DummyEnv()
        callback = CompetenceTrackerCallback(
            config=cfg,
            env=env,
            threshold=0.70,
            advance_step=0.05,
            min_episodes_between_advances=10,
            competence_progress=0.10,
            competence_ema=0.60,
            verbose=False,
        )
        callback.model = DummyModel()
        callback._episode_count = 10

        competence_ema, progress = callback._update_competence(0.60)
        self.assertAlmostEqual(progress, 0.10)
        self.assertLess(competence_ema, 0.70)

        callback.competence_ema = 0.80
        callback.competence_progress = 0.10
        callback.last_advance_episode = 0
        callback._episode_count = 10

        competence_ema, progress = callback._update_competence(0.90)
        self.assertGreaterEqual(competence_ema, 0.70)
        self.assertAlmostEqual(progress, 0.15)

    def test_progress_never_decreases(self):
        cfg = Config()
        env = DummyEnv()
        callback = CompetenceTrackerCallback(
            config=cfg,
            env=env,
            threshold=0.70,
            advance_step=0.05,
            min_episodes_between_advances=10,
            competence_progress=0.40,
            competence_ema=0.90,
            last_advance_episode=20,
            verbose=False,
        )
        callback.model = DummyModel()
        callback._episode_count = 21

        _, progress = callback._update_competence(0.10)
        self.assertAlmostEqual(progress, 0.40)


if __name__ == "__main__":
    unittest.main()
