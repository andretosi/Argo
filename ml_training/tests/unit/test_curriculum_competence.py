import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.curriculum import CompetenceProgress


def dummy_env() -> SimpleNamespace:
    """Just the attributes CompetenceProgress reads from SpotmicroEnv."""
    return SimpleNamespace(
        reward=SimpleNamespace(
            params=SimpleNamespace(tracking_lin_vel=1.25, tracking_ang_vel=0.75)
        ),
        last_episode_length=1500,
        max_episode_len=3000,
        last_episode_reward_info=[
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
        ],
    )


def tracker(**state: float) -> CompetenceProgress:
    progress = CompetenceProgress(
        CompetenceProgress.Params(
            threshold=0.70, advance_step=0.05, min_episodes_between_advances=10
        )
    )
    progress.load_state_dict({**progress.state_dict(), **state})  # the given values, else defaults
    return progress


class CompetenceProgressTestCase(unittest.TestCase):
    def test_score_episode(self) -> None:
        scores = tracker().score_episode(dummy_env())

        self.assertGreater(scores["tracking"], 0.0)
        self.assertLessEqual(scores["tracking"], 1.0)
        self.assertAlmostEqual(scores["survival"], 0.5)
        self.assertGreater(scores["stability"], 0.0)
        self.assertLessEqual(scores["stability"], 1.0)
        self.assertGreater(scores["raw"], 0.0)
        self.assertLessEqual(scores["raw"], 1.0)

    def test_advances_only_above_threshold(self) -> None:
        progress = tracker(value=0.10, ema=0.60, episodes=9, last_advance_episode=0)
        progress.update(0.60)  # 10th episode, EMA stays below the threshold
        self.assertAlmostEqual(progress.value, 0.10)
        self.assertLess(progress.ema, 0.70)

        progress = tracker(value=0.10, ema=0.80, episodes=9, last_advance_episode=0)
        progress.update(0.90)  # 10th episode, EMA above the threshold
        self.assertGreaterEqual(progress.ema, 0.70)
        self.assertAlmostEqual(progress.value, 0.15)

    def test_waits_between_advances(self) -> None:
        progress = tracker(value=0.10, ema=0.90, episodes=5, last_advance_episode=0)
        progress.update(0.90)  # only 6 episodes since the last advance
        self.assertAlmostEqual(progress.value, 0.10)

    def test_progress_never_decreases(self) -> None:
        progress = tracker(value=0.40, ema=0.90, episodes=20, last_advance_episode=20)
        progress.update(0.10)
        self.assertAlmostEqual(progress.value, 0.40)

    def test_state_round_trip(self) -> None:
        progress = tracker(
            value=0.35, ema=0.72, raw_score=0.8, episodes=42, last_advance_episode=40
        )
        copy = tracker()
        copy.load_state_dict(progress.state_dict())
        self.assertEqual(copy.state_dict(), progress.state_dict())


if __name__ == "__main__":
    unittest.main()
