import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.config import ConfigError, config_from_dict, config_to_dict, load_config, save_config
from training.rewards.robust_walking import RobustWalkingReward

CONFIGS = ROOT_DIR / "training" / "configs"
MINIMAL = {"command": {"fixed": {}}, "reward": {"standing": {}}}


class RunConfigTestCase(unittest.TestCase):
    def test_all_shipped_configs_load(self) -> None:
        files = sorted(CONFIGS.glob("*.yaml"))
        self.assertTrue(files)
        for path in files:
            with self.subTest(config=path.name):
                load_config(path)

    def test_defaults_fill_missing_values(self) -> None:
        config = config_from_dict(
            {"command": {"fixed": {}}, "reward": {"robust_walking": {"tracking_sigma": 0.3}}}
        )
        self.assertEqual(config.reward.name, "robust_walking")
        self.assertEqual(config.reward.tracking_sigma, 0.3)
        self.assertEqual(config.reward.alive_bonus, RobustWalkingReward.Params().alive_bonus)
        self.assertIsNone(config.curriculum)

    def test_sections_hold_each_component_s_own_params(self) -> None:
        from commands import COMMANDS, RandomController
        from training.curriculum import CURRICULA, TerrainCurriculum
        from training.rewards import REWARDS

        config = load_config(CONFIGS / "robust_walking.yaml")
        self.assertIsInstance(config.reward, RobustWalkingReward.Params)
        self.assertIsInstance(config.reward.make(), RobustWalkingReward)
        self.assertIsInstance(config.command.make(), RandomController)
        terrain = config.curriculum.curricula["terrain"]
        self.assertEqual(terrain.name, "terrain")
        self.assertIsInstance(terrain.make(), TerrainCurriculum)
        for registry in (REWARDS, COMMANDS, CURRICULA):
            for name, component in registry.items():
                with self.subTest(component=name):
                    self.assertEqual(component.Params.name, name)

    def test_lists_become_tuples_and_numbers_parse(self) -> None:
        config = config_from_dict(
            {
                **MINIMAL,
                "curriculum": {"terrain": {"z_max": [0.01, 0.1]}},
                "ppo": {"learning_rate": "3e-4"},  # PyYAML reads `3e-4` as a string
            }
        )
        self.assertEqual(config.curriculum.curricula["terrain"].z_max, (0.01, 0.1))
        self.assertEqual(config.ppo.learning_rate, 3e-4)

    def test_errors(self) -> None:
        bad = {
            "unknown reward parameter": {**MINIMAL, "reward": {"standing": {"nope": 1}}},
            "two rewards": {**MINIMAL, "reward": {"standing": {}, "jumping": {}}},
            "unknown reward": {**MINIMAL, "reward": {"flying": {}}},
            "unknown section": {**MINIMAL, "rewards": {}},
            "unknown curriculum": {**MINIMAL, "curriculum": {"wind": {}}},
            "bad progression": {**MINIMAL, "curriculum": {"progression": "vibes"}},
            "missing reward": {"command": {"fixed": {}}},
            "parameters not a mapping": {**MINIMAL, "ppo": [1, 2]},
            "not a number": {**MINIMAL, "ppo": {"learning_rate": "fast"}},
            "curriculum not a mapping": {**MINIMAL, "curriculum": "hard"},
            "no parallel environments": {**MINIMAL, "run": {"n_envs": 0}},
            "not a mapping": ["run"],
        }
        for label, data in bad.items():
            with self.subTest(label), self.assertRaises(ConfigError):
                config_from_dict(data)

    def test_file_errors_name_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("reward: {standing: {}}\n")
            with self.assertRaisesRegex(ConfigError, "bad.yaml.*command"):
                load_config(path)

    def test_save_load_round_trip(self) -> None:
        config = load_config(CONFIGS / "robust_walking.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            save_config(config, path)
            again = load_config(path)
        self.assertEqual(config_to_dict(config), config_to_dict(again))


if __name__ == "__main__":
    unittest.main()
