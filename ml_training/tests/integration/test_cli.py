"""train.py and evaluate.py from the command line, and the less common training paths:
init_from, Ctrl+C, failures, the pipe-based fallback, the viewer (mocked: no display)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from tests.integration.test_training import tiny_config
from training import evaluate as evaluate_script
from training import train as train_script
from training.builders import read_state
from training.config import PPOParams, config_from_dict, load_config


def write_config(tmp: str, data: dict) -> Path:
    path = Path(tmp) / f"{data['run']['name']}.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def standing_config(tmp: str, name: str = "stand") -> dict:
    """A tiny run without curricula."""
    return {
        "run": {
            "name": name,
            "total_steps": 128,
            "output_dir": tmp,
            "checkpoints": {"every_steps": 0, "every_minutes": 0},
        },
        "env": {"max_episode_len": 30},
        "command": {"fixed": {"mode": "still"}},
        "reward": {"standing": {}},
        "ppo": {"n_steps": 64, "batch_size": 32, "n_epochs": 1, "verbose": 0},
    }


def run_main(module: object, *args: str) -> None:
    with mock.patch.object(sys, "argv", ["prog", *map(str, args)]):
        module.main()


class TrainCommandLineTestCase(unittest.TestCase):
    def test_train_resume_and_nothing_left_to_do(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = write_config(tmp, standing_config(tmp))
            run_main(train_script, config, "--no-mlflow", "--name", "cli", "--total-steps", "64")
            final = Path(tmp) / "cli" / "final.zip"
            self.assertEqual(read_state(final)["num_timesteps"], 64)

            run_main(train_script, "--resume", final, "--no-mlflow", "--total-steps", "128")
            self.assertEqual(read_state(final)["num_timesteps"], 128)

            with mock.patch("builtins.print") as printed:
                run_main(train_script, "--resume", final, "--no-mlflow")  # already 128 of 128
            self.assertIn("Nothing to do", str(printed.call_args_list))

    def test_needs_a_config_or_a_checkpoint(self) -> None:
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            run_main(train_script)

    def test_gui_trains_one_environment_with_the_viewer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = standing_config(tmp)
            data["run"]["n_envs"] = 4
            with mock.patch("mujoco.viewer.launch_passive") as viewer:
                run_main(train_script, write_config(tmp, data), "--gui", "--no-mlflow")
            viewer.assert_called_once()
            self.assertEqual(load_config(Path(tmp) / "stand" / "config.yaml").run.n_envs, 1)

            config = config_from_dict(data)
            config.run.gui = True
            with self.assertRaisesRegex(ValueError, "n_envs"):
                train_script.train(config, mlflow=False)


class TrainPathsTestCase(unittest.TestCase):
    def test_init_from_another_run_with_new_hyperparameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = standing_config(tmp, "a")
            data["ppo"]["log_std_init"] = -1.5  # initial action std
            source = train_script.train(config_from_dict(data), mlflow=False)
            np.testing.assert_allclose(PPO.load(source).policy.log_std.detach(), -1.5, atol=0.05)
            data = standing_config(tmp, "b")
            data["run"]["init_from"] = str(source)
            data["ppo"]["learning_rate"] = 1e-4
            final = train_script.train(config_from_dict(data), mlflow=False)
            model = PPO.load(final)
            self.assertAlmostEqual(model.learning_rate, 1e-4)
            self.assertEqual(read_state(final)["num_timesteps"], 128)  # counted afresh
            # The loaded policy keeps its action std (b's log_std_init, 0, is not applied)
            np.testing.assert_allclose(model.policy.log_std.detach(), -1.5, atol=0.05)

    def test_ctrl_c_saves_an_interrupted_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = config_from_dict(standing_config(tmp))
            with mock.patch.object(PPO, "learn", side_effect=KeyboardInterrupt):
                path = train_script.train(config, mlflow=False)
            self.assertEqual(path.name, "interrupted.zip")
            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".state.yaml").exists())

    def test_failure_still_saves_and_reports_to_mlflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = config_from_dict(standing_config(tmp))
            with (
                mock.patch.object(PPO, "learn", side_effect=RuntimeError("boom")),
                mock.patch.object(train_script, "Tracker") as tracker,
                self.assertRaisesRegex(RuntimeError, "boom"),
            ):
                tracker.return_value.start.return_value = "id"
                tracker.return_value.output_format.return_value = mock.Mock(spec=[])
                train_script.train(config)
            self.assertTrue((Path(tmp) / "stand" / "final.zip").exists())
            tracker.return_value.end.assert_called_once_with("FAILED")

    def test_mlflow_that_cannot_start_stops_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = config_from_dict(standing_config(tmp))
            with (
                mock.patch.object(train_script, "Tracker") as tracker,
                self.assertRaisesRegex(RuntimeError, "--no-mlflow"),
            ):
                tracker.return_value.start.side_effect = ConnectionError("no network")
                train_script.train(config)

    def test_pipe_fallback_with_time_progress_and_timed_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = tiny_config(tmp)
            data["run"].update(n_envs=2, total_steps=128)
            data["run"]["checkpoints"] = {"every_steps": 0, "every_minutes": 1e-9}
            data["curriculum"]["progression"] = "time"
            data["curriculum"].pop("competence")
            config = config_from_dict(data)
            with mock.patch.object(train_script, "shared_memory_available", return_value=False):
                env = train_script.make_vec_env(config)
                self.assertIsInstance(env.venv, SubprocVecEnv)
                env.close()
                final = train_script.train(config, mlflow=False)
            state = read_state(final)
            self.assertAlmostEqual(state["curriculum"]["progress"]["value"], 1.0)
            self.assertTrue(list((Path(tmp) / "tiny" / "checkpoints").glob("timed_*.zip")))

    def test_drain_pending_replies_after_an_interrupted_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = tiny_config(tmp)
            data["run"]["n_envs"] = 2
            config = config_from_dict(data)
            with mock.patch.object(train_script, "shared_memory_available", return_value=False):
                env = train_script.make_vec_env(config)
            try:
                env.reset()
                env.step_async(np.zeros((2, 12), dtype=np.float32))  # Ctrl+C before step_wait
                train_script.drain_pending_replies(env)
                self.assertEqual(train_script.env_steps(env), 2)  # its own reply, not a step's
            finally:
                env.close()
            shared = train_script.make_vec_env(config)  # shared memory: nothing to drain
            train_script.drain_pending_replies(shared)
            shared.close()

    def test_constant_learning_rate(self) -> None:
        self.assertEqual(train_script.learning_rate(PPOParams(learning_rate_final=None)), 3e-4)
        schedule = train_script.learning_rate(PPOParams(learning_rate=1.0, learning_rate_final=0.1))
        self.assertEqual((schedule(1.0), schedule(0.5), schedule(0.0)), (1.0, 0.5, 0.1))


class EvaluateTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        data = tiny_config(cls.tmp.name)
        data["run"]["total_steps"] = 64
        cls.walker = train_script.train(config_from_dict(data), mlflow=False)
        cls.stander = train_script.train(
            config_from_dict(standing_config(cls.tmp.name)), mlflow=False
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_command_line_with_overrides(self) -> None:
        with mock.patch("builtins.print") as printed:
            run_main(
                evaluate_script,
                self.walker,
                "--no-gui",
                "--episodes",
                "1",
                "--max-episode-len",
                "5",
                "--progress",
                "1.0",
                "--push",
                "0.5",
                "--terrain-z-max",
                "0.03",
                "--command",
                "fixed",
                "--stochastic",
            )
        output = str(printed.call_args_list)
        self.assertIn("'velocity': 0.5", output)
        self.assertIn("Mean reward", output)

    def test_flat_with_a_curriculum_the_run_did_not_have(self) -> None:
        curricula = evaluate_script.evaluation_curricula(
            load_config(self.stander.parent / "config.yaml"),
            self.stander,
            flat=True,
            overrides={"friction": 0.4},
        )
        self.assertEqual(list(curricula), ["friction"])
        self.assertEqual(curricula["friction"].friction, 0.4)

    def test_gui_runs_in_real_time(self) -> None:
        with (
            mock.patch("mujoco.viewer.launch_passive"),
            mock.patch.object(evaluate_script.time, "sleep") as sleep,
        ):
            rewards = evaluate_script.evaluate(self.stander, episodes=1, max_episode_len=3)
        self.assertEqual(len(rewards), 1)
        self.assertEqual(sleep.call_count, 3)

    def test_models_outside_run_folders_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loose = Path(tmp) / "model.zip"
            loose.write_bytes(self.stander.read_bytes())
            with self.assertRaisesRegex(FileNotFoundError, "run folder"):
                evaluate_script.evaluate(loose, gui=False)


if __name__ == "__main__":
    unittest.main()
