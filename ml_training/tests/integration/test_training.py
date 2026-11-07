"""End-to-end: a tiny training run, a resume from its checkpoint, and an evaluation (MuJoCo, headless)."""

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3.common.vec_env import SubprocVecEnv

from spotmicro.env import SpotmicroEnv
from training.builders import env_factory, read_state
from training.config import config_from_dict, load_config
from training.evaluate import evaluate
from training.train import make_vec_env, train
from training.vec_env import SharedMemoryVecEnv
from training.vec_env import available as shared_memory_available


def tiny_config(output_dir: str) -> dict:
    with open(ROOT_DIR / "training" / "configs" / "robust_walking.yaml") as f:
        data = yaml.safe_load(f)
    data["run"] = {
        "name": "tiny",
        "total_steps": 256,
        "output_dir": output_dir,
        "checkpoints": {"every_steps": 128, "every_minutes": 0},
    }
    data["env"]["max_episode_len"] = 40
    data["ppo"].update(n_steps=64, batch_size=32, n_epochs=1, verbose=0)
    return data


class TrainingTestCase(unittest.TestCase):
    def test_train_resume_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            final = train(config_from_dict(tiny_config(tmp)), mlflow=False)
            run_dir = Path(tmp) / "tiny"
            self.assertEqual(final, run_dir / "final.zip")
            self.assertTrue(final.exists())
            self.assertEqual(load_config(run_dir / "config.yaml").reward.name, "robust_walking")

            checkpoint = run_dir / "checkpoints" / "step_000000128.zip"
            state = read_state(checkpoint)
            self.assertEqual(state["num_timesteps"], 128)
            self.assertIn("terrain", state["curriculum"])

            config = load_config(run_dir / "config.yaml")
            config.run.total_steps = 320
            train(config, resume=checkpoint, mlflow=False)
            self.assertEqual(read_state(run_dir / "final.zip")["num_timesteps"], 320)

            rewards = evaluate(final, episodes=1, gui=False, max_episode_len=20)
            self.assertEqual(len(rewards), 1)

    def test_parallel_environments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = tiny_config(tmp)
            data["run"]["n_envs"] = 2
            config = config_from_dict(data)

            # Each worker has its own random stream (different random commands, obs 94-96)
            env = make_vec_env(config)
            try:
                env.reset()
                commands = []
                for _ in range(30):
                    obs, *_ = env.step(np.zeros((2, 12), dtype=np.float32))
                    commands.append(obs[:, 94:97])
                commands = np.array(commands)
                self.assertFalse(np.allclose(commands[:, 0], commands[:, 1]))
            finally:
                env.close()

            final = train(config, mlflow=False)
            state = read_state(final)
            self.assertEqual(state["num_timesteps"], 256)
            self.assertEqual(state["env_steps"], 256)  # counted over both environments
            self.assertIn("terrain", state["curriculum"])

            config = load_config(Path(tmp) / "tiny" / "config.yaml")
            self.assertEqual(config.run.n_envs, 2)
            config.run.total_steps = 384
            train(
                config,
                resume=Path(tmp) / "tiny" / "checkpoints" / "step_000000128.zip",
                mlflow=False,
            )
            self.assertEqual(read_state(final)["num_timesteps"], 384)

    def test_shared_memory_matches_pipes(self) -> None:
        """SharedMemoryVecEnv gives exactly what SB3's pipe-based SubprocVecEnv gives."""
        if not shared_memory_available():
            self.skipTest("needs fork and x86-64")
        with tempfile.TemporaryDirectory() as tmp:
            data = tiny_config(tmp)
            data["run"]["n_envs"] = 3
            config = config_from_dict(data)
            factories = [env_factory(config, rank, subprocess=True) for rank in range(3)]
            envs = [
                SharedMemoryVecEnv(factories, *SpotmicroEnv.spaces()),
                SubprocVecEnv(factories, start_method="fork"),
            ]
            results = []
            try:
                for env in envs:
                    env.env_method("start_curriculum", 0.5, None)
                    env.seed(0)
                    rng = np.random.default_rng(0)
                    obs, rewards, dones = [env.reset()], [], []
                    for _ in range(150):  # several 40-step episodes
                        actions = rng.uniform(-1, 1, (3, 12)).astype(np.float32)
                        o, r, d, infos = env.step(actions)
                        obs.append(o)
                        rewards.append(r)
                        dones.append(d)
                    results.append((np.array(obs), np.array(rewards), np.array(dones), infos))
            finally:
                for env in envs:
                    env.close()
            shared, pipes = results
            self.assertTrue(pipes[2].any())  # episodes did end
            np.testing.assert_array_equal(shared[0], pipes[0])
            np.testing.assert_allclose(shared[1], pipes[1], rtol=1e-6)
            np.testing.assert_array_equal(shared[2], pipes[2])


if __name__ == "__main__":
    unittest.main()


class MLflowTrackingTestCase(unittest.TestCase):
    """Training logs to MLflow (a local SQLite store here, not Databricks)."""

    def test_training_id_metrics_and_resume(self) -> None:
        try:
            import mlflow
        except ImportError:
            self.skipTest("mlflow is not installed")
        import getpass
        import os
        import socket
        from unittest import mock

        from training.tracking import read_tracking_state

        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "MLFLOW_TRACKING_URI": f"sqlite:///{tmp}/mlflow.db",
                "MLFLOW_EXPERIMENT_NAME": "test",
            }
            # chdir: MLflow puts artifacts in ./mlruns next to a SQLite store
            with mock.patch.dict(os.environ, env), contextlib.chdir(tmp):
                train(config_from_dict(tiny_config(tmp)))
                run_dir = Path(tmp) / "tiny"
                tracking = read_tracking_state(run_dir)
                self.assertTrue(tracking["training_id"].startswith(f"tiny-{socket.gethostname()}-"))

                client = mlflow.MlflowClient(env["MLFLOW_TRACKING_URI"])
                run = client.get_run(tracking["mlflow_run_id"])
                self.assertEqual(run.info.status, "FINISHED")
                self.assertEqual(run.info.run_name, tracking["training_id"])
                self.assertEqual(run.data.tags["training_id"], tracking["training_id"])
                self.assertEqual(run.data.tags["mlflow.user"], getpass.getuser())
                self.assertEqual(run.data.tags["machine"], socket.gethostname())
                self.assertEqual(run.data.params["run.total_steps"], "256")
                self.assertIn("rollout/ep_rew_mean", run.data.metrics)
                artifacts = {a.path for a in client.list_artifacts(run.info.run_id, "model")}
                self.assertIn("model/final.zip", artifacts)

                config = load_config(run_dir / "config.yaml")
                config.run.total_steps = 384
                train(config, resume=run_dir / "checkpoints" / "step_000000128.zip")
                self.assertEqual(read_tracking_state(run_dir), tracking)  # same training
                steps = [
                    m.step
                    for m in client.get_metric_history(run.info.run_id, "time/total_timesteps")
                ]
                self.assertEqual(max(steps), 384)
