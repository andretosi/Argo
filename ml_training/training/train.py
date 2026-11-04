"""Train a policy from a run config.

From ml_training/::

    python training/train.py training/configs/robust_walking.yaml
    python training/train.py training/configs/robust_walking.yaml --name try2 --total-steps 2000000
    python training/train.py --resume training/data/robust_walking/checkpoints/step_000500000.zip

With ``run.n_envs`` > 1, that many simulations run in parallel processes.

A run writes to ``<run.output_dir>/<run.name>/``:

- ``config.yaml``: the fully resolved config (every value, defaults included)
- ``checkpoints/``: ``step_*.zip`` and ``timed_*.zip``, each with a ``.state.yaml``
  (step count and curriculum state, used by ``--resume`` and by evaluate.py)
- ``final.zip`` / ``interrupted.zip`` (+ ``.state.yaml``)
- ``logs/<session>/``: TensorBoard and CSV logs, one folder per training session
- ``tracking.yaml``: the training ID and its MLflow run (see training/tracking.py)

Each run is also logged to MLflow (Databricks by default): metrics as it trains, then the
config and the final model. ``--no-mlflow`` turns this off.
"""

import argparse
import dataclasses
import multiprocessing
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import numpy as np
import torch

# Import PyTorch's compiler stack before any MuJoCo viewer opens. With a CUDA build of
# PyTorch it loads Triton, whose bundled LLVM crashes a process that already runs the
# viewer's software OpenGL (another LLVM). The Docker image uses the CPU build, without Triton.
import torch._dynamo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from spotmicro.env import SpotmicroEnv
from training.builders import (
    env_factory,
    load_run,
    make_progress,
    read_state,
    resolve_path,
    run_dir_of,
    write_state,
)
from training.config import PPOParams, RunConfig, config_to_dict, load_config, save_config
from training.curriculum import Progress
from training.tracking import Tracker
from training.vec_env import SharedMemoryVecEnv
from training.vec_env import available as shared_memory_available


def learning_rate(ppo: PPOParams) -> float | Callable[[float], float]:
    """Constant, or linear decay from ``learning_rate`` to ``learning_rate_final``."""
    if ppo.learning_rate_final is None:
        return ppo.learning_rate
    start, end = ppo.learning_rate, ppo.learning_rate_final

    def schedule(progress_remaining: float) -> float:
        return max(progress_remaining * start, end)

    return schedule


def ppo_kwargs(ppo: PPOParams) -> dict[str, Any]:
    kwargs = dataclasses.asdict(ppo)
    kwargs.pop("learning_rate_final")
    kwargs["learning_rate"] = learning_rate(ppo)
    kwargs["policy_kwargs"] = {"log_std_init": kwargs.pop("log_std_init")}
    return kwargs


def make_vec_env(config: RunConfig) -> VecEnv:
    """``run.n_envs`` training environments: in this process if 1, else one process each."""
    n = config.run.n_envs
    if n == 1:
        return VecMonitor(DummyVecEnv([env_factory(config, 0)]))
    factories = [env_factory(config, rank, subprocess=True) for rank in range(n)]
    if shared_memory_available():
        # Steps go through shared memory (see training/vec_env.py); the workers are forked,
        # so nothing is re-imported and scripts need no `if __name__ == "__main__"` guard
        return VecMonitor(SharedMemoryVecEnv(factories, *SpotmicroEnv.spaces()))
    method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "forkserver"
    return VecMonitor(SubprocVecEnv(factories, start_method=method))


def drain_pending_replies(env: VecEnv) -> None:
    """After Ctrl+C during a step, read the replies the workers still send, so that later
    calls (saving the state, closing) get their own answers instead of stale step results.
    """
    venv = env.venv if isinstance(env, VecMonitor) else env
    if not isinstance(venv, SubprocVecEnv):
        return
    time.sleep(0.5)  # a step takes milliseconds: let every worker finish it
    for remote in venv.remotes:
        while remote.poll(0):
            remote.recv()
    venv.waiting = False


def env_steps(env: VecEnv) -> int:
    return int(env.get_attr("num_steps", indices=[0])[0])


class CurriculumCallback(BaseCallback):
    """Tracks curriculum progress and sends it to every environment's ``CurriculumEnv``.

    Progress lives here, in the training process: it advances with the steps
    (``TimeProgress``) or with the competence scores the environments report at the end of
    each episode (``CompetenceProgress``).
    """

    def __init__(self, progress: Progress, env: VecEnv) -> None:
        super().__init__()
        self.progress = progress
        self.env = env
        self._saved_state: dict[str, Any] | None = None
        self._sent = -1.0
        self._metrics: dict[str, float] = {}

    def _on_training_start(self) -> None:
        self.env.env_method("start_curriculum", self.progress.value, self._saved_state)
        self._sent = self.progress.value

    def _on_step(self) -> bool:
        self.progress.on_step(self.num_timesteps)
        ended = False
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "curriculum" in info:
                self.progress.on_episode_end(info["curriculum"]["scores"])
                self._metrics = info["curriculum"]["metrics"]
                ended = True
        # Sending is an inter-process call: only when progress has moved
        if abs(self.progress.value - self._sent) >= 1e-3 or (
            ended and self.progress.value != self._sent
        ):
            self.env.env_method("set_progress", self.progress.value)
            self._sent = self.progress.value
        if ended:
            for name, value in {**self.progress.metrics(), **self._metrics}.items():
                self.logger.record(name, value)
        return True

    def state_dict(self) -> dict[str, Any]:
        """Progress, plus the curricula state of the first environment."""
        state: dict[str, Any] = {"progress": self.progress.state_dict()}
        state.update(self.env.env_method("curriculum_state", indices=[0])[0])
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore progress now and the curricula when training starts (in every env)."""
        self.progress.load_state_dict(state["progress"])
        self._saved_state = {k: v for k, v in state.items() if k != "progress"}


class Checkpointer(BaseCallback):
    """Save the model and its state file every N steps and every M minutes."""

    def __init__(
        self,
        directory: Path,
        env: VecEnv,
        curriculum: CurriculumCallback | None,
        every_steps: int,
        every_minutes: float,
    ) -> None:
        super().__init__()
        self.directory = directory
        self.env = env
        self.curriculum = curriculum
        self.every_steps = every_steps
        self.every_minutes = every_minutes
        self._last_step_save = 0
        self._last_timed_save = time.time()

    def _on_training_start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._last_step_save = self.num_timesteps
        self._last_timed_save = time.time()

    def save(self, stem: str) -> None:
        path = self.directory / stem
        self.model.save(str(path))
        curriculum = self.curriculum.state_dict() if self.curriculum is not None else None
        write_state(
            path.with_suffix(".state.yaml"), self.num_timesteps, env_steps(self.env), curriculum
        )
        if self.verbose or self.model.verbose:
            print(f"[checkpoint] {path}.zip")

    def _on_step(self) -> bool:
        if self.every_steps > 0 and self.num_timesteps - self._last_step_save >= self.every_steps:
            self._last_step_save = self.num_timesteps
            self.save(f"step_{self.num_timesteps:09d}")
        if (
            self.every_minutes > 0
            and time.time() - self._last_timed_save >= 60 * self.every_minutes
        ):
            self._last_timed_save = time.time()
            self.save(f"timed_{time.strftime('%Y%m%d_%H%M%S')}_{self.num_timesteps:09d}")
        return True


def seed_everything(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def train(
    config: RunConfig,
    resume: str | Path | None = None,
    mlflow: bool = True,
    experiment: str | None = None,
) -> Path:
    """Train as described by ``config``; with ``resume``, continue from that checkpoint.
    Returns the path of the final model (``final.zip``, or ``interrupted.zip`` on Ctrl+C).
    """
    if resume is not None:
        run_dir = run_dir_of(resume)
    else:
        run_dir = resolve_path(config.run.output_dir) / config.run.name
    if config.run.gui and config.run.n_envs > 1:
        raise ValueError("run.gui shows one simulation: set run.n_envs to 1")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    seed_everything(config.run.seed)
    torch.set_num_threads(config.run.torch_threads)
    env = make_vec_env(config)
    curriculum = None
    if config.curriculum is not None:
        curriculum = CurriculumCallback(make_progress(config), env)
    kwargs = ppo_kwargs(config.ppo)

    if resume is not None:
        model = PPO.load(str(resume), env=env, device=config.ppo.device)
        state = read_state(resume)
        env.env_method("set_num_steps", state["env_steps"])
        if curriculum is not None:
            curriculum.load_state_dict(state["curriculum"])
        reset_num_timesteps = False
    elif config.run.init_from is not None:
        source = resolve_path(config.run.init_from)
        # The loaded policy keeps its own network and action std
        skip = ("device", "verbose", "policy_kwargs")
        custom = {k: v for k, v in kwargs.items() if k not in skip}
        model = PPO.load(str(source), env=env, device=config.ppo.device, custom_objects=custom)
        model.verbose = config.ppo.verbose
        env.env_method("set_num_steps", read_state(source)["env_steps"])
        reset_num_timesteps = True
    else:
        model = PPO("MlpPolicy", env, seed=config.run.seed, **kwargs)
        reset_num_timesteps = True

    session = time.strftime("%Y%m%d_%H%M%S")
    model.set_logger(configure(str(run_dir / "logs" / session), ["stdout", "csv", "tensorboard"]))

    checkpoints = config.run.checkpoints
    checkpointer = Checkpointer(
        run_dir / "checkpoints", env, curriculum, checkpoints.every_steps, checkpoints.every_minutes
    )
    callbacks: list[BaseCallback] = [checkpointer]
    if curriculum is not None:
        callbacks.insert(0, curriculum)

    done = 0 if reset_num_timesteps else model.num_timesteps
    remaining = config.run.total_steps - done
    if remaining <= 0:
        print(f"Nothing to do: {done:,} of {config.run.total_steps:,} steps already trained.")
        env.close()
        return run_dir / "final.zip"

    tracker = None
    if mlflow:
        tracker = Tracker(run_dir, experiment)
        try:
            training_id = tracker.start(config_to_dict(config), resume=resume is not None)
        except Exception as e:
            env.close()
            raise RuntimeError(
                f"MLflow tracking could not start ({e}). Check your Databricks credentials "
                "(DATABRICKS_HOST/DATABRICKS_TOKEN or ~/.databrickscfg), or train with "
                "--no-mlflow."
            ) from e
        model.logger.output_formats.append(tracker.output_format())
        print(f"Training ID: {training_id}  (MLflow run {tracker.run_id})")

    print(
        f"Run: {run_dir}  ({remaining:,} steps to train, reward: {config.reward.name}, "
        f"{config.run.n_envs} environment(s))"
    )
    final = "final"
    status = "FINISHED"
    try:
        model.learn(
            total_timesteps=remaining,
            callback=CallbackList(callbacks),
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        final = "interrupted"
        status = "KILLED"
        print("\nTraining interrupted.")
        drain_pending_replies(env)
    except BaseException:
        status = "FAILED"
        raise
    finally:
        model.save(str(run_dir / final))
        curriculum_state = curriculum.state_dict() if curriculum is not None else None
        write_state(
            run_dir / f"{final}.state.yaml", model.num_timesteps, env_steps(env), curriculum_state
        )
        env.close()
        if tracker is not None:
            tracker.log_artifact(run_dir / f"{final}.zip", "model")
            tracker.log_artifact(run_dir / f"{final}.state.yaml", "model")
            tracker.end(status)
    print(f"Saved: {run_dir / final}.zip")
    return run_dir / f"{final}.zip"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", nargs="?", help="run config (YAML)")
    parser.add_argument(
        "--resume", help="checkpoint (.zip) to continue from; uses its run's config"
    )
    parser.add_argument("--name", help="override run.name")
    parser.add_argument("--total-steps", type=int, help="override run.total_steps")
    parser.add_argument(
        "--gui", action="store_true", help="show the MuJoCo viewer (trains with one environment)"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    parser.add_argument(
        "--experiment",
        help="MLflow experiment (default: $MLFLOW_EXPERIMENT_NAME, else /Shared/spotmicro)",
    )
    args = parser.parse_args()

    if args.resume:
        config = load_config(args.config) if args.config else load_run(args.resume)
    elif args.config:
        config = load_config(args.config)
    else:
        parser.error("give a config file, or --resume CHECKPOINT")
    if args.name:
        config.run.name = args.name
    if args.total_steps:
        config.run.total_steps = args.total_steps
    if args.gui:
        config.run.gui = True
        config.run.n_envs = 1  # the viewer shows one simulation
    train(config, resume=args.resume, mlflow=not args.no_mlflow, experiment=args.experiment)


if __name__ == "__main__":
    main()
