import logging
import mmap
import multiprocessing
import os
import platform
import signal
import time
import traceback
from collections.abc import Callable, Sequence
from multiprocessing.connection import Connection
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

_logger = logging.getLogger(__name__)

_STEP = 0  # request kinds
_COMMAND = 1

_WORKER_SPIN_S = 0.005  # busy-poll this long, then poll with sleeps
_WORKER_IDLE_S = 0.05  # after this long, sleep longer between polls
_MAIN_SPIN_S = 0.02
_LIVENESS_CHECK_S = 0.5


def available() -> bool:
    """Whether this platform supports SharedMemoryVecEnv."""
    return "fork" in multiprocessing.get_all_start_methods() and platform.machine().lower() in (
        "x86_64",
        "amd64",
    )


class _Buffers:
    """Shared arrays, laid out in one anonymous shared mapping created before forking."""

    def __init__(self, n_envs: int, obs_shape: tuple[int, ...], act_shape: tuple[int, ...]) -> None:
        fields: list[tuple[str, tuple[int, ...], Any]] = [
            ("request", (n_envs,), np.int64),  # trainer: bumped for each request
            ("done", (n_envs,), np.int64),  # worker: set to request when it is served
            ("kind", (n_envs,), np.int64),  # trainer: _STEP or _COMMAND
            ("has_message", (n_envs,), np.int64),  # worker: a message waits in the pipe
            ("actions", (n_envs, *act_shape), np.float32),
            ("obs", (n_envs, *obs_shape), np.float32),
            ("rewards", (n_envs,), np.float32),
            ("dones", (n_envs,), np.bool_),
        ]
        sizes = [int(np.prod(shape)) * np.dtype(dtype).itemsize for _, shape, dtype in fields]
        aligned = [(size + 63) // 64 * 64 for size in sizes]
        self._mmap = mmap.mmap(-1, sum(aligned))  # anonymous + shared: inherited on fork
        offset = 0
        for (name, shape, dtype), size in zip(fields, aligned):
            setattr(self, name, np.ndarray(shape, dtype, buffer=self._mmap, offset=offset))
            offset += size

    request: np.ndarray
    done: np.ndarray
    kind: np.ndarray
    has_message: np.ndarray
    actions: np.ndarray
    obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def close(self) -> None:
        for name in (
            "request",
            "done",
            "kind",
            "has_message",
            "actions",
            "obs",
            "rewards",
            "dones",
        ):
            setattr(self, name, None)
        self._mmap.close()


def _wait_for_request(buffers: _Buffers, index: int, last: int, parent: int) -> int | None:
    """Worker side: the next request number, or None if the trainer process is gone."""
    request = buffers.request
    start = time.perf_counter()
    while True:
        for _ in range(256):
            value = int(request[index])
            if value != last:
                return value
        waited = time.perf_counter() - start
        if waited > _WORKER_SPIN_S:
            time.sleep(0.0001 if waited < _WORKER_IDLE_S else 0.001)
            if os.getppid() != parent:
                return None


def _worker(
    index: int,
    env_fn: Callable[[], gym.Env],
    conn: Connection,
    buffers: _Buffers,
    parent: int,
) -> None:
    env = None
    try:
        env = env_fn()
        error = None
    except Exception:  # noqa: BLE001 - reported to the trainer at its first request
        error = traceback.format_exc()
    last = 0
    while True:
        request = _wait_for_request(buffers, index, last, parent)
        if request is None:
            return
        last = request
        try:
            if buffers.kind[index] == _STEP:
                if env is None:  # it could not be created: report why
                    raise RuntimeError(error)
                obs, reward, terminated, truncated, info = env.step(buffers.actions[index].copy())
                done = terminated or truncated
                if done:
                    info["TimeLimit.truncated"] = truncated and not terminated
                    info["terminal_observation"] = obs
                    obs, _ = env.reset()
                    conn.send(("info", info))
                    buffers.has_message[index] = 1
                buffers.obs[index] = obs
                buffers.rewards[index] = reward
                buffers.dones[index] = done
            else:
                command, args = (
                    conn.recv()
                )  # always read: the next command must find the pipe empty
                if command == "close":  # also when the environment could not be created
                    if env is not None:
                        env.close()
                    conn.send(("ok", None))
                    buffers.done[index] = request
                    return
                if env is None:
                    raise RuntimeError(error)
                conn.send(("ok", _run_command(env, command, args)))
        except Exception:  # noqa: BLE001 - any environment error is sent to the trainer
            conn.send(("error", f"environment {index}:\n{traceback.format_exc()}"))
            if buffers.kind[index] == _STEP:  # a command's reply is read right away
                buffers.has_message[index] = 1
        buffers.done[index] = request


def _run_command(env: gym.Env, command: str, args: Any) -> Any:
    if command == "reset":
        seed, options = args
        return env.reset(seed=seed, options=options or None)
    if command == "env_method":
        name, method_args, method_kwargs = args
        return env.get_wrapper_attr(name)(*method_args, **method_kwargs)
    if command == "get_attr":
        return env.get_wrapper_attr(args)
    if command == "set_attr":
        name, value = args
        return setattr(env, name, value)
    if command == "is_wrapped":
        return is_wrapped(env, args)
    raise NotImplementedError(command)


class SharedMemoryVecEnv(VecEnv):
    """Environments in worker processes, stepped through shared memory (see module docs).

    ``env_fns`` create the environments (in the workers); their observation and action
    spaces must be ``observation_space`` and ``action_space``, which size the shared
    arrays before the workers start.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
    ) -> None:
        if not available():
            raise RuntimeError("SharedMemoryVecEnv needs the fork start method and x86-64")
        n = len(env_fns)
        cpus = os.cpu_count() or 1
        if n > cpus - 4:
            _logger.warning(
                "%d environment workers on %d CPU threads: polling workers compete for the CPU "
                "and can slow everything down (best measured: about 2/3 of the threads)",
                n,
                cpus,
            )
        self.num_envs = n
        self._closed = False
        self._buffers = _Buffers(n, observation_space.shape, action_space.shape)
        context = multiprocessing.get_context("fork")
        self._conns: list[Connection] = []
        self._processes = []
        for index, env_fn in enumerate(env_fns):
            parent_conn, child_conn = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(index, env_fn, child_conn, self._buffers, os.getpid()),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._processes.append(process)
        try:
            super().__init__(n, observation_space, action_space)  # queries the workers
            # Every environment was created; check that it matches the given spaces
            obs_spaces = self.get_attr("observation_space")
            act_spaces = self.get_attr("action_space")
            for index, (obs, act) in enumerate(zip(obs_spaces, act_spaces)):
                if obs.shape != observation_space.shape or act.shape != action_space.shape:
                    raise ValueError(f"environment {index} has other observation/action spaces")
        except BaseException:
            self.close()
            raise

    # ── Synchronization ──────────────────────────────────────────────────────

    def _wait(self) -> None:
        """Wait until every worker has served its latest request."""
        request, done = self._buffers.request, self._buffers.done
        start = last_check = time.perf_counter()
        while not np.array_equal(done, request):
            now = time.perf_counter()
            if now - start > _MAIN_SPIN_S:
                time.sleep(0.0001)
            if now - last_check > _LIVENESS_CHECK_S:
                last_check = now
                for index, process in enumerate(self._processes):
                    if not process.is_alive() and done[index] != request[index]:
                        raise RuntimeError(f"environment worker {index} died")

    def _take_message(self, index: int) -> tuple[str, Any]:
        self._buffers.has_message[index] = 0
        return self._conns[index].recv()

    def _settle(self) -> None:
        """Finish any step left running (e.g. by Ctrl+C) and drop its stale messages."""
        self._wait()
        for index in np.flatnonzero(self._buffers.has_message):
            self._take_message(int(index))

    def _command_all(
        self, command: str, args: Sequence[Any], indices: Sequence[int] = ()
    ) -> list[Any]:
        """Run a pipe command in workers ``indices`` (default: all), one arg per worker."""
        indices = list(indices) or list(range(self.num_envs))
        self._settle()
        buffers = self._buffers
        # A Ctrl+C between sending a command and bumping the counter would desynchronize
        # the worker: hold SIGINT until both are done
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            for index, arg in zip(indices, args):
                self._conns[index].send((command, arg))
                buffers.kind[index] = _COMMAND
                buffers.request[index] += 1
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
        replies = [self._conns[index].recv() for index in indices]  # read all, even on errors
        self._wait()
        for status, value in replies:
            if status == "error":
                raise RuntimeError(value)
        return [value for _, value in replies]

    # ── VecEnv API ───────────────────────────────────────────────────────────

    def reset(self) -> VecEnvObs:
        args = [(self._seeds[i], self._options[i]) for i in range(self.num_envs)]
        results = self._command_all("reset", args)
        self._reset_seeds()
        self._reset_options()
        obs = np.stack([np.asarray(o, dtype=np.float32) for o, _ in results])
        self.reset_infos = [info for _, info in results]
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._settle()
        buffers = self._buffers
        np.copyto(buffers.actions, np.asarray(actions).reshape(buffers.actions.shape), "same_kind")
        buffers.kind[:] = _STEP
        buffers.request += 1  # the actions are written first (x86-64 keeps the order)

    def step_wait(self) -> VecEnvStepReturn:
        self._wait()
        buffers = self._buffers
        infos: list[dict[str, Any]] = [{"TimeLimit.truncated": False} for _ in range(self.num_envs)]
        error = None
        for index in np.flatnonzero(buffers.has_message):
            status, value = self._take_message(int(index))
            if status == "info":
                infos[index] = value
            elif error is None:
                error = value
        if error is not None:
            raise RuntimeError(error)
        return buffers.obs.copy(), buffers.rewards.copy(), buffers.dones.copy(), infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._command_all("close", [None] * self.num_envs)
        except (RuntimeError, OSError, EOFError):
            _logger.warning("Error while closing the environment workers", exc_info=True)
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for conn in self._conns:
            conn.close()
        self._buffers.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        targets = self._get_indices(indices)
        return self._command_all("get_attr", [attr_name] * len(targets), targets)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        targets = self._get_indices(indices)
        self._command_all("set_attr", [(attr_name, value)] * len(targets), targets)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        targets = self._get_indices(indices)
        arg = (method_name, method_args, method_kwargs)
        return self._command_all("env_method", [arg] * len(targets), targets)

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> list[bool]:
        targets = self._get_indices(indices)
        return self._command_all("is_wrapped", [wrapper_class] * len(targets), targets)

    def _get_indices(self, indices: VecEnvIndices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)
