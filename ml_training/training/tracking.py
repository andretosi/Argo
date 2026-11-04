"""Experiment tracking with MLflow"""

import getpass
import os
import re
import secrets
import socket
import time
import warnings
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3.common.logger import KVWriter

DEFAULT_EXPERIMENT = "argo"
DATABRICKS_DEFAULT_EXPERIMENT = "/Shared/argo"
STATE_FILE = "tracking.yaml"
_MAX_PARAMS_PER_CALL = 100  # MLflow's log_batch limit
_MAX_PARAM_LENGTH = 500


def new_training_id(run_name: str, machine: str) -> str:
    """``<run name>-<machine>-<date>-<time>-<6 hex>``."""
    machine = re.sub(r"[^\w.-]+", "_", machine) or "unknown"
    return f"{run_name}-{machine}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"ppo": {"gamma": 0.99}}`` -> ``{"ppo.gamma": 0.99}``."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def metric_name(key: str) -> str:
    """MLflow allows letters, digits, ``_ - . / :`` and spaces in metric names."""
    return re.sub(r"[^\w\-./: ]", "_", key)


class MLflowOutputFormat(KVWriter):
    """SB3 logger output that sends the numeric values of each dump to MLflow.

    Logging errors (e.g. the network drops) are reported once and never stop training.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._warned = False

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        import mlflow

        metrics = {}
        for key, value in key_values.items():
            excluded = key_excluded.get(key)
            if excluded is not None and "mlflow" in excluded:
                continue
            if isinstance(value, bool) or not hasattr(value, "__float__"):
                continue  # strings, videos, figures...
            try:
                metrics[metric_name(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if not metrics:
            return
        try:
            mlflow.log_metrics(metrics, step=int(step), run_id=self.run_id, synchronous=False)
        except Exception as e:  # noqa: BLE001 - tracking must not stop training
            if not self._warned:
                warnings.warn(f"MLflow logging failed (training continues): {e}", stacklevel=2)
                self._warned = True

    def close(self) -> None:
        pass


class Tracker:
    """One MLflow run for one training run (across ``--resume`` sessions)."""

    def __init__(self, run_dir: Path, experiment: str | None = None) -> None:
        self.run_dir = run_dir
        self.experiment = experiment
        self.training_id: str | None = None
        self.run_id: str | None = None

    def start(self, config_dict: dict[str, Any], resume: bool) -> str:
        """Start (or, on resume, reopen) the MLflow run; returns the training ID."""
        os.environ.setdefault("_MLFLOW_ENABLE_UC_TRACE_UPSELL", "false")
        os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
        import mlflow

        uri = os.environ.get("MLFLOW_TRACKING_URI", "databricks")
        mlflow.set_tracking_uri(uri)
        user = current_user(uri)  # on Databricks, also checks the credentials
        machine = socket.gethostname()
        experiment = self.experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if not experiment:
            experiment = default_experiment(uri)
        mlflow.set_experiment(experiment)

        saved = read_tracking_state(self.run_dir) if resume else None
        if saved and saved.get("mlflow_run_id"):
            self.training_id = saved["training_id"]
            self.run_id = saved["mlflow_run_id"]
            mlflow.start_run(run_id=self.run_id)
        else:
            run_name = config_dict.get("run", {}).get("name", "run")
            self.training_id = (saved or {}).get("training_id") or new_training_id(
                run_name, machine
            )
            run = mlflow.start_run(
                run_name=self.training_id,
                tags={
                    "training_id": self.training_id,
                    "mlflow.user": user,
                    "machine": machine,
                    "reward": next(iter(config_dict.get("reward", {})), ""),
                    "command": next(iter(config_dict.get("command", {})), ""),
                    "run_dir": str(self.run_dir),
                },
            )
            self.run_id = run.info.run_id
            log_params(flatten(config_dict))
        write_tracking_state(
            self.run_dir,
            {
                "training_id": self.training_id,
                "mlflow_run_id": self.run_id,
                "experiment": experiment,
            },
        )
        mlflow.set_tags(
            {
                "last_session": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_session_user": user,
                "last_session_machine": machine,
            }
        )
        self.log_artifact(self.run_dir / "config.yaml")
        return self.training_id

    def output_format(self) -> MLflowOutputFormat:
        assert self.run_id is not None, "start() first"
        return MLflowOutputFormat(self.run_id)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        import mlflow

        if not path.exists():
            return
        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_path, run_id=self.run_id)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"MLflow: could not upload {path}: {e}", stacklevel=2)

    def end(self, status: str = "FINISHED") -> None:
        """``status``: FINISHED, KILLED (Ctrl+C) or FAILED."""
        import mlflow

        try:
            mlflow.flush_async_logging()
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"MLflow: some metrics may not have been sent: {e}", stacklevel=2)
        mlflow.end_run(status=status)


def default_experiment(uri: str) -> str:
    """On Databricks, a workspace folder the whole team can write to."""
    return DATABRICKS_DEFAULT_EXPERIMENT if is_databricks(uri) else DEFAULT_EXPERIMENT


def is_databricks(uri: str) -> bool:
    return uri == "databricks" or uri.startswith("databricks://")


def current_user(uri: str) -> str:
    """The Databricks user (e.g. ``name@example.com``) for ``databricks[://<profile>]``,
    else the OS user. In the container the OS user is always ``dev``, which says nothing.
    """
    if not is_databricks(uri):
        return getpass.getuser()
    from databricks.sdk import WorkspaceClient

    profile = uri.partition("://")[2] or None
    return WorkspaceClient(profile=profile).current_user.me().user_name


def log_params(params: dict[str, Any]) -> None:
    import mlflow

    items = [(key, str(value)[:_MAX_PARAM_LENGTH]) for key, value in params.items()]
    for i in range(0, len(items), _MAX_PARAMS_PER_CALL):
        mlflow.log_params(dict(items[i : i + _MAX_PARAMS_PER_CALL]))


def read_tracking_state(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / STATE_FILE
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or None


def write_tracking_state(run_dir: Path, state: dict[str, Any]) -> None:
    with open(run_dir / STATE_FILE, "w") as f:
        yaml.safe_dump(state, f, sort_keys=False)
