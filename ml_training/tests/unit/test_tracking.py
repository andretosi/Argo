"""MLflow tracking helpers: what reaches MLflow, and that tracking failures never stop training."""

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training import tracking


class NotANumber:
    def __float__(self) -> float:
        raise ValueError("not a number")


class OutputFormatTestCase(unittest.TestCase):
    def test_only_numbers_not_excluded_are_logged(self) -> None:
        output = tracking.MLflowOutputFormat("run")
        values = {
            "train/loss": 0.5,
            "time/fps": 100,
            "flag": True,
            "name": "text",
            "bad": NotANumber(),
            "hidden": 1.0,
            "rollout/ep[0]": 2.0,
        }
        with mock.patch("mlflow.log_metrics") as log:
            output.write(values, {"hidden": ("mlflow",), "train/loss": ("csv",)}, step=7)
            output.write({"name": "only text"}, {}, step=8)  # nothing to send
        log.assert_called_once_with(
            {"train/loss": 0.5, "time/fps": 100.0, "rollout/ep_0_": 2.0},
            step=7,
            run_id="run",
            synchronous=False,
        )
        output.close()

    def test_logging_failures_warn_once(self) -> None:
        output = tracking.MLflowOutputFormat("run")
        with (
            mock.patch("mlflow.log_metrics", side_effect=ConnectionError("offline")),
            self.assertWarnsRegex(UserWarning, "training continues") as caught,
        ):
            output.write({"a": 1.0}, {}, step=1)
            output.write({"a": 2.0}, {}, step=2)
        self.assertEqual(len(caught.warnings), 1)


class TrackerTestCase(unittest.TestCase):
    def test_upload_and_flush_failures_only_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = tracking.Tracker(Path(tmp))
            artifact = Path(tmp) / "model.zip"
            tracker.log_artifact(artifact)  # missing: skipped
            artifact.write_bytes(b"x")
            with (
                mock.patch("mlflow.log_artifact", side_effect=OSError("disk")),
                self.assertWarnsRegex(UserWarning, "could not upload"),
            ):
                tracker.log_artifact(artifact)
            with (
                mock.patch("mlflow.flush_async_logging", side_effect=OSError("net")),
                mock.patch("mlflow.end_run") as end_run,
                self.assertWarnsRegex(UserWarning, "may not have been sent"),
            ):
                tracker.end("KILLED")
            end_run.assert_called_once_with(status="KILLED")

    def test_default_experiment_and_user(self) -> None:
        self.assertEqual(tracking.default_experiment("databricks"), "/Shared/argo")
        self.assertEqual(tracking.default_experiment("sqlite:///x.db"), "argo")
        client = mock.Mock()
        client.return_value.current_user.me.return_value.user_name = "me@example.com"
        with mock.patch("databricks.sdk.WorkspaceClient", client):
            self.assertEqual(tracking.current_user("databricks://team"), "me@example.com")
        client.assert_called_once_with(profile="team")

    def test_default_experiment_is_used_when_none_is_set(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict("os.environ", {"MLFLOW_TRACKING_URI": f"sqlite:///{tmp}/m.db"}),
            contextlib.chdir(tmp),  # MLflow puts artifacts in ./mlruns next to a SQLite store
        ):
            import os

            os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
            (Path(tmp) / "config.yaml").write_text("{}")
            tracker = tracking.Tracker(Path(tmp))
            tracker.start({"run": {"name": "x"}}, resume=False)
            tracker.end()
            self.assertEqual(tracking.read_tracking_state(Path(tmp))["experiment"], "argo")
        self.assertIsNone(tracking.read_tracking_state(Path(tmp) / "missing"))


if __name__ == "__main__":
    unittest.main()
