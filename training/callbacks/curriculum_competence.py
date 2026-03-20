"""Competence-driven progress tracker for curriculum learning."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable


@configurable
class CompetenceTrackerCallback(BaseCallback):
    """Track walking competence and expose a shared curriculum progress."""

    __config_exclude__ = {"env"}

    def __init__(
        self,
        config: Config,
        env=None,
        total_timesteps: int = 1_000_000,
        ema_alpha: float = 0.10,
        threshold: float = 0.70,
        advance_step: float = 0.05,
        min_episodes_between_advances: int = 10,
        raw_score=None,
        competence_ema=None,
        competence_progress=None,
        last_advance_episode: int = 0,
        verbose: bool = False,
    ):
        super().__init__(verbose)
        self.config = config
        self.env = env
        self.total_timesteps = total_timesteps
        self.ema_alpha = ema_alpha
        self.threshold = threshold
        self.advance_step = advance_step
        self.min_episodes_between_advances = min_episodes_between_advances
        self.raw_score = raw_score
        self.competence_ema = competence_ema
        self.competence_progress = competence_progress
        self.last_advance_episode = int(last_advance_episode)

        self._episode_count = 0

    def _get_unwrapped_env(self):
        env = self.env
        if hasattr(env, "envs"):
            env = env.envs[0]
        if hasattr(env, "env"):
            env = env.env
        return env

    def _sync_config(self, **params) -> None:
        serializable = {
            name: value for name, value in params.items()
            if self.config.is_acceptable(value)
        }
        if not serializable:
            return

        for name, value in serializable.items():
            setattr(self, name, value)
        self.config.update(self, serializable)

    def _record_metrics(self, metrics: dict) -> None:
        if not metrics or not hasattr(self, "model"):
            return

        logger = self.logger

        for name, value in metrics.items():
            if value is None:
                continue
            if np.isscalar(value):
                logger.record(name, float(value))

    def _get_fallback_progress(self) -> float:
        if self.total_timesteps <= 0:
            return 0.0
        current_timesteps = int(getattr(self.model, "num_timesteps", 0))
        return float(np.clip(current_timesteps / self.total_timesteps, 0.0, 1.0))

    def _mean_component(self, reward_history, key, default=0.0) -> float:
        if not reward_history:
            return float(default)
        values = [float(step.get(key, default)) for step in reward_history]
        return float(np.mean(values))

    def _compute_episode_scores(self, env):
        reward_history = getattr(env, "_episode_reward_info", None) or []
        reward_cfg = getattr(getattr(env, "reward_state", None), "config", None)

        tracking_lin_weight = float(getattr(reward_cfg, "tracking_lin_vel", 1.0) or 1.0)
        tracking_ang_weight = float(getattr(reward_cfg, "tracking_ang_vel", 1.0) or 1.0)

        mean_tracking_lin = self._mean_component(reward_history, "tracking_lin_vel", 0.0)
        mean_tracking_ang = self._mean_component(reward_history, "tracking_ang_vel", 0.0)
        tracking_lin_score = np.clip(mean_tracking_lin / tracking_lin_weight, 0.0, 1.0)
        tracking_ang_score = np.clip(mean_tracking_ang / tracking_ang_weight, 0.0, 1.0)
        tracking_score = float(0.5 * (tracking_lin_score + tracking_ang_score))

        max_episode_len = float(getattr(env, "max_episode_len", 1) or 1)
        survival_score = float(np.clip(env._episode_step_counter / max_episode_len, 0.0, 1.0))

        mean_orientation = self._mean_component(reward_history, "orientation", 0.0)
        mean_base_height = self._mean_component(reward_history, "base_height", 0.0)
        orientation_score = float(np.exp(-max(0.0, -mean_orientation)))
        base_height_score = float(np.exp(-max(0.0, -mean_base_height)))
        stability_score = float(np.clip(0.5 * (orientation_score + base_height_score), 0.0, 1.0))

        raw_score = float(np.clip(
            0.5 * tracking_score + 0.3 * survival_score + 0.2 * stability_score,
            0.0,
            1.0,
        ))

        return tracking_score, survival_score, stability_score, raw_score

    def _update_competence(self, raw_score: float):
        raw_score = float(np.clip(raw_score, 0.0, 1.0))
        if self.competence_ema is None:
            competence_ema = raw_score
        else:
            competence_ema = (
                self.ema_alpha * raw_score
                + (1.0 - self.ema_alpha) * float(self.competence_ema)
            )

        if self.competence_progress is None:
            competence_progress = self._get_fallback_progress()
        else:
            competence_progress = float(self.competence_progress)

        if (
            competence_ema >= self.threshold
            and self._episode_count - self.last_advance_episode >= self.min_episodes_between_advances
        ):
            competence_progress = min(1.0, competence_progress + self.advance_step)
            self.last_advance_episode = self._episode_count

        competence_progress = float(np.clip(competence_progress, 0.0, 1.0))
        return float(competence_ema), competence_progress

    def _on_training_start(self) -> None:
        if self.competence_progress is None:
            initial_progress = self._get_fallback_progress()
            self._sync_config(
                competence_progress=float(initial_progress),
                last_advance_episode=int(self.last_advance_episode),
            )
        else:
            self._sync_config(
                raw_score=self.raw_score,
                competence_ema=self.competence_ema,
                competence_progress=self.competence_progress,
                last_advance_episode=int(self.last_advance_episode),
            )

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [False])
        if not any(dones):
            return True

        self._episode_count += 1
        unwrapped = self._get_unwrapped_env()
        if unwrapped is None:
            return True

        tracking_score, survival_score, stability_score, raw_score = self._compute_episode_scores(unwrapped)
        competence_ema, competence_progress = self._update_competence(raw_score)

        self._sync_config(
            raw_score=float(raw_score),
            competence_ema=float(competence_ema),
            competence_progress=float(competence_progress),
            last_advance_episode=int(self.last_advance_episode),
        )
        self._record_metrics({
            "competence/raw_score": raw_score,
            "competence/ema": competence_ema,
            "competence/progress": competence_progress,
            "competence/tracking_score": tracking_score,
            "competence/survival_score": survival_score,
            "competence/stability_score": stability_score,
        })

        return True
