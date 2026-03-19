from __future__ import annotations

from robust_common import (
    build_run_paths,
    create_robust_env,
    default_reward_config,
    load_model,
    print_snapshot_summary,
    resolve_model_stem,
    restore_snapshot_callbacks,
    run_policy_rollout,
)
from spotmicro.tools.config import Config


# ============================================================================
# Snapshot Test Configuration
# Edit these values directly in the file, then run the script from your editor.
# ============================================================================

RUN_NAME = "robust_walk_v1"
SNAPSHOT_MODEL_STEM = None

BACKEND = "pybullet"
USE_GUI = True
DEVICE = "cpu"

MAX_EPISODE_LEN = 3000
SIM_FREQUENCY = 240
CONTROL_FREQUENCY = 60

N_EVAL_EPISODES = 3
DETERMINISTIC = True
REALTIME_SLEEP = 1.0 / 240.0


def main() -> None:
    paths = build_run_paths(RUN_NAME)
    model_stem = resolve_model_stem(paths, SNAPSHOT_MODEL_STEM)
    snapshot_path = model_stem.with_suffix(".yaml")

    cfg = Config(str(snapshot_path))
    reward_config = default_reward_config()

    env = create_robust_env(
        cfg,
        backend_name=BACKEND,
        use_gui=USE_GUI,
        max_episode_len=MAX_EPISODE_LEN,
        sim_frequency=SIM_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        reward_config=reward_config,
    )

    try:
        snapshot_callbacks = restore_snapshot_callbacks(cfg, env, verbose=False)
        model = load_model(
            model_stem,
            env,
            log_dir=paths.log_dir,
            device=DEVICE,
        )

        print(f"Testing snapshot: {model_stem}.zip")
        print_snapshot_summary(snapshot_callbacks)

        run_policy_rollout(
            model=model,
            env=env,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=DETERMINISTIC,
            force_callback=snapshot_callbacks["force"],
            realtime_sleep=REALTIME_SLEEP if USE_GUI else 0.0,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
