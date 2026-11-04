"""Replay a trained policy.

From ml_training/::

    python training/evaluate.py training/data/robust_walking/final.zip
    python training/evaluate.py <model.zip> --command joystick          # drive it yourself
    python training/evaluate.py <model.zip> --progress 1.0              # hardest curriculum level
    python training/evaluate.py <model.zip> --flat --push 0.8           # flat ground, strong pushes

The model must be in a run folder written by train.py. By default the policy runs under the
conditions saved with it: its run's ``config.yaml`` and the curriculum state in
``<model>.state.yaml``. Flags override them.
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import numpy as np
import torch._dynamo  # noqa: F401
from stable_baselines3 import PPO

from commands import COMMANDS
from training.builders import load_run, make_curricula, make_env, read_state
from training.config import RunConfig
from training.curriculum import CURRICULA, Curriculum

# CLI override -> (curriculum, state key)
OVERRIDES = {
    "terrain_z_max": ("terrain", "z_max"),
    "terrain_seed": ("terrain", "seed"),
    "friction": ("friction", "friction"),
    "push": ("push", "velocity"),
    "motor_noise": ("motor_noise", "std"),
    "sensor_noise": ("sensor_noise", "scale"),
}


def evaluation_curricula(
    config: RunConfig,
    model_path: str | Path,
    progress: float | None = None,
    flat: bool = False,
    overrides: dict[str, float] | None = None,
) -> dict[str, Curriculum]:
    """Curricula set to the checkpoint's saved state, a given progress, or explicit values."""
    curricula = {} if flat else make_curricula(config)
    if progress is not None:
        for curriculum in curricula.values():
            curriculum.jump_to_factor(curriculum.factor_at(progress))
    else:
        saved = read_state(model_path)["curriculum"] or {}
        for name, curriculum in curricula.items():
            curriculum.load_state_dict(saved[name])
    for key, value in (overrides or {}).items():
        name, field = OVERRIDES[key]
        if name not in curricula:
            curricula[name] = CURRICULA[name]()  # default settings, only this value set
        state = curricula[name].state_dict()
        state[field] = value
        curricula[name].load_state_dict(state)
    return curricula


def evaluate(
    model_path: str | Path,
    episodes: int = 3,
    gui: bool = True,
    deterministic: bool = True,
    command: str | None = None,
    progress: float | None = None,
    flat: bool = False,
    overrides: dict[str, float] | None = None,
    max_episode_len: int | None = None,
) -> list[float]:
    """Run ``episodes`` episodes and return their total rewards."""
    config = load_run(model_path)
    if max_episode_len is not None:
        config.env = replace(config.env, max_episode_len=max_episode_len)

    device = None
    if command is not None:
        device = COMMANDS[command].Params().make()
    env = make_env(config, gui=gui, command=device)
    curricula = evaluation_curricula(config, model_path, progress, flat, overrides)
    for curriculum in curricula.values():
        curriculum.attach(env)
    for name, curriculum in curricula.items():
        print(f"  {name}: {curriculum.state_dict()}")

    model = PPO.load(str(model_path), device="cpu")
    step_time = 1.0 / env.control_frequency
    totals = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            total, steps, done = 0.0, 0, False
            while not done:
                start = time.time()
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                for curriculum in curricula.values():
                    curriculum.on_step(env)
                total += reward
                steps += 1
                done = terminated or truncated
                if gui:
                    time.sleep(max(0.0, step_time - (time.time() - start)))
            totals.append(total)
            end = "fell" if terminated else "time limit"
            print(f"Episode {episode + 1}: reward {total:.2f}, {steps} steps ({end})")
    finally:
        env.close()
    if totals:
        print(f"Mean reward: {np.mean(totals):.2f} +/- {np.std(totals):.2f}")
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", help="trained model (.zip)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument(
        "--stochastic", action="store_true", help="sample actions instead of the mean"
    )
    parser.add_argument("--command", choices=sorted(COMMANDS), help="override the command source")
    parser.add_argument("--max-episode-len", type=int)
    group = parser.add_argument_group("conditions (default: as saved with the checkpoint)")
    group.add_argument("--progress", type=float, help="curriculum progress 0..1 to evaluate at")
    group.add_argument("--flat", action="store_true", help="no curricula (flat ground, no noise)")
    group.add_argument("--terrain-z-max", type=float)
    group.add_argument("--terrain-seed", type=int)
    group.add_argument("--friction", type=float)
    group.add_argument("--push", type=float, help="push velocity [m/s]")
    group.add_argument("--motor-noise", type=float)
    group.add_argument("--sensor-noise", type=float)
    args = parser.parse_args()

    overrides = {k: getattr(args, k) for k in OVERRIDES if getattr(args, k) is not None}
    evaluate(
        args.model,
        episodes=args.episodes,
        gui=not args.no_gui,
        deterministic=not args.stochastic,
        command=args.command,
        progress=args.progress,
        flat=args.flat,
        overrides=overrides,
        max_episode_len=args.max_episode_len,
    )


if __name__ == "__main__":
    main()
