from stable_baselines3 import PPO

import sys
import os

# Start from the current working directory (where notebook is)
cwd = os.getcwd()

# Go two levels up (to the "grandparent")
grandparent_dir = os.path.abspath(os.path.join(cwd, "..", ".."))

# Add to sys.path if not already there
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from reward_function import reward_function, RewardState
from spotmicro.env.spotmicro_env_mujoco import SpotmicroEnv
from spotmicro.devices.fixed_controller import FixedController
from spotmicro.tools.config import Config

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

# ========= CONFIG ==========
TOTAL_STEPS = 2_000_000
run = "NewStand"
log_dir = f"./logs/{run}"

def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

checkpoint_callback = CheckpointCallback(
    save_freq=TOTAL_STEPS // 10,
    save_path=f"{run}_checkpoints",
    name_prefix=f"ppo_{run}"
)

# ========= ENV ==========
cfg = Config()
dev = FixedController("still")
env = SpotmicroEnv(
    dev,
    cfg,
    reward_function,
    RewardState(), 
    use_gui=False,
)
check_env(env, warn=True)

# ========= MODEL ==========
model = PPO(
    "MlpPolicy", 
    env,
    verbose=0,   # no default printouts
    learning_rate=clipped_linear_schedule(3e-4),
    ent_coef=0.001,
    clip_range=0.1,
    tensorboard_log=log_dir,
    n_steps=2048,
    batch_size=64
)

# Custom logger: ONLY csv + tensorboard (no stdout table)
new_logger = configure(log_dir, ["csv", "tensorboard"])
model.set_logger(new_logger)
model.learn(
    total_timesteps=TOTAL_STEPS,
    reset_num_timesteps=False,
    callback=checkpoint_callback
)
model.save(f"ppo_{run}")
env.close()
