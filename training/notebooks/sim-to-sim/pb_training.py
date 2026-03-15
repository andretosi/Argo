import time
from stable_baselines3 import PPO

from spotmicro.env.spotmicro_env import SpotmicroEnv
from spotmicro.physics.factory import create_backend
from spotmicro.devices.fixed_controller import FixedController
from spotmicro.tools.config import Config
from reward_function import reward_function, RewardState
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

# ========= CONFIG ==========
TOTAL_STEPS = 2_000_000
run = "standPB"
log_dir = f"./logs/{run}"

def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

checkpoint_callback = CheckpointCallback(
    save_freq=TOTAL_STEPS // 5,
    save_path=f"{run}_checkpoints",
    name_prefix=f"ppo_{run}"
)

# ========= ENV ==========
cfg = Config()
dev = FixedController("still") #not a configurable class
backend = create_backend("pybullet", use_gui=False)
env = SpotmicroEnv(
    backend,
    dev,
    cfg,
    reward_function,
    RewardState(),
    use_gui=False
)
check_env(env, warn=True)


# ========= MODEL ==========
model = PPO(
    "MlpPolicy", 
    env,
    verbose=1,   # no default printouts
    learning_rate=clipped_linear_schedule(3e-4),
    ent_coef=0.001,
    clip_range=0.1,
    tensorboard_log=log_dir,
    device = 'cpu'
)

# Custom logger: ONLY csv + tensorboard (no stdout table)
new_logger = configure(log_dir, ["csv", "tensorboard"])
model.set_logger(new_logger)