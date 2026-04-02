import time
from stable_baselines3 import PPO

from spotmicro.env.spotmicro_env import SpotmicroEnv
from spotmicro.physics.factory import create_backend
from spotmicro.devices.random_controller import RandomController
from spotmicro.tools.config import Config
from reward_function import reward_function, RewardState
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

# ========= CONFIG ==========
TOTAL_STEPS = 20_000_000
run = "walkMJ"
log_dir = f"./logs/{run}"

def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

checkpoint_callback = CheckpointCallback(
    save_freq=TOTAL_STEPS // 40,
    save_path=f"{run}_checkpoints",
    name_prefix=f"ppo_{run}"
)

# ========= ENV ==========
cfg = Config()
dev = RandomController(cfg, p_base2still=1.0, p_base2turn=0.0, p_base2walk=0.0) 
backend = create_backend("mujoco", use_gui=False)
env = SpotmicroEnv(
    backend,
    dev,
    cfg,
    reward_function,
    RewardState(),
    use_gui=False,
    survival_reward=15,
)
check_env(env, warn=True)
cfg.save("configs/slow_start.yaml")


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

# ========= TRAIN ==========
model.learn(
    total_timesteps=TOTAL_STEPS,
    reset_num_timesteps=False,
    callback=checkpoint_callback
)
model.save(f"ppo_{run}")
env.close()