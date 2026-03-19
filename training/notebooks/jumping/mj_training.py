import os
from stable_baselines3 import PPO
from spotmicro.env.spotmicro_env import SpotmicroEnv
from spotmicro.physics.factory import create_backend
from spotmicro.devices.fixed_controller import FixedController
from spotmicro.tools.config import Config
from reward_function import reward_function, RewardState
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

# ========= CONFIG ==========
TOTAL_STEPS = 20_000_000 # Scratch training usually needs more time
run = "jump_mujoco_scratch2"
log_dir = f"./logs/{run}"

# Standard Linear Schedule for Scratch Training
def linear_schedule(initial_value=3e-4):
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule

checkpoint_callback = CheckpointCallback(
    save_freq=500_000,
    save_path=f"{run}_checkpoints",
    name_prefix=f"ppo_{run}"
)

# ========= ENV (MUJOCO) ==========
cfg = Config()
dev = FixedController("still") 
backend = create_backend("mujoco", use_gui=False) 

env = SpotmicroEnv(
    backend,
    dev,
    cfg,
    reward_function,
    RewardState(),
    use_gui=False
)

# ========= MODEL (FRESH) ==========
model = PPO(
    "MlpPolicy", 
    env,
    verbose=1,
    learning_rate=linear_schedule(3e-4), # Higher LR to explore the landscape
    n_steps=2048,           # Batch size for stable updates
    batch_size=64,
    n_epochs=10,
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,
    clip_range=0.2,         # More room to move than the 0.1 fine-tune
    ent_coef=0.01,          # Increased entropy to encourage "trying" jumps
    tensorboard_log=log_dir
)

# Custom logger
new_logger = configure(log_dir, ["csv", "tensorboard"])
model.set_logger(new_logger)

# ========= TRAIN ==========
print(f"--- Starting MuJoCo Scratch Training for {TOTAL_STEPS} steps ---")
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=checkpoint_callback
)

model.save(f"ppo_{run}")
env.close()