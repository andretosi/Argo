import time
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
TOTAL_STEPS = 10_000_000
source_policy = "jumpPB"  # Your existing PyBullet model name
run = "jump_mujoco_finetune"
log_dir = f"./logs/{run}"

# Low Learning Rate for Fine-Tuning
# Starts at 1e-5 and decays, preventing the model from "forgetting" PB skills
def fine_tune_schedule(initial_value=1e-5, min_value=1e-6):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

checkpoint_callback = CheckpointCallback(
    save_freq=TOTAL_STEPS // 10,
    save_path=f"{run}_checkpoints",
    name_prefix=f"ppo_{run}"
)

# ========= ENV (MUJOCO) ==========
cfg = Config()
dev = FixedController("still") 

# CRITICAL: Swapping to MuJoCo backend
backend = create_backend("mujoco", use_gui=False) 

env = SpotmicroEnv(
    backend,
    dev,
    cfg,
    reward_function,
    RewardState(),
    use_gui=False
)

# ========= LOAD PRE-TRAINED MODEL ==========
model_path = f"ppo_{source_policy}.zip"

if os.path.exists(model_path):
    print(f"--- Loading existing PyBullet policy: {model_path} ---")
    # Load the weights but point them to the new MuJoCo environment
    model = PPO.load(
        model_path, 
        env=env, 
        device='cpu',
        custom_objects={
            "learning_rate": fine_tune_schedule(1e-5),
            "clip_range": 0.1 # Keep clipping tight to prevent wild updates
        }
    )
else:
    print(f"--- Warning: {model_path} not found. Starting from scratch! ---")
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        learning_rate=fine_tune_schedule(3e-4), # Faster if starting fresh
        tensorboard_log=log_dir,
        device='cpu'
    )

# Custom logger
new_logger = configure(log_dir, ["csv", "tensorboard"])
model.set_logger(new_logger)

# ========= TRAIN ==========
print(f"--- Starting MuJoCo Transfer Learning for {TOTAL_STEPS} steps ---")
model.learn(
    total_timesteps=TOTAL_STEPS,
    reset_num_timesteps=False, # Keeps the step count continuing from PB
    callback=checkpoint_callback
)

model.save(f"ppo_{run}")
env.close()