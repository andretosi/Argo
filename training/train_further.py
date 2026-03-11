from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from src.spotmicro.env.spotmicro_env_mujoco import SpotmicroEnv
from reward_functions.walking_reward_function import reward_function, RewardState

TOTAL_STEPS = 10_000_000
run = "walk27M-1"
base = "shuffle"


def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

checkpoint_callback = CheckpointCallback(
    save_freq=TOTAL_STEPS / 10,                
    save_path=f"policies/{run}_checkpoints",  # Folder to save in
    name_prefix=f"ppo_{run}"            # File name prefix
)

env = SpotmicroEnv(
    use_gui=False,
    reward_fn=reward_function, 
    reward_state=RewardState(), 
    src_save_file=f"states/{base}.pkl",
    dest_save_file=f"states/{run}.pkl"
    )
check_env(env, warn=True) #optional

model = PPO.load(f"policies/ppo_{base}")
model.set_env(env)
model.tensorboard_log = "./logs"
model.learn(
    total_timesteps=TOTAL_STEPS,
    reset_num_timesteps=False,
    callback=checkpoint_callback
    )
model.save(f"policies/ppo_{run}")
env.close()
