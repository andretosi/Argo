import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

<<<<<<< HEAD
from src.spotmicro.env.spotmicro_env_mujoco import SpotmicroEnv
=======
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spotmicro.env.spotmicro_env import SpotmicroEnv
from spotmicro.devices.random_controller import RandomController
from spotmicro.physics.factory import create_backend
from spotmicro.tools.config import Config
>>>>>>> 67c64a9 (Add robust training workflow and snapshot tools)
from reward_functions.walking_reward_function import reward_function, RewardState

TOTAL_STEPS = 10_000_000
run = "walk27M-1"
base = "shuffle"
BASE_DIR = Path(__file__).resolve().parent
POLICY_DIR = BASE_DIR / "policies"
STATE_DIR = BASE_DIR / "states"
POLICY_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)


def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

checkpoint_callback = CheckpointCallback(
<<<<<<< HEAD
    save_freq=TOTAL_STEPS / 10,                
    save_path=f"policies/{run}_checkpoints",  # Folder to save in
    name_prefix=f"ppo_{run}"            # File name prefix
)

env = SpotmicroEnv(
=======
    save_freq=TOTAL_STEPS // 10,
    save_path=str(POLICY_DIR / f"{run}_checkpoints"),  # Folder to save in
    name_prefix=f"ppo_{run}"            # File name prefix
)

cfg = Config(str(BASE_DIR / "configs" / "test_config.yaml"))
dev = RandomController(cfg)
backend = create_backend("pybullet", use_gui=False)
env = SpotmicroEnv(
    backend=backend,
    device=dev,
    config=cfg,
>>>>>>> 67c64a9 (Add robust training workflow and snapshot tools)
    use_gui=False,
    reward_fn=reward_function, 
    reward_state=RewardState(), 
    src_save_file=str(STATE_DIR / f"{base}.pkl"),
    dest_save_file=str(STATE_DIR / f"{run}.pkl")
    )
check_env(env, warn=True) #optional

model = PPO.load(str(POLICY_DIR / f"ppo_{base}"))
model.set_env(env)
model.tensorboard_log = str(BASE_DIR / "logs")
model.learn(
    total_timesteps=TOTAL_STEPS,
    reset_num_timesteps=False,
    callback=checkpoint_callback
    )
model.save(str(POLICY_DIR / f"ppo_{run}"))
env.close()
