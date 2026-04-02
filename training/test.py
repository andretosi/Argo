import time
from stable_baselines3 import PPO

<<<<<<< HEAD
from spotmicro.env.spotmicro_env_mujoco import SpotmicroEnv
from reward_functions.walking_reward_function import reward_function, RewardState
<<<<<<< HEAD
<<<<<<< HEAD
from spotmicro.devices.random_controller import RandomController
from spotmicro.devices.fixed_controller import FixedController
from spotmicro.devices.joystick_device import Joystick
from spotmicro.tools.config import Config
=======
from src.spotmicro import RandomController
from src.spotmicro.tools.config import Config
>>>>>>> ab1736d (add mujoco and env / agent interface)

cfg = Config("configs/test_config.yaml")
#dev = RandomController(cfg)
#dev = FixedController(mode="walk")
dev = Joystick()
env = SpotmicroEnv(dev, cfg, reward_function, RewardState(), use_gui=True, ghost_on=True)
=======

from spotmicro.devices.random_controller import RandomController
=======
from spotmicro.env.spotmicro_env import SpotmicroEnv
from spotmicro.physics.factory import create_backend
>>>>>>> 66a7325 (modified for training)
from spotmicro.devices.fixed_controller import FixedController
from spotmicro.tools.config import Config
from reward_functions.standing_reward_function import reward_function, RewardState
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

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
<<<<<<< HEAD
>>>>>>> 2e39146 (moved from src.spotmicro to spotmicro. Fucking install the package before changing everything, will ya?)
obs, _ = env.reset()

cfg.save("configs/test_config2.yaml")
=======
>>>>>>> 66a7325 (modified for training)

for _ in range(3001):
    action = env.action_space.sample()  # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1.0 / 60.0)  # Slow down simulation for visualization
    if terminated or truncated:
        obs, _ = env.reset()