import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.spotmicro_env import SpotmicroEnv
from physics_backends.factory import create_backend
from reward_functions.walking_reward_function import reward_function, RewardState

from command_devices.random_controller import RandomController
from command_devices.fixed_controller import FixedController
from configuration.config import Config

BASE_DIR = Path(__file__).resolve().parent
cfg = Config(str(BASE_DIR / "configs" / "test_config.yaml"))
#dev = RandomController(cfg)
dev = FixedController(mode="walk")
backend = create_backend("pybullet", use_gui=True)
env = SpotmicroEnv(
    backend=backend,
    device=dev,
    config=cfg,
    reward_fn=reward_function,
    reward_state=RewardState(),
    use_gui=True,
)
obs, _ = env.reset()

cfg.save(str(BASE_DIR / "configs" / "test_config2.yaml"))

for _ in range(3001):
    action = env.action_space.sample()  # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1.0 / 60.0)  # Slow down simulation for visualization
    if terminated or truncated:
        obs, _ = env.reset()
