import time
import sys
from stable_baselines3 import PPO
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.spotmicro_env import SpotmicroEnv
from command_devices.random_controller import RandomController
from physics_backends.factory import create_backend
from configuration.config import Config
from reward_functions.walking_reward_function import reward_function, RewardState

run = "prova2"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / f"{run}_results"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # ensure directory exists

conf = Config(str(BASE_DIR / "configs" / "test_config.yaml"))
dev = RandomController(conf)
backend = create_backend("pybullet", use_gui=True)
env = SpotmicroEnv(
    backend=backend,
    device=dev,
    config=conf,
    use_gui=True, 
    reward_fn=reward_function,
    reward_state=RewardState(),
    #src_save_file=str(DATA_DIR / f"{run}.pkl")
    )
obs, _ = env.reset()

# Load your trained model
model = PPO.load( str(DATA_DIR / f"ppo_{run}_final"))  # or path to your .zip
#model = PPO.load(str(DATA_DIR / "checkpoints" / f"ppo_{run}_1000000_steps"))
print(f"num steps: {env.num_steps}")

# Run rollout
for _ in range(3001):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Terminated")
        env.plot_reward_components()  # plot per episode
        obs, _ = env.reset()
    
    
    time.sleep(1/60.)  # Match simulation step time for real-time playback

env.close()
