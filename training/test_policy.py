import time
from stable_baselines3 import PPO
from pathlib import Path

from spotmicro.env.spotmicro_env_mujoco import SpotmicroEnv
from reward_functions.walking_reward_function import reward_function, RewardState
<<<<<<< HEAD
<<<<<<< HEAD
#from spotmicro.devices.random_controller import RandomController
from spotmicro.devices.joystick_device import Joystick

from spotmicro.tools.config import Config
=======
from src.spotmicro import RandomController
from src.spotmicro.tools.config import Config
>>>>>>> ab1736d (add mujoco and env / agent interface)
=======
from spotmicro.devices.random_controller import RandomController
from spotmicro.tools.config import Config
>>>>>>> 2e39146 (moved from src.spotmicro to spotmicro. Fucking install the package before changing everything, will ya?)

run = "prova2"
DATA_DIR =  Path("data") / f"{run}_results"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # ensure directory exists

#dev = RandomController()
dev = Joystick()    #ora usa la tastiera

conf = Config("configs/test_config.yaml")

env = SpotmicroEnv(
    dev,
    conf,
    use_gui=True, 
    reward_fn=reward_function,
    reward_state=RewardState(),
    #src_save_file=str(DATA_DIR / f"{run}.pkl")
    ghost_on=True
    )
obs, _ = env.reset()

# Load your trained model
model = PPO.load( str(DATA_DIR / f"ppo_{run}_final"))  # or path to your .zip
#model = PPO.load(str(DATA_DIR / "checkpoints" / f"ppo_{run}_1000000_steps"))
print(f"num steps: {env.num_steps}")

# Run rollout
for _ in range(3001):
    #aggiunte per verifica tastiera
    cmd = dev.read() # Legge lo stato attuale della tastiera
    
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Terminated")
        env.plot_reward_components()  # plot per episode
        obs, _ = env.reset()
    
    
    time.sleep(1/60.)  # Match simulation step time for real-time playback

env.close()