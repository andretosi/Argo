import time
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path

from spotmicro.env.spotmicro_env import SpotmicroEnv
<<<<<<< HEAD
from reward_functions.walking_reward_function import reward_function, RewardState
#from spotmicro.devices.random_controller import RandomController
from spotmicro.devices.joystick_device import Joystick

=======
from reward_functions.standing_reward_function import reward_function, RewardState
from spotmicro.devices.random_controller import RandomController
from spotmicro.devices.keyboard_device import Keyboard
>>>>>>> 403050b (Keyboard update)
from spotmicro.tools.config import Config

run = "prova2"
DATA_DIR =  Path("data") / f"{run}_results"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # ensure directory exists

#dev = RandomController()
dev = Joystick()    #ora usa la tastiera

conf = Config("configs/test_config.yaml")
<<<<<<< HEAD

=======
#dev = RandomController(conf)
dev = Keyboard()
>>>>>>> 403050b (Keyboard update)
env = SpotmicroEnv(
    dev,
    conf,
    use_gui=True,
    ghost_on=True, 
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