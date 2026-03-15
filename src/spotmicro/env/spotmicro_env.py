import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import inspect, os, pickle, warnings
from importlib.resources import files

<<<<<<< HEAD
<<<<<<< HEAD
from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable
from spotmicro.agent.agent import Agent
from spotmicro.env.terrain import Terrain
from spotmicro.devices.device import Device
from spotmicro.tools.kinematic_ghost import KinematicGhost
from spotmicro.tools.kg_renderer import KG_Renderer, PyBulletRenderer
=======
from src.spotmicro.physics.backend import PhysicsBackend
from src.spotmicro.tools.config import Config
from src.spotmicro.tools.configurable import configurable
from src.spotmicro.agent.agent import Agent
from src.spotmicro.devices.device import Device
from src.spotmicro.tools.tracker import Tracker
>>>>>>> ab1736d (add mujoco and env / agent interface)
=======
from spotmicro.physics.backend import PhysicsBackend
from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable
from spotmicro.agent.agent import Agent
from spotmicro.devices.device import Device
from spotmicro.tools.kinematic_ghost import KinematicGhost
from spotmicro.tools.kg_renderer import KG_Renderer, PyBulletRenderer
>>>>>>> f10079f (wrong dir, i'm tired. Removed src. from ALL the GODDAM imports. You need to install the package by tunning pip install -e . in the root directory of the project in order to have everything up and running. Also, added mujoco to ddependencies)

@configurable
class SpotmicroEnv(gym.Env):
    """
    Physics-agnostic SpotMicroEnv.  All simulator interaction goes through a PhysicsBackend.

    SB3 <-> Env <-> PhysicsBackend (pybullet | mujoco)

    Methods
    --------------------------
<<<<<<< HEAD
    Here is a list of the **methods of the gym.Env class that are overriden:**
    - step: Method exposed and used by SB3 to execute one time step within the environment.

    - reset: resets the environment to an inital state, and returns the firts observation value.

    - close: Closes the pybullet simulation, saves the state if a destination path is provided.

    - render (ignored)

    **Public methods:** 
    - save_state: saves a dictionary containing useful data about the current state:
                (total steps counter, previous action, joint state history)
                The file is stored in _dest_save_file, parameter of the constructor method: src_save_file
    - load_state: loads the dictionary saved by save_state(). The path of the file is a 
                parameter of the constructor method: dest_save_file

    **Private methods**
    - _step_simulation: accepts an action and returns an observation. This is done by the Agent class:
                        apply_action(action) -> pybullet.stepSimulation -> sync_state()
                        Eventually tilts the plane if a variable is set.
                        Returns the observation using _get_observation().

    These methods are used to compute the values that are returned by step() (and other relevant methods):
    - _get_observation (np.ndarray): Agent's observation of the environment.
            
    - _calculate_reward (float): Placeholder method that calls the reward function provided as an input.

    - _is_target_state (bool): Private method that returns wether the state of the agent is a target state 
        (one in which to end the simulation) or not. It also returns a penalty to 
        apply when the specific target condition is met.

    - _is_terminated (bool): Function that returns wether an episode was terminated artificially or timed out

    - _get_info (dict): Function that returns a dict containing the following fields:
            - height (of the body)
            - pitch: (of the base)
            - episode_step
"""
    def __init__(self, device: Device, config: Config, reward_fn: callable, reward_state, use_gui=False, ghost_on=False, dest_save_file=None, src_save_file=None, writer=None,
=======
    gym.Env overrides: step, reset, close, render (no-op)
    Public: save_state, load_state
    Private: _step_simulation, _get_observation, _calculate_reward,
             _is_target_state, _is_truncated, _get_info
    """
    def __init__(self, backend: PhysicsBackend,
                 device: Device, config: Config, reward_fn: callable, reward_state,
                 model_path: str | None = None,
                 use_gui=False, tracker_on=False, dest_save_file=None, src_save_file=None, writer=None,
>>>>>>> ab1736d (add mujoco and env / agent interface)
                 max_episode_len=3000, sim_frequency=240, control_frequency=60, joint_history_max_len=5,
                 min_height=0.15, max_height=0.4, max_pitchroll=0.96, tipping_penalty=-2, jump_fall_penalty=-100, survival_reward=3.0, 
                 spawn_height=0.230, target_body_to_feet_height=0.2
                ):
        super().__init__()
        self.config = config
        self._backend = backend

        #<----- INITIALIZATIONS ----->
        self.use_gui = use_gui
        self.ghost_on = ghost_on
        self.np_random = None
        self.reward_state = reward_state
        self._episode_reward_info = None
        self.writer = writer

        self._OBS_SPACE_SIZE = 97
        self._ACT_SPACE_SIZE = 12
        
        self.max_episode_len = max_episode_len
        self.sim_frequency = sim_frequency
        self.control_frequnecy = control_frequency
        self.joint_history_max_len = joint_history_max_len

        self._episode_step_counter = 0
        self._total_steps_counter = 0

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._OBS_SPACE_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._ACT_SPACE_SIZE,), dtype=np.float32
        )

        self.min_height = min_height
        self.max_height = max_height
        self.max_pitchroll = max_pitchroll
        self.tipping_penalty = tipping_penalty
        self.jump_fall_penalty = jump_fall_penalty
        self.survival_reward = survival_reward
        self.spawn_height = spawn_height
        self.target_body_to_feet_height = target_body_to_feet_height

        # Tempo fisico della simulazione
        self.dt = 1.0 / self.sim_frequency

        #<----- END OF PARAMETER INITIALIZATIONS ----->

        if not callable(reward_fn):
            raise ValueError(f"reward_fn must be callable (function), but a {type(reward_fn)} was given")
        self._reward_fn = reward_fn
        self._model_path = self._resolve_model_path(model_path)

<<<<<<< HEAD
        # <-- PHYSICS_ENV: init? -->
        #Initialize pybullet: connects to the simulation server and sets the initial view
        if self.physics_client is None:
            self.physics_client = pybullet.connect(pybullet.GUI if self.use_gui else pybullet.DIRECT)
            pybullet.resetDebugVisualizerCamera(
                cameraDistance=1.2,   # zoom out a bit
                cameraYaw=45,         # rotate around robot
                cameraPitch=-30,      # look slightly down
                cameraTargetPosition=[0, 0, 0.2],  # center around robot base
                physicsClientId=self.physics_client
            )

        pybullet.resetSimulation(physicsClientId=self.physics_client)
        pybullet.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        pybullet.setTimeStep(1/self.sim_frequency, physicsClientId=self.physics_client)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        # <-- -->
        
        #Initialize the terrain object
        self._terrain = Terrain(self.physics_client, self.config)

        #TODO: put terrain initialization logic here? subsequent lines are just kept for legacy, might as well be deleted when the integration begins
        self._terrain_evo_coefficients = np.array([0.0, 0.0, 0.0]) 
        self._terrain.generate(self._terrain_evo_coefficients)

        # <-- PHYSICS_ENV: attach terrain? -->
        pybullet.changeDynamics(
            bodyUniqueId=self._terrain.terrain_id,
            linkIndex=-1,
            lateralFriction=1.0,           
            spinningFriction=0.0,
            rollingFriction=0.0,
            restitution=0.0,
            physicsClientId=self.physics_client
=======
        # Create agent — it will call backend.load_model() internally
        self._agent = Agent(
            backend=self._backend,
            model_path=self._model_path,
            spawn_height=self.spawn_height,
            device=device,
            config=config,
            action_space_size=self._ACT_SPACE_SIZE,
>>>>>>> ab1736d (add mujoco and env / agent interface)
        )

        if ghost_on:
            renderer = PyBulletRenderer(client_id=self.physics_client)  # Renderer specifico (PyBullet o MuJoCo)
            self._ghost = KinematicGhost(renderer, self.dt)

        self._dest_save = dest_save_file
        if self._dest_save is not None:
            if not isinstance(self._dest_save, str):
                raise TypeError("Destination file path must be a string.")
            if os.path.exists(dest_save_file):
                warnings.warn(f"File '{self._dest_save}' already exists and will be overwritten.", UserWarning)
            if not self._dest_save.endswith(".pkl"):
                raise ValueError("Expected a .pkl file for environment state save destination")
            
        self._src_file = src_save_file
        if self._src_file is not None:
            if not isinstance(self._src_file, str):
                raise TypeError("Source file path must be a string.")
            if not os.path.exists(self._src_file):
                raise FileNotFoundError(f"No file found at {self._src_file}")
            if not src_save_file.endswith(".pkl"):
                raise ValueError("Expected a .pkl file for environment state save source")
            self.load_state()

    def _resolve_model_path(self, model_path: str | None) -> str:
        if model_path is not None:
            return model_path

        if self._backend.engine_name == "mujoco":
            return str(files("spotmicro.data").joinpath("spotmicroai.mujoco.xml"))

        if self._backend.engine_name == "pybullet":
            return str(files("spotmicro.data").joinpath("spotmicroai.urdf"))

        raise ValueError(
            f"Could not infer a model path for backend {self._backend.engine_name!r}. "
            "Pass model_path explicitly."
        )
        
    def save_state(self):
        state = {
            "total_steps_counter": self._total_steps_counter,
            "previous_action": self._agent.previous_action,
            "joint_history": list(self._agent.joint_history),
            "target_linear_velocity": self._agent.controller.input.as_array[:2],
            "target_angular_velocity": self._agent.controller.input.as_array[2]
        }
        with open(self._dest_save, "wb") as f:
            pickle.dump(state, f)

    def load_state(self):
        with open(self._src_file, 'rb') as f:
            state = pickle.load(f)
        self._total_steps_counter = state["total_steps_counter"]
        self._agent.previous_action = state["previous_action"]
        self._agent.joint_history = deque(state["joint_history"], maxlen=self._agent.joint_history_maxlen)

    def close(self):
        self._backend.close()
        if self._dest_save is not None:
            self.save_state() 

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._episode_step_counter = 0
        self._episode_reward_info = []

        self._agent.reset(self.spawn_height)

        if self.reward_state is not None:
            self.reward_state.populate(self)

        # Let physics settle with homing applied
        for _ in range(5):
            self._agent.apply_action(self._agent.default_actions)
            self._backend.step()
            self._agent.sync_state()

        # Sanity check reward function signature
        sig = inspect.signature(self._reward_fn)
        if len(sig.parameters) != 2:
            raise ValueError("reward_fn must accept exactly 2 parameters (env, action)")

        try:
            dummy_action = np.array(self._agent.homing_positions, dtype=np.float32)
            reward, info = self._reward_fn(self, dummy_action)
            if not isinstance(reward, (int, float)):
                raise ValueError("reward_fn must return a number as first return value")
            if not isinstance(info, dict):
                raise ValueError("reward_fn must return a dict as second return value")
        except Exception as e:
            raise ValueError(f"Error testing reward_fn: {str(e)}")
        
<<<<<<< HEAD
        if self.ghost_on:
            self._ghost.reset(start_pos=self._agent.state.base_position, start_quat=self._agent.state.base_orientation)
=======
        if self.tracker_on:
            self._tracker.reset()
>>>>>>> ab1736d (add mujoco and env / agent interface)

        return self._get_observation(), self._get_info()
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._agent.controller.update()
        observation = self._step_simulation(action)
        self._episode_step_counter += 1
        reward, reward_info = self._calculate_reward(action)

        terminated, term_penalty = self._is_target_state()
        truncated = self._is_truncated()
        info = self._get_info()

        self._episode_reward_info.append(reward_info)
        if truncated:
            reward += self.survival_reward
        if terminated:
            reward += term_penalty

        self._total_steps_counter += 1
        return observation, reward, terminated, truncated, info
    
    def plot_reward_components(self):
        keys = self._episode_reward_info[0].keys()
        values = {k: [] for k in keys}
        for step_info in self._episode_reward_info:
            for k in keys:
                values[k].append(step_info[k])
        plt.figure(figsize=(12, 6))
        for k in keys:
            plt.plot(values[k], label=k)
        plt.title("Reward Components Over Episode")
        plt.xlabel("Timestep")
        plt.ylabel("Reward Contribution")
        plt.legend()
        plt.grid(True)
        plt.savefig("plot.png")
        plt.close()
    
    def log_rewards(self, reward_dict: dict):
        if self.writer is None:
            return
        for key, value in reward_dict.items():
            try:
                self.writer.add_scalar(f"reward_components/{key}", value, self.num_steps)
            except Exception as e:
                print(f"[Logging Error] Could not log {key}: {e}")

    def _step_simulation(self, action: np.ndarray) -> np.ndarray:
        self._agent.apply_action(action)
<<<<<<< HEAD

        #TODO: deprecated      
        #if self._terrain.config.mode == "tilting":
        #    self._terrain.tilt_plane()
        
        for _ in range(self.sim_frequency // self.control_frequnecy):
            pybullet.stepSimulation()

            if self.ghost_on:
                self._ghost.apply_command(self._agent.controller.input)

            if self.use_gui:
                time.sleep(1/70.) # MAGIC NUMBER, MAKES THE SIMULATION LOOK REAL-TIME (not slow, not too fast)
=======
        
        for _ in range(self.sim_frequency // self.control_frequnecy):
            self._backend.step()
>>>>>>> ab1736d (add mujoco and env / agent interface)

        self._backend.sync_viewer()
        self._agent.sync_state()
        return self._get_observation()

    def _get_gravity_vector(self) -> np.ndarray:
        gravity_world = np.array([0, 0, -1])
        rot_matrix = self._backend.get_rotation_matrix(self._agent.state.base_orientation)
        gravity_base = rot_matrix.T @ gravity_world
        return gravity_base   
    
    def _joint_positions_norm(self, pos):
        pos_norm = []
        for i, joint in enumerate(self._agent.motor_joints):
            pos_norm.append(((2 * (pos[i] - joint.limits[0])) / (joint.limits[1] - joint.limits[0])) - 1)
        return pos_norm
    
    def _joint_velocities_norm(self, vels):
        vel_norm = [np.tanh(vel / self._agent.max_joint_velocity) for vel in vels]
        return vel_norm

    def _get_observation(self) -> np.ndarray:
        """
        - 0-2: gravity vector
        - 3: height of the robot
        - 4-6: linear velocity of the base
        - 7-9: angular velocity of the base
        - 10-21: positions of the joints
        - 22-33: velocities of the joints
        - 34-81: history
        - 82-93: previous action
        - 94-95: linear velocity reference in robot frame
        - 96: angular velocity reference in robot frame
        """
        obs = []
        obs.extend(self._get_gravity_vector())
        obs.append((self._agent.state.base_position[2] - self.target_body_to_feet_height) / self._agent.max_norm_height)
        obs.extend(self._agent.state.linear_velocity / self._agent.max_linear_velocity)
        obs.extend(self._agent.state.angular_velocity / self._agent.max_angular_velocity)
        obs.extend(self._joint_positions_norm(self._agent.state.joint_positions)) 
        obs.extend(self._joint_velocities_norm(self._agent.state.joint_velocities))
        obs.extend(self._joint_positions_norm(self._agent.joint_history[1][0]))
        obs.extend(self._joint_velocities_norm(self._agent.joint_history[1][1]))
        obs.extend(self._joint_positions_norm(self._agent.joint_history[4][0]))
        obs.extend(self._joint_velocities_norm(self._agent.joint_history[4][1]))
        obs.extend(self._agent.previous_action)
        obs.extend(self._agent.controller.input.as_array)

        assert len(obs) == self._OBS_SPACE_SIZE, f"Expected {self._OBS_SPACE_SIZE} elements, got {len(obs)}"
        return np.array(obs, dtype=np.float32)

    def _is_target_state(self) -> tuple[bool, int]:
        base_pos = self._agent.state.base_position
        roll, pitch, _ = self._agent.state.roll_pitch_yaw
        height = base_pos[2]

        if height <= self.min_height or height > self.max_height:
            return (True, self.jump_fall_penalty) 
        elif abs(roll) > self.max_pitchroll or abs(pitch) > self.max_pitchroll:
            return (True, self.tipping_penalty)
        else:
            return (False, 0)
    
    def _is_truncated(self) -> bool:
        return (self._episode_step_counter >= self.max_episode_len)

    def _get_info(self) -> dict:
        return {
            "height": self._agent.state.base_position[2],
            "pitch": self._agent.state.roll_pitch_yaw[1],
            "episode_step": self._episode_step_counter
        }

    def _calculate_reward(self, action: np.ndarray) -> tuple[float, dict]:
        return self._reward_fn(self, action)

    @property
    def agent(self):
        return self._agent
    
    @property
    def num_steps(self) -> int:
        return self._total_steps_counter

    @property
    def simulation_frequency(self) -> int:
        return self.sim_frequency
    
    @property
    def maximum_episode_len(self) -> int:
        return self.max_episode_len
