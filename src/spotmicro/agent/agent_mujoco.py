import numpy as np
import mujoco
from dataclasses import dataclass, field
from collections import deque
from importlib.resources import files

from src.spotmicro.tools.config import Config
from src.spotmicro.tools.configurable import configurable
from src.spotmicro.devices.device import Device
from src.spotmicro.agent.controller import Controller
from src.spotmicro.physics.mujoco_backend import MujocoBackend


def quaternion_from_euler(roll, pitch, yaw):
    """Convert euler angles to quaternion (w, x, y, z)"""
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def quaternion_to_euler(q):
    """Convert quaternion to euler angles (roll, pitch, yaw)"""
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def quaternion_to_matrix(q):
    """Convert quaternion to rotation matrix (3x3)"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


#questa classe contiene soltanto dei dati
#nello specifico contiene tutte informazioni relative allo stato corrente 
#dell'agent nell'ambiente. 

#@dataclass è un'etichetta che serve per rendere molto più leggibile la definizione
#degli attributi della classe

@dataclass
class AgentState:
    """
    this dataclass contains all the useful data about the current state of the agent, with the velocities being stored in its own space coordinates
    """
    base_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    feet_contacts: set = field(default_factory=set)
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(0))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def roll_pitch_yaw(self):
        return quaternion_to_euler(self.base_orientation)

class Joint:
    """
    All data used to define a Joint
    """
    def __init__(self, name: str, joint_id: int, joint_link_idx: int, joint_type: str, limits: tuple,
                 max_torque, shoulder_deadzone, leg_deadzone, foot_deadzone, 
                 left_shoulder_hp, right_shoulder_hp, front_legs_hp, rear_legs_hp, front_feet_hp, rear_feet_hp,
                 qposadr=0, qveladr=0
            ):
        """
        Parameters
        ----------
        name : str
            name of the joint.
        joint_id: int
            position of the joint in the array with all the joints
        joint_link_idx: int
            internal id used by mujoco to identify the link associated with the joint
        joint_type: str
            type of the joint: shoulder, leg, foot
        limits: tuple
            (min, max) positional limits of the joint
        config: Config
            set of attributes taken from agentConfig.yaml
        qposadr: int
            address of this joint's position in the qpos array
        qveladr: int
            address of this joint's velocity in the qvel array
        """
        self.name = name
        self.leftright = name.split("_")[1]
        self.frontback = name.split("_")[0]
        self.id = joint_id
        self.link_id = joint_link_idx
        self.qposadr = qposadr
        self.qveladr = qveladr
        if limits[0] >= limits[1]:
            raise ValueError(f"Joint {self.name} has invalid limits: {limits}")
        self.limits = limits
        self.mid = 0.5 * (self.limits[0] + self.limits[1])
        self.type = joint_type  # shoulder, leg, foot
        self.effort = 0
        self.max_torque = max_torque

        # --- Type-dependent homing & gains ---
        if self.type == "shoulder":
            self.homing_position = left_shoulder_hp if self.leftright == "left" else right_shoulder_hp
            self.deadzone = shoulder_deadzone

        elif self.type == "leg":
            self.homing_position = front_legs_hp if self.frontback == "front" else rear_legs_hp
            self.deadzone = leg_deadzone

        elif self.type == "foot":
            self.homing_position = front_feet_hp if self.frontback == "front" else rear_feet_hp
            self.deadzone = foot_deadzone
    # the neural network outputs a NORMALIZED vector with the action that the robot should perform.
    # This function converts the vector into a joint position, used by mujoco to move the robot.
    def from_position_to_action(self, pos: float) -> float:
        high, low = self.limits
        return (2*pos - high - low) / (high - low)

    def from_action_to_position(self, action: float) -> float:
        """Map action ∈ [-1,1] → joint position."""
        a = float(np.clip(action, -1.0, 1.0))
        low, high = self.limits
        norm_hp = self.from_position_to_action(self.homing_position)

        lin_map = lambda x, xa, ya, xb, yb: yb + (yb - ya) / (xb - xa) * (x - xb)

        if abs(a - norm_hp) < self.deadzone:
            return self.homing_position
        elif (a - norm_hp) < 0:
            return lin_map(a, -1, low, norm_hp-self.deadzone, self.homing_position)
        else:
            return lin_map(a, 1, high, norm_hp+self.deadzone, self.homing_position)


@configurable
class Agent:
    """
    This class represents the Robot in the simulation. The data is taken from the mujoco simulation.
    
    Attributes
    ------------
    - _config : Config 
        contains all the data written in agentConfig.yaml
    - _state : AgentState 
        contains all the useful data about the current state of the Agent
    - _model : mujoco.MjModel
        MuJoCo model containing the robot definition
    - _data : mujoco.MjData
        MuJoCo data containing simulation state
    - _action : npArray
        it's a vector the size of action_space_size
    - _motor_joints : tuple(Joint, ...)
        list of all the joints that can revolute of the robot.    
        It's the same size of the action vector
    - _joint_history: queue [tuple(joint_positions : npArray, joint_velocities : npArray), ...]
        Queue that holds history of joint_positions and joint_velocities.
    
    Methods
    ----------
    - reset(spawn_height: float):
        Reset agent state and simulation. Body position, orientation, 
        and joint position are set to homing in mujoco simulation.

    - apply_action(action: np.ndarray):
        This method takes as input a NORMALIZED action, maps it to joint positions and applies it
        to the joints through mujoco. All the data about the Agent is then updated, based on the 
        new state in which it ended up.
    
    - sync_state():
        This method updates AgentState and the joint_history. 
        This must be called after mujoco.mj_step() in order to update values of the Agent class.

    - _get_feet_contacts():
        This method saves which feet are touching the ground. This info is part of the state vector
    
    - _update_state():
        Query mujoco and update AgentState.
    
    - _update_joint_history(): 
        Enqueues the current joint velocities and joint positions.

    Notes
    --------
    The methods are meant to be called in this order:
    apply_action() -> mujoco.mj_step() -> sync_state()


    """
    def __init__(self, env, device: Device, config: Config, action_space_size: int,
                 joint_max_torque=6.5, left_shoulder_hp=-0.0502, right_shoulder_hp=0.0502, front_legs_hp=-0.55, rear_legs_hp=-0.5, front_feet_hp=1.1, rear_feet_hp=1,
                 shoulder_deadzone=0.07, leg_deadzone=0.075, foot_deadzone=0.075, homing_pitch=-0.065,
                 max_joint_velocity=10, max_norm_height=0.235, max_linear_velocity=2.23, max_forward_linear_velocity=2.0, max_lateral_linear_velocity=1.0, max_angular_velocity=5,
                 joint_history_maxlen=5
            ):
        
        self.config = config
        self._action_space_size = action_space_size
        self._controller = Controller(device)
        self._env = env

        #<----- PARAMTERS INITIALIZATION ----->
        self.homing_pitch = homing_pitch
        self.max_joint_velocity = max_joint_velocity
        self.max_norm_height = max_norm_height
        self.max_linear_velocity = max_linear_velocity
        self.max_forward_linear_velocity = max_forward_linear_velocity
        self.max_lateral_linear_velocity = max_lateral_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.joint_history_maxlen = joint_history_maxlen

        # <----- State ----->
        self._state = AgentState(
            base_position=np.array([0.0, 0.0, self._env.spawn_height]),
            base_orientation=quaternion_from_euler(0, self.homing_pitch, np.pi),
        )
        self._action = np.zeros(self._action_space_size, dtype=np.float32)
        self._previous_action = np.zeros(self._action_space_size, dtype=np.float32)
        self._joint_history = deque(maxlen=self.joint_history_maxlen)

        # Load MuJoCo model
        xml_path = str(files("src.spotmicro.data").joinpath("spotmicroai.mujoco.xml"))
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        self._physics_backend = MujocoBackend()
        self._physics_backend._model = self._model
        self._physics_backend._data = self._data
        self._physics_backend._ground_geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
        )

        # --- Joints ---
        motor_joints = []
        homing_positions = []

        # Build a mapping from joint name to actuator ID
        joint_to_actuator = {}
        for i in range(self._model.nu):
            act_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name:
                # Actuator names match joint names (e.g., "front_left_shoulder_ctrl" -> joint "front_left_shoulder")
                joint_name = act_name.replace("_ctrl", "")
                joint_to_actuator[joint_name] = i

        # Query joint information from mujoco model
        for i in range(self._model.njnt):
            joint_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None:
                continue
            
            joint_type = self._model.jnt_type[i]
            if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                joint_limits = (self._model.jnt_range[i, 0], self._model.jnt_range[i, 1])
                qposadr = self._model.jnt_qposadr[i]
                qveladr = self._model.jnt_dofadr[i]
                
                joint_category = joint_name.split("_")[-1]
                actuator_id = joint_to_actuator.get(joint_name, i)
                joint = Joint(
                    joint_name, actuator_id, i, joint_category, joint_limits, 
                    joint_max_torque, shoulder_deadzone, leg_deadzone, foot_deadzone, left_shoulder_hp, right_shoulder_hp,
                    front_legs_hp, rear_legs_hp, front_feet_hp, rear_feet_hp,
                    qposadr=qposadr, qveladr=qveladr
                )
                motor_joints.append(joint)
                homing_positions.append(joint.homing_position)

        self._motor_joints = tuple(motor_joints)
        self._homing_positions = np.array(homing_positions)

        self.default_actions = np.array([j.from_position_to_action(j.homing_position) for j in self.motor_joints])

    def reset(self, spawn_heigt: float):
        """
        Reset agent state
        """

        self._state = AgentState(
            base_position=np.array([0.0, 0.0, spawn_heigt]),
            base_orientation=quaternion_from_euler(0, self.homing_pitch, np.pi),
        )
        self._joint_history.clear()
        dummy_joint_state = (np.copy(self._homing_positions), np.zeros(len(self._motor_joints)))
        for _ in range(5):
            self._joint_history.append(dummy_joint_state)

        # Reset MuJoCo state
        self._data.qpos[:] = 0
        self._data.qvel[:] = 0
        self._data.ctrl[:] = 0
        
        # Set base position (first 3 elements of qpos for free joint)
        self._data.qpos[0] = 0.0
        self._data.qpos[1] = 0.0
        self._data.qpos[2] = spawn_heigt
        
        # Set base orientation (next 4 elements for quaternion)
        ori = quaternion_from_euler(0, self.homing_pitch, np.pi)
        self._data.qpos[3] = ori[0]  # w
        self._data.qpos[4] = ori[1]  # x
        self._data.qpos[5] = ori[2]  # y
        self._data.qpos[6] = ori[3]  # z
        
        # Set joint positions to homing using qposadr
        for i, joint in enumerate(self._motor_joints):
            self._data.qpos[joint.qposadr] = self.homing_positions[i]

        # Reset actions to "homing" which is 0
        self._action = np.zeros(len(self._motor_joints), dtype=np.float32)
        self._previous_action = np.zeros(len(self._motor_joints), dtype=np.float32)

        #Reset controller
        self.controller.reset()
    
    def apply_action(self, action: np.ndarray):
        """
        This method takes as input a NORMALIZED action, maps it to joint positions and sets 
        the commands of the joints in the mujoco simulation. 
        """
        self._previous_action = self._action.copy()
        self._action = action
        
        # Set joint control targets
        for i, joint in enumerate(self._motor_joints):
            self._data.ctrl[joint.id] = joint.from_action_to_position(action[i])

        return

    def sync_state(self):
        self._update_state()
        self._update_joint_history()

    def _get_feet_contacts(self) -> set:
        """
        This method saves which feet are touching the ground (part of the state vector)
        returns a set of link indices of the feet in contact with the ground
        """
        foot_link_ids = [joint.link_id for joint in self._motor_joints if joint.type == "foot"]
        return self._physics_backend.get_feet_contacts(foot_link_ids)

    def _update_state(self):
        """Query mujoco and update AgentState with velocities in robot-space coordinates"""

        # Get base position, orientation (world frame) - body 1 is base_link
        pos = self._data.xpos[1].copy()
        ori = self._data.xquat[1].copy()

        # Get base linear and angular velocity (world frame)
        lin_vel_world = self._data.cvel[1, :3].copy()  # linear velocity
        ang_vel_world = self._data.cvel[1, 3:].copy()  # angular velocity

        # Compute rotation matrix world -> body (robot) frame
        rot_matrix = quaternion_to_matrix(ori)
        world_to_body = rot_matrix.T

        # Transform velocities to robot frame
        lin_vel_body = world_to_body @ lin_vel_world
        ang_vel_body = world_to_body @ ang_vel_world

        # Get joint positions, velocities, and efforts
        joint_positions = []
        joint_velocities = []
        for joint in self._motor_joints:
            joint_positions.append(self._data.qpos[joint.qposadr])
            joint_velocities.append(self._data.qvel[joint.qveladr])
            joint.effort = self._data.actuator_force[joint.id]

        # Update agent state
        self._state.base_position = pos
        self._state.base_orientation = ori
        self._state.linear_velocity = lin_vel_body
        self._state.angular_velocity = ang_vel_body
        self._state.joint_positions = np.array(joint_positions)
        self._state.joint_velocities = np.array(joint_velocities)
        self._state.feet_contacts = self._get_feet_contacts()

    def _update_joint_history(self):
        self._joint_history.append((self._state.joint_positions, self._state.joint_velocities))

    #this methods return the values of some of the class attributes
    # --- Accessors ---
    @property
    def state(self) -> AgentState:
        """
        Return the current state of the agent. This property provides access to:
        - position of the base in worldspace coordinates
        - orientation of the base as a quaternion
        - linear velocity of the base in the agent's own space coordinates
        - angular velocity of the base in the agent's own space coordiantes
        - a set of the agent's feet currently touching the ground
        - the position of each joint as an angle (rad) inside an array
        - the velocity of each joint (rad/s) inside an array
        """
        return self._state

    @property
    def agent_id(self):
        return 1  # MuJoCo body ID for base_link

    @property
    def previous_action(self) -> np.ndarray:
        return self._previous_action

    @previous_action.setter
    def previous_action(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray) and len(value) == self._action_space_size:
            self._previous_action = value
        else:
            raise ValueError(f"previous_action must be a numpy array of length {self._action_space_size}")

    @property
    def action(self) -> np.ndarray:
        return self._action
    
    #TODO: dunno why i put it here, makes much more sense for it to be in the env. Still...
    @property
    def joint_history(self) -> deque:
        return self._joint_history
    
    @joint_history.setter
    def joint_history(self, history: deque):
        if isinstance(history, deque) and len(history) <= self.config.joint_history_maxlen:
            self._joint_history = history
    
    @property
    def homing_positions(self) -> np.ndarray:
        return self._homing_positions
    
    @property
    def motor_joints(self) -> tuple:
        return self._motor_joints

    @property
    def controller(self) -> Controller:
        return self._controller
    
    def get_feet_positions(self) -> np.ndarray:
        """ Returns the positions of the feet with respect to the Golbal Frame

        Returns:
            np.ndarray: Matrix [4, 3] with the 4 position vectors of the feet
        """
        feet_positions = []
        for joint in self._motor_joints:
            if joint.type == "foot":
                # Get the body ID for this joint's body
                body_id = self._model.jnt_bodyid[joint.id]
                feet_positions.append(self._data.xpos[body_id].copy())
        
        return np.array(feet_positions)

    def get_body_to_feet_height_projected(self) -> float:
        """Calculates the projected height of the body over the feet centroid.

        This method determines the body's height relative to the average position
        (centroid) of the feet. The calculation is performed by projecting the
        vector from the feet centroid to the body's center of mass onto the
        robot's own vertical axis ($z_b$). This makes the measurement
        independent of the robot's overall tilt.

        The height is computed using the dot product:

        $$
        H = (\\mathbf{p}_\\text{body} - \\mathbf{p}_\\text{feet\\_avg}) \\cdot \\mathbf{u}_\\text{body}
        $$

        Where:
            - $\\mathbf{p}_\\text{body}$: Position of the body's center of mass.
            - $\\mathbf{p}_\\text{feet\\_avg}$: Centroid of the feet positions.
            - $\\mathbf{u}_\\text{body}$: Unit vector of the body's vertical axis.

        Returns:
            float: The projected scalar height of the body.
        """
        # 1. Calcolo del centroide dei piedi from the feet centroid, p_feet_avg
        feet_positions = self.get_feet_positions()
        p_feet_avg = np.mean(feet_positions, axis=0)

        # 2. Calcolo del vettore che collega centroide e corpo, v_feet->body
        p_body = self._state.base_position
        v_feet_to_body = p_body - p_feet_avg

        # 3. Trovare la direzione "su" del robot, u_body
        orientation_quat = self._data.xquat[1].copy()
        rot_matrix = quaternion_to_matrix(orientation_quat)
        u_body = rot_matrix[:, 2]

        # 4. Proiezione scalare (prodotto scalare)
        height = np.dot(v_feet_to_body, u_body)

        return height
