import numpy as np
from dataclasses import dataclass, field
from collections import deque

from spotmicro.physics.backend import PhysicsBackend, JointInfo
from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable
from spotmicro.devices.device import Device
from spotmicro.agent.controller import Controller


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

    _backend: PhysicsBackend = field(default=None, repr=False)

    @property
    def roll_pitch_yaw(self):
        return self._backend.euler_from_quaternion(self.base_orientation)


class Joint:
    """
    All data used to define a Joint (physics-agnostic).
    """
    def __init__(self, name: str, joint_id: int, joint_link_idx: int, joint_type: str, limits: tuple,
                 max_torque, shoulder_deadzone, leg_deadzone, foot_deadzone, 
                 left_shoulder_hp, right_shoulder_hp, front_legs_hp, rear_legs_hp, front_feet_hp, rear_feet_hp,
                 qposadr=0
            ):
        self.name = name
        self.leftright = name.split("_")[1]
        self.frontback = name.split("_")[0]
        self.id = joint_id
        self.link_id = joint_link_idx
        self.qposadr = qposadr
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
    Physics-agnostic Agent.  All simulator interaction goes through a PhysicsBackend.

    The methods are meant to be called in this order:
    apply_action() -> backend.step() -> sync_state()
    """
    def __init__(self, backend: PhysicsBackend, model_path: str, spawn_height: float,
                 device: Device, config: Config, action_space_size: int,
                 joint_max_torque=6.5, left_shoulder_hp=-0.0502, right_shoulder_hp=0.0502, front_legs_hp=-0.55, rear_legs_hp=-0.5, front_feet_hp=1.1, rear_feet_hp=1,
                 shoulder_deadzone=0.07, leg_deadzone=0.075, foot_deadzone=0.075, homing_pitch=-0.065,
                 max_joint_velocity=10, max_norm_height=0.235, max_linear_velocity=2.23, max_forward_linear_velocity=2.0, max_lateral_linear_velocity=1.0, max_angular_velocity=5,
                 joint_history_maxlen=5, target_body_to_feet_height=0.21
            ):
        
        self.config = config
        self._action_space_size = action_space_size
        self._controller = Controller(device)
        self._backend = backend

        #<----- PARAMTERS INITIALIZATION ----->
        self.homing_pitch = homing_pitch
        self.max_joint_velocity = max_joint_velocity
        self.max_norm_height = max_norm_height
        self.max_linear_velocity = max_linear_velocity
        self.max_forward_linear_velocity = max_forward_linear_velocity
        self.max_lateral_linear_velocity = max_lateral_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.joint_history_maxlen = joint_history_maxlen
        self.target_body_to_feet_height = target_body_to_feet_height

        self.ORIENTATION = [0, self.homing_pitch, np.pi]

        # <----- State ----->
        init_orientation = self._backend.quaternion_from_euler(0, self.homing_pitch, np.pi)
        self._state = AgentState(
<<<<<<< HEAD
            base_position=np.array([0.0, 0.0, self._env.spawn_height]), #TODO: spawn_height must be deduced, it cannot be (only) a manual parameter
<<<<<<< HEAD
            base_orientation=pybullet.getQuaternionFromEuler(self.ORIENTATION),
=======
            base_orientation=pybullet.getQuaternionFromEuler([0, self.homing_pitch, np.pi]), # <-- PHYSICS_ENV: utility -->
>>>>>>> 1bbb306 (notes on pybullet use in agent.py)
=======
            base_position=np.array([0.0, 0.0, spawn_height]),
            base_orientation=init_orientation,
            _backend=self._backend,
>>>>>>> ab1736d (add mujoco and env / agent interface)
        )
        self._action = np.zeros(self._action_space_size, dtype=np.float32)
        self._previous_action = np.zeros(self._action_space_size, dtype=np.float32)
        self._joint_history = deque(maxlen=self.joint_history_maxlen)

        # Load model via backend
        self._backend.load_model(model_path, self._state.base_position, init_orientation)

        # --- Joints ---
        motor_joints = []
        homing_positions = []

        for ji in self._backend.get_joint_infos():
            joint = Joint(
                ji.name, ji.joint_id, ji.link_id, ji.joint_type, ji.limits,
                joint_max_torque, shoulder_deadzone, leg_deadzone, foot_deadzone,
                left_shoulder_hp, right_shoulder_hp,
                front_legs_hp, rear_legs_hp, front_feet_hp, rear_feet_hp,
                qposadr=ji.qposadr,
            )
            motor_joints.append(joint)
            homing_positions.append(joint.homing_position)

        self._motor_joints = tuple(motor_joints)
        self._homing_positions = np.array(homing_positions)
        self.default_actions = np.array([j.from_position_to_action(j.homing_position) for j in self.motor_joints])

    def reset(self, spawn_heigt: float):
        """Reset agent state."""
        init_orientation = self._backend.quaternion_from_euler(0, self.homing_pitch, np.pi)

        self._state = AgentState(
            base_position=np.array([0.0, 0.0, spawn_heigt]),
<<<<<<< HEAD
<<<<<<< HEAD
            base_orientation=pybullet.getQuaternionFromEuler(self.ORIENTATION),
=======
            base_orientation=pybullet.getQuaternionFromEuler([0, self.homing_pitch, np.pi]), #<--PHYSICS_ENV: utils -->
>>>>>>> 1bbb306 (notes on pybullet use in agent.py)
            #linear_velocity=np.array([0.0, 0.0, 0.0])
            #angular_velocity=np.array([0.0, 0.0, 0.0])
        )#
=======
            base_orientation=init_orientation,
            _backend=self._backend,
        )
>>>>>>> ab1736d (add mujoco and env / agent interface)
        self._joint_history.clear()
        dummy_joint_state = (np.copy(self._homing_positions), np.zeros(len(self._motor_joints)))
        for _ in range(5):
            self._joint_history.append(dummy_joint_state)

        self._backend.reset_robot(
            position=np.array([0.0, 0.0, spawn_heigt]),
            orientation=init_orientation,
            joint_ids=[j.id for j in self._motor_joints],
            joint_positions=self._homing_positions.tolist(),
        )

        self._action = np.zeros(len(self._motor_joints), dtype=np.float32)
        self._previous_action = np.zeros(len(self._motor_joints), dtype=np.float32)
        self.controller.reset()
    
    def apply_action(self, action: np.ndarray):
        """Takes a NORMALIZED action, maps to joint positions and sends to backend."""
        self._previous_action = self._action.copy()
        self._action = action

        positions = [j.from_action_to_position(action[i]) for i, j in enumerate(self._motor_joints)]
        torques = [j.max_torque for j in self._motor_joints]
        ids = [j.id for j in self._motor_joints]
        self._backend.apply_joint_controls(ids, positions, torques)

    def sync_state(self):
        self._update_state()
        self._update_joint_history()

    def _update_state(self):
        """Query backend and update AgentState with velocities in robot-space coordinates."""
        pos = self._backend.get_base_position()
        ori = self._backend.get_base_orientation()
        lin_vel_world, ang_vel_world = self._backend.get_base_velocity()

        rot_matrix = self._backend.get_rotation_matrix(ori)
        world_to_body = rot_matrix.T

        lin_vel_body = world_to_body @ lin_vel_world
        ang_vel_body = world_to_body @ ang_vel_world

        joint_positions = []
        joint_velocities = []
        for joint in self._motor_joints:
            jpos, jvel = self._backend.get_joint_state(joint.id)
            joint_positions.append(jpos)
            joint_velocities.append(jvel)

        self._state.base_position = pos
        self._state.base_orientation = ori
        self._state.linear_velocity = lin_vel_body
        self._state.angular_velocity = ang_vel_body
        self._state.joint_positions = np.array(joint_positions)
        self._state.joint_velocities = np.array(joint_velocities)

        foot_link_ids = [j.link_id for j in self._motor_joints if j.type == "foot"]
        self._state.feet_contacts = self._backend.get_feet_contacts(foot_link_ids)

    def _update_joint_history(self):
        self._joint_history.append((self._state.joint_positions, self._state.joint_velocities))

    # --- Accessors ---
    @property
    def state(self) -> AgentState:
        return self._state

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
        """Returns the positions of the feet with respect to the Global Frame.

        Returns:
            np.ndarray: Matrix [4, 3] with the 4 position vectors of the feet
        """
        feet_positions = []
        for joint in self._motor_joints:
            if joint.type == "foot":
                feet_positions.append(self._backend.get_link_position(joint.link_id))
        return np.array(feet_positions)

    def get_body_to_feet_height_projected(self) -> float:
        """Calculates the projected height of the body over the feet centroid.

        $$
        H = (\\mathbf{p}_\\text{body} - \\mathbf{p}_\\text{feet\\_avg}) \\cdot \\mathbf{u}_\\text{body}
        $$

        Returns:
            float: The projected scalar height of the body.
        """
        feet_positions = self.get_feet_positions()
        p_feet_avg = np.mean(feet_positions, axis=0)

        p_body = self._state.base_position
        v_feet_to_body = p_body - p_feet_avg

        ori = self._backend.get_base_orientation()
        rot_matrix = self._backend.get_rotation_matrix(ori)
        u_body = rot_matrix[:, 2]

        height = np.dot(v_feet_to_body, u_body)
        return height
