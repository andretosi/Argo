from collections import deque
from dataclasses import dataclass, field

import numpy as np

from commands.device import Device, Input
from spotmicro.sim import MujocoSim, quaternion_to_euler, quaternion_to_matrix


def body_to_robot_quaternion(q_body: np.ndarray) -> np.ndarray:
    """Orientation of the robot frame, given the orientation of the MuJoCo base body.

    In the model the head points to the body's -x axis (and its left to -y). The robot frame
    is the body frame turned by pi about z, so x points to the head, y to the left, z up:
    ``q_robot = q_body * (0, 0, 0, 1)``.
    """
    w, x, y, z = q_body
    return np.array([-z, y, -x, w])


@dataclass
class AgentState:
    """Current state of the robot.

    Orientation and velocities are in the robot frame: x towards the head, y to the left,
    z up. A positive forward command (vx > 0) therefore means walking head first.
    """

    base_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    feet_contacts: set[int] = field(default_factory=set)
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(0))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(0))
    joint_efforts: np.ndarray = field(default_factory=lambda: np.zeros(0))  # actuator torques [N m]
    # Derived from base_orientation (kept in sync by set_orientation)
    roll_pitch_yaw: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # robot -> world

    def set_orientation(self, quaternion: np.ndarray) -> None:
        self.base_orientation = quaternion
        self.roll_pitch_yaw = quaternion_to_euler(quaternion)
        self.rotation = quaternion_to_matrix(quaternion)


class Joint:
    """
    All data used to define a Joint.
    """

    def __init__(
        self,
        name: str,
        joint_id: int,
        joint_link_idx: int,
        joint_type: str,
        limits: tuple[float, float],
        max_torque: float,
        shoulder_deadzone: float,
        leg_deadzone: float,
        foot_deadzone: float,
        left_shoulder_hp: float,
        right_shoulder_hp: float,
        front_legs_hp: float,
        rear_legs_hp: float,
        front_feet_hp: float,
        rear_feet_hp: float,
    ) -> None:
        self.name = name
        self.leftright = name.split("_")[1]
        self.frontback = name.split("_")[0]
        self.id = joint_id
        self.link_id = joint_link_idx
        if limits[0] >= limits[1]:
            raise ValueError(f"Joint {self.name} has invalid limits: {limits}")
        self.limits = limits
        self.type = joint_type  # shoulder, leg, foot
        self.max_torque = max_torque

        # --- Type-dependent homing & gains ---
        if self.type == "shoulder":
            self.homing_position = (
                left_shoulder_hp if self.leftright == "left" else right_shoulder_hp
            )
            self.deadzone = shoulder_deadzone
        elif self.type == "leg":
            self.homing_position = front_legs_hp if self.frontback == "front" else rear_legs_hp
            self.deadzone = leg_deadzone
        elif self.type == "foot":
            self.homing_position = front_feet_hp if self.frontback == "front" else rear_feet_hp
            self.deadzone = foot_deadzone

    def from_position_to_action(self, pos: float) -> float:
        """Map joint position -> action in [-1, 1] (lower limit -> -1, upper limit -> 1)."""
        low, high = self.limits
        return (2 * pos - high - low) / (high - low)

    def from_action_to_position(self, action: float) -> float:
        """Map action ∈ [-1,1] → joint position."""
        a = float(np.clip(action, -1.0, 1.0))
        low, high = self.limits
        norm_hp = self.from_position_to_action(self.homing_position)

        def lin_map(x: float, xa: float, ya: float, xb: float, yb: float) -> float:
            return yb + (yb - ya) / (xb - xa) * (x - xb)

        if abs(a - norm_hp) < self.deadzone:
            return self.homing_position
        elif (a - norm_hp) < 0:
            return lin_map(a, -1, low, norm_hp - self.deadzone, self.homing_position)
        else:
            return lin_map(a, 1, high, norm_hp + self.deadzone, self.homing_position)


@dataclass
class AgentParams:
    """Robot settings (the ``agent:`` section of a run config)."""

    joint_max_torque: float = 6.5
    left_shoulder_hp: float = -0.0502
    right_shoulder_hp: float = 0.0502
    front_legs_hp: float = -0.55
    rear_legs_hp: float = -0.5
    front_feet_hp: float = 1.1
    rear_feet_hp: float = 1
    shoulder_deadzone: float = 0.07
    leg_deadzone: float = 0.075
    foot_deadzone: float = 0.075
    homing_pitch: float = -0.065  # spawn pitch of the MuJoCo base body (head towards its -x)
    max_joint_velocity: float = 10
    max_norm_height: float = 0.235
    max_linear_velocity: float = 2.23
    max_angular_velocity: float = 5
    joint_history_maxlen: int = 5  # at least 5: the observation uses the state 4 steps back


class Agent:
    """
    The robot: joints, state and action handling. All simulator access goes through MujocoSim.

    The methods are meant to be called in this order:
    apply_action() -> sim.step() -> sync_state()
    """

    def __init__(
        self,
        sim: MujocoSim,
        model_path: str,
        spawn_height: float,
        command: Device,
        action_space_size: int,
        params: AgentParams | None = None,
    ) -> None:
        self.params = params if params is not None else AgentParams()
        p = self.params
        self._action_space_size = action_space_size
        self.command = command
        self._sim = sim

        # <----- PARAMETERS ----->
        self.homing_pitch = p.homing_pitch
        self.max_joint_velocity = p.max_joint_velocity
        self.max_norm_height = p.max_norm_height
        self.max_linear_velocity = p.max_linear_velocity
        self.max_angular_velocity = p.max_angular_velocity
        if p.joint_history_maxlen < 5:
            raise ValueError("joint_history_maxlen must be at least 5")
        self.joint_history_maxlen = p.joint_history_maxlen

        # <----- State ----->
        init_orientation = self._sim.quaternion_from_euler(0, self.homing_pitch, np.pi)
        self._state = AgentState(base_position=np.array([0.0, 0.0, spawn_height]))
        self._state.set_orientation(body_to_robot_quaternion(init_orientation))
        self._action = np.zeros(self._action_space_size, dtype=np.float32)
        self._previous_action = np.zeros(self._action_space_size, dtype=np.float32)
        self._joint_history = deque(maxlen=self.joint_history_maxlen)

        # Load the model into the simulation
        self._sim.load_model(model_path, self._state.base_position, init_orientation)

        # --- Joints ---
        motor_joints = []
        homing_positions = []

        for ji in self._sim.get_joint_infos():
            joint = Joint(
                ji.name,
                ji.joint_id,
                ji.link_id,
                ji.joint_type,
                ji.limits,
                p.joint_max_torque,
                p.shoulder_deadzone,
                p.leg_deadzone,
                p.foot_deadzone,
                p.left_shoulder_hp,
                p.right_shoulder_hp,
                p.front_legs_hp,
                p.rear_legs_hp,
                p.front_feet_hp,
                p.rear_feet_hp,
            )
            motor_joints.append(joint)
            homing_positions.append(joint.homing_position)

        self._motor_joints = tuple(motor_joints)
        self._homing_positions = np.array(homing_positions)
        self.default_actions = np.array(
            [j.from_position_to_action(j.homing_position) for j in self.motor_joints]
        )

        # The same joint data as arrays (joint order), for vectorized per-step work
        self._ids = np.array([j.id for j in motor_joints])
        self._low = np.array([j.limits[0] for j in motor_joints])
        self._high = np.array([j.limits[1] for j in motor_joints])
        self._deadzone = np.array([j.deadzone for j in motor_joints])
        self._norm_hp = self.default_actions.copy()
        self._foot_link_ids = np.array([j.link_id for j in motor_joints if j.type == "foot"])

    def reset(self, spawn_heigt: float) -> None:
        """Reset agent state."""
        init_orientation = self._sim.quaternion_from_euler(0, self.homing_pitch, np.pi)

        self._state = AgentState(base_position=np.array([0.0, 0.0, spawn_heigt]))
        self._state.set_orientation(body_to_robot_quaternion(init_orientation))
        self._joint_history.clear()
        dummy_joint_state = (np.copy(self._homing_positions), np.zeros(len(self._motor_joints)))
        for _ in range(self.joint_history_maxlen):
            self._joint_history.append(dummy_joint_state)

        self._sim.reset_robot(
            position=np.array([0.0, 0.0, spawn_heigt]),
            orientation=init_orientation,
            joint_ids=[j.id for j in self._motor_joints],
            joint_positions=self._homing_positions.tolist(),
        )

        self._action = np.zeros(len(self._motor_joints), dtype=np.float32)
        self._previous_action = np.zeros(len(self._motor_joints), dtype=np.float32)
        self.command.reset()

    def apply_action(self, action: np.ndarray) -> None:
        """Map a normalized action ([-1, 1] per joint) to joint targets and send them to the simulation."""
        self._previous_action = self._action.copy()
        self._action = action

        self._sim.apply_joint_controls(self._ids, self.actions_to_positions(action))

    def actions_to_positions(self, action: np.ndarray) -> np.ndarray:
        """Vectorized ``Joint.from_action_to_position`` over all joints (same arithmetic)."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        hp, norm_hp, dz = self._homing_positions, self._norm_hp, self._deadzone
        xb_low, xb_high = norm_hp - dz, norm_hp + dz
        below = hp + (hp - self._low) / (xb_low - -1) * (a - xb_low)
        above = hp + (hp - self._high) / (xb_high - 1) * (a - xb_high)
        offset = a - norm_hp
        return np.where(np.abs(offset) < dz, hp, np.where(offset < 0, below, above))

    def sync_state(self) -> None:
        self._sim.forward()  # bring body poses and contacts up to date with qpos/qvel
        self._update_state()
        self._update_joint_history()

    def _update_state(self) -> None:
        """Query the simulation and update AgentState (orientation and velocities in the robot frame)."""
        state = self._state
        state.base_position = self._sim.get_base_position()
        state.set_orientation(body_to_robot_quaternion(self._sim.get_base_orientation()))
        lin_vel_world, ang_vel_world = self._sim.get_base_velocity()

        world_to_robot = state.rotation.T
        state.linear_velocity = world_to_robot @ lin_vel_world
        state.angular_velocity = world_to_robot @ ang_vel_world
        state.joint_positions, state.joint_velocities = self._sim.get_joint_states(self._ids)
        state.joint_efforts = self._sim.get_actuator_forces(self._ids)
        state.feet_contacts = self._sim.get_feet_contacts(self._foot_link_ids)

    def _update_joint_history(self) -> None:
        self._joint_history.append((self._state.joint_positions, self._state.joint_velocities))

    # --- Accessors ---
    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def last_action(self) -> np.ndarray:
        """The action applied at the latest control step."""
        return self._action

    @property
    def previous_action(self) -> np.ndarray:
        """The action applied at the control step before the latest one."""
        return self._previous_action

    @property
    def joint_history(self) -> deque[tuple[np.ndarray, np.ndarray]]:
        return self._joint_history

    @property
    def homing_positions(self) -> np.ndarray:
        return self._homing_positions

    @property
    def motor_joints(self) -> tuple[Joint, ...]:
        return self._motor_joints

    @property
    def command_input(self) -> Input:
        """The current velocity command; raises if it is outside [-1, 1]."""
        command = self.command.read()
        if not np.all(np.abs(command.as_array) <= 1.0):
            raise ValueError(f"Command {command} is not normalized to [-1, 1]")
        return command

    def get_feet_positions(self) -> np.ndarray:
        """Returns the positions of the feet (the toes, which touch the ground) in the world frame.

        Returns:
            np.ndarray: Matrix [4, 3] with the 4 position vectors of the feet
        """
        return self._sim.get_foot_tip_positions(self._foot_link_ids)

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

        u_body = self._state.rotation[:, 2]  # body z axis (the same in body and robot frames)

        height = np.dot(v_feet_to_body, u_body)
        return height
