"""MuJoCo simulation of the SpotMicro robot.

Conventions: quaternions are (w, x, y, z); positions and velocities are in the world frame.
"""

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class JointInfo:
    """A revolute joint of the robot, as read from the MuJoCo model."""

    name: str
    joint_id: int  # actuator id used to command the joint
    link_id: int  # MuJoCo joint index
    joint_type: str  # "shoulder", "leg" or "foot" (last part of the joint name)
    limits: tuple[float, float]  # (lower, upper)


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
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


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, sinp) if np.abs(sinp) >= 1 else np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class MujocoSim:
    """Owns the MuJoCo model, its data and (optionally) the viewer window.

    The timestep comes from the model XML; ``SpotmicroEnv`` overrides it with
    ``1 / sim_frequency`` (``set_timestep``).
    """

    def __init__(self, use_gui: bool = False) -> None:
        self._model = None
        self._data = None
        self._viewer = None
        self._use_gui = use_gui
        # Mapping from joint name to actuator id (populated in load_model)
        self._joint_to_actuator: dict[str, int] = {}
        # Per actuator: index of its joint in data.qpos / data.qvel (populated in load_model)
        self._act_qposadr = np.zeros(0, dtype=int)
        self._act_dofadr = np.zeros(0, dtype=int)
        self._ground_geom_id: int = -1
        self._collision_geom_ids = np.zeros(0, dtype=int)
        # Per geom: the foot (joint index) it belongs to, -1 for none; built on first use
        self._geom_to_foot: np.ndarray | None = None
        self._geom_to_foot_key: tuple[int, ...] = ()

    # ── Lifecycle ──────────────────────────────────────────────
    def load_model(self, model_path: str, position: np.ndarray, orientation: np.ndarray) -> None:
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self._ground_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self._ground_geom_id < 0:
            raise RuntimeError("No geom named 'floor' found in the model XML.")

        # Build joint_name -> actuator_id mapping
        self._joint_to_actuator = {}
        for i in range(self._model.nu):
            act_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name:
                joint_name = act_name.replace("_ctrl", "")
                self._joint_to_actuator[joint_name] = i

        # The geoms that can touch the ground: the ground itself and the robot's collision geoms
        collides = (self._model.geom_contype != 0) | (self._model.geom_conaffinity != 0)
        self._collision_geom_ids = np.flatnonzero(collides)

        actuated_joints = self._model.actuator_trnid[:, 0]
        self._act_qposadr = self._model.jnt_qposadr[actuated_joints].copy()
        self._act_dofadr = self._model.jnt_dofadr[actuated_joints].copy()

        if self._use_gui:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            # close() only asks the viewer to stop: it renders in its own thread, which then
            # tears the window down. If the program exits before that, the interpreter can
            # crash or hang during shutdown, so wait for the thread to finish.
            for thread in threading.enumerate():
                if "_launch_internal" in thread.name and thread is not threading.current_thread():
                    thread.join(timeout=5.0)

    # ── Simulation stepping ────────────────────────────────────
    def step(self) -> None:
        mujoco.mj_step(self._model, self._data)

    def forward(self) -> None:
        """Recompute derived quantities (body poses, contacts, sensors) for the current state.

        ``mj_step`` leaves them as they were at the start of its last step, one timestep
        behind ``qpos``/``qvel``.
        """
        mujoco.mj_forward(self._model, self._data)

    def set_timestep(self, timestep: float) -> None:
        self._model.opt.timestep = timestep

    def sync_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()

    # ── Robot queries ──────────────────────────────────────────
    # The base is read from the free joint's qpos/qvel, which are always current.
    def get_base_position(self) -> np.ndarray:
        return self._data.qpos[0:3].copy()

    def get_base_orientation(self) -> np.ndarray:
        """Quaternion (w, x, y, z) of the base body (MuJoCo body frame: head towards -x)."""
        q = self._data.qpos[3:7].copy()
        return q / np.linalg.norm(q)

    def get_base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (linear, angular) velocity of the base body origin, in the world frame."""
        # Free joint: qvel[0:3] is the linear velocity in the world frame, qvel[3:6] the
        # angular velocity in the body frame.
        rot = quaternion_to_matrix(self.get_base_orientation())
        return self._data.qvel[0:3].copy(), rot @ self._data.qvel[3:6]

    def get_joint_states(self, actuator_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions, velocities) of the joints driven by ``actuator_ids``."""
        return (
            self._data.qpos[self._act_qposadr[actuator_ids]],
            self._data.qvel[self._act_dofadr[actuator_ids]],
        )

    def get_actuator_forces(self, actuator_ids: np.ndarray) -> np.ndarray:
        """Torques [N m] the given actuators apply (after their force limits)."""
        return self._data.actuator_force[actuator_ids].copy()

    def get_joint_infos(self) -> list[JointInfo]:
        infos = []
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is None:
                continue
            if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                category = name.split("_")[-1]
                actuator_id = self._joint_to_actuator.get(name, i)
                infos.append(
                    JointInfo(
                        name=name,
                        joint_id=actuator_id,
                        link_id=i,
                        joint_type=category,
                        limits=(
                            float(self._model.jnt_range[i, 0]),
                            float(self._model.jnt_range[i, 1]),
                        ),
                    )
                )
        return infos

    def get_foot_tip_positions(self, foot_link_ids: np.ndarray) -> np.ndarray:
        """World positions of the foot tips (toe body origins), one row per foot joint."""
        bodies = self._model.jnt_bodyid[foot_link_ids]
        tips = []
        for body in bodies:
            children = np.flatnonzero(self._model.body_parentid == body)
            children = children[children != body]  # the world body is its own parent
            tips.append(children[0] if len(children) == 1 else body)
        return self._data.xpos[tips]

    def get_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        return quaternion_to_matrix(quaternion)

    def euler_from_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        return quaternion_to_euler(quaternion)

    def quaternion_from_euler(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        return quaternion_from_euler(roll, pitch, yaw)

    # ── Robot control ──────────────────────────────────────────
    def apply_joint_controls(self, actuator_ids: np.ndarray, positions: np.ndarray) -> None:
        """Set the position targets of the given actuators."""
        self._data.ctrl[actuator_ids] = positions

    def reset_robot(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        joint_ids: list[int],
        joint_positions: list[float],
    ) -> None:
        self._data.qpos[:] = 0
        self._data.qvel[:] = 0
        self._data.ctrl[:] = 0

        # Base position (free joint: first 3 = pos, next 4 = quat)
        self._data.qpos[0:3] = position
        self._data.qpos[3:7] = orientation

        # Joint homing (joint_ids are actuator ids)
        for jid, jpos in zip(joint_ids, joint_positions):
            self._data.qpos[self._act_qposadr[jid]] = jpos

        self.clear_external_forces()
        mujoco.mj_forward(self._model, self._data)

    # ── Contact detection ──────────────────────────────────────
    def _foot_geom_map(self, foot_link_ids: Sequence[int]) -> np.ndarray:
        """Per geom: the foot joint it belongs to, or -1.

        A foot touches the ground through its own geoms and those of its child bodies (the toe).
        """
        key = tuple(int(i) for i in foot_link_ids)
        if self._geom_to_foot is None or key != self._geom_to_foot_key:
            body_to_foot = np.full(self._model.nbody, -1)
            for link_id in key:
                foot_body = self._model.jnt_bodyid[link_id]
                for body in range(self._model.nbody):
                    b = body
                    while b > 0 and b != foot_body:
                        b = self._model.body_parentid[b]
                    if b == foot_body:
                        body_to_foot[body] = link_id
            self._geom_to_foot = body_to_foot[self._model.geom_bodyid]
            self._geom_to_foot_key = key
        return self._geom_to_foot

    def get_feet_contacts(self, foot_link_ids: Sequence[int]) -> set[int]:
        """The feet (joint indices from ``foot_link_ids``) touching the ground."""
        geom_to_foot = self._foot_geom_map(foot_link_ids)
        ncon = self._data.ncon
        if ncon == 0:
            return set()
        geom1 = self._data.contact.geom1[:ncon]
        geom2 = self._data.contact.geom2[:ncon]
        ground = self._ground_geom_id
        other = np.where(geom1 == ground, geom2, geom1)[(geom1 == ground) | (geom2 == ground)]
        feet = geom_to_foot[other]
        return {int(f) for f in feet[feet >= 0]}

    # ── Extra accessors ────────────────────────────────────────
    @property
    def model(self) -> mujoco.MjModel | None:
        return self._model

    @property
    def data(self) -> mujoco.MjData | None:
        return self._data

    # ── Terrain management ──────────────────────────────────────
    def spawn_terrain(
        self, heightmap_data: np.ndarray, scale: Sequence[float], origin: Sequence[float]
    ) -> int:
        """Load a heightmap into the model's ``terrain`` hfield, in place.

        Heights are kept in metres (multiplied by ``scale[2]``), cells are ``scale[0]`` x
        ``scale[1]`` metres, and the terrain is centred vertically around ``origin``. Heights
        above the hfield's elevation limit (set in the XML) are clipped.

        Returns:
            int: Hfield ID of the terrain (pass it to ``remove_terrain``)
        """
        if self._model is None:
            raise RuntimeError(
                "Model must be loaded before spawning terrain. Call load_model() first."
            )
        hfield_id = self._terrain_hfield_id()
        nrow = self._model.hfield_nrow[hfield_id]
        ncol = self._model.hfield_ncol[hfield_id]

        # Resize heightmap to match hfield dimensions
        from scipy.ndimage import zoom

        scale_factors = (nrow / heightmap_data.shape[0], ncol / heightmap_data.shape[1])
        heights = zoom(heightmap_data, scale_factors) * scale[2]
        heights = heights - heights.min()

        elevation = self._model.hfield_size[hfield_id][2]
        if heights.max() > elevation:
            heights = np.minimum(heights, elevation)

        adr = self._model.hfield_adr[hfield_id]
        self._model.hfield_data[adr : adr + nrow * ncol] = (heights / elevation).flatten()
        # Footprint: (n - 1) cells of `scale` metres; hfield_size holds half-extents
        self._model.hfield_size[hfield_id][0] = (ncol - 1) * scale[0] / 2
        self._model.hfield_size[hfield_id][1] = (nrow - 1) * scale[1] / 2
        self._model.geom_pos[self._ground_geom_id] = [
            origin[0],
            origin[1],
            origin[2] - heights.max() / 2,
        ]
        self._refresh_terrain(hfield_id)
        return hfield_id

    def remove_terrain(self, terrain_handle: int) -> None:
        """Make the floor flat again, at z = 0."""
        if self._model is None:
            return
        hfield_id = terrain_handle
        nrow = self._model.hfield_nrow[hfield_id]
        ncol = self._model.hfield_ncol[hfield_id]
        adr = self._model.hfield_adr[hfield_id]
        self._model.hfield_data[adr : adr + nrow * ncol] = 0.0
        self._model.geom_pos[self._ground_geom_id] = [0.0, 0.0, 0.0]
        self._refresh_terrain(hfield_id)

    def _terrain_hfield_id(self) -> int:
        hfield_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        if hfield_id < 0:
            raise RuntimeError(
                "No hfield named 'terrain' found in model. Add <hfield name='terrain' .../> to XML."
            )
        return hfield_id

    def _refresh_terrain(self, hfield_id: int) -> None:
        mujoco.mj_forward(self._model, self._data)
        if self._viewer is not None:
            self._viewer.update_hfield(hfield_id)
            self._viewer.sync()

    # ── External forces (for domain randomization) ─────────────
    def apply_external_force(self, force: np.ndarray) -> None:
        """Push the base: apply ``force`` (world frame, at its centre of mass) until
        ``clear_external_forces`` is called.

        The environment clears forces after every control step, so a force lasts for the
        control step(s) during which it is applied.
        """
        # xfrc_applied: (nbody, 6) rows of [force(3), torque(3)]; body 1 is base_link
        self._data.xfrc_applied[1, 0:3] = force

    def clear_external_forces(self) -> None:
        """Remove all applied external forces (MuJoCo keeps them until cleared)."""
        self._data.xfrc_applied[:] = 0.0

    def get_base_mass(self) -> float:
        """Return the total mass of the robot."""
        # body 0 is the world body and should not be counted
        return float(np.sum(self._model.body_mass[1:]))

    def set_friction(self, friction: float) -> None:
        """Set the sliding friction between the robot and the ground."""
        if self._model is not None and self._ground_geom_id >= 0:
            # MuJoCo geom_friction: [sliding, torsional, rolling]
            self._model.geom_friction[self._collision_geom_ids, 0] = friction
