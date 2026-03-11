from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class JointInfo:
    """Physics-agnostic representation of a joint queried from the simulation."""
    name: str
    joint_id: int
    link_id: int
    joint_type: str  # "revolute", "fixed", etc.
    limits: tuple     # (lower, upper)
    qposadr: int = 0  # MuJoCo-specific, ignored by PyBullet


class PhysicsBackend(ABC):
    """
    Abstract interface that isolates all physics-engine calls.

    Every method that touches PyBullet / MuJoCo lives behind this wall.
    The Agent and Env classes depend only on this interface.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Return the backend identifier, e.g. ``mujoco`` or ``pybullet``."""
        ...

    # ── Lifecycle ──────────────────────────────────────────────
    @abstractmethod
    def load_model(self, model_path: str, position: np.ndarray, orientation: np.ndarray) -> None:
        """Load the robot model into the simulation."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Tear down the simulation / viewer."""
        ...

    # ── Simulation stepping ────────────────────────────────────
    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one timestep."""
        ...

    @abstractmethod
    def sync_viewer(self) -> None:
        """Update the GUI viewer (no-op when headless)."""
        ...

    # ── Robot queries ──────────────────────────────────────────
    @abstractmethod
    def get_base_position(self) -> np.ndarray:
        """Return (3,) world-frame position of the robot base."""
        ...

    @abstractmethod
    def get_base_orientation(self) -> np.ndarray:
        """Return (4,) quaternion (w,x,y,z) of the robot base."""
        ...

    @abstractmethod
    def get_base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (linear_vel (3,), angular_vel (3,)) in world frame."""
        ...

    @abstractmethod
    def get_joint_state(self, joint_id: int) -> tuple[float, float]:
        """Return (position, velocity) of a single joint."""
        ...

    @abstractmethod
    def get_joint_infos(self) -> list[JointInfo]:
        """Return a list of JointInfo for every *revolute* joint."""
        ...

    @abstractmethod
    def get_link_position(self, link_id: int) -> np.ndarray:
        """Return (3,) world-frame position of the given link."""
        ...

    @abstractmethod
    def get_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert a (4,) quaternion to a (3,3) rotation matrix."""
        ...

    @abstractmethod
    def euler_from_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert a (4,) quaternion to (roll, pitch, yaw)."""
        ...

    @abstractmethod
    def quaternion_from_euler(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert (roll, pitch, yaw) to a (4,) quaternion."""
        ...

    # ── Robot control ──────────────────────────────────────────
    @abstractmethod
    def apply_joint_controls(self, joint_ids: list[int], positions: list[float], max_torques: list[float]) -> None:
        """Set position targets for the given joints."""
        ...

    @abstractmethod
    def reset_robot(self, position: np.ndarray, orientation: np.ndarray,
                    joint_ids: list[int], joint_positions: list[float]) -> None:
        """Reset base pose, zero velocities, and set joint homing."""
        ...

    # ── Contact detection ──────────────────────────────────────
    @abstractmethod
    def get_feet_contacts(self, foot_link_ids: list[int]) -> set[int]:
        """Return the set of foot link IDs currently in contact with the ground."""
        ...
