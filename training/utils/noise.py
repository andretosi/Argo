"""
Sensor Noise Injection for Sim-to-Real Transfer
================================================

Provides configurable noise injection following legged_gym scales.

Default noise scales (from legged_gym):
- DOF position: 0.01 rad
- DOF velocity: 1.5 rad/s
- Linear velocity: 0.1 m/s
- Angular velocity: 0.2 rad/s
- Gravity: 0.05 (direction noise)

Usage:
    from training.utils.noise import SensorNoise
    from spotmicro.tools.config import Config

    cfg = Config()
    noise = SensorNoise(
        config=cfg,
        dof_pos_noise=0.01,
        dof_vel_noise=1.5,
        enable=True
    )

    # In observation function:
    noisy_obs = noise.apply(clean_obs)
"""

import numpy as np
from dataclasses import dataclass

from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable


@dataclass
class NoiseConfig:
    """
    Configuration for sensor noise injection (legacy dataclass for backwards compat).

    All values are noise scales (standard deviation for Gaussian noise).
    """
    dof_pos_noise: float = 0.01
    dof_vel_noise: float = 1.5
    lin_vel_noise: float = 0.1
    ang_vel_noise: float = 0.2
    gravity_noise: float = 0.05
    height_noise: float = 0.02
    enable: bool = True


@configurable
class SensorNoise:
    """
    Applies configurable noise to observations for sim-to-real transfer.

    This class adds Gaussian noise to sensor readings, matching the noise
    characteristics expected from real robot sensors.

    The noise is applied to specific indices of the observation vector
    based on the SpotmicroEnv observation structure:

    Observation indices:
    - 0-2: gravity vector
    - 3: height
    - 4-6: linear velocity
    - 7-9: angular velocity
    - 10-21: joint positions (12 joints)
    - 22-33: joint velocities (12 joints)
    - 34-45: history positions t-1
    - 46-57: history velocities t-1
    - 58-69: history positions t-4
    - 70-81: history velocities t-4
    - 82-93: previous action
    - 94-96: velocity commands

    Parameters
    ----------
    config : Config
        Central config registry for saving/loading parameters
    dof_pos_noise : float
        Joint position noise std [rad] (default: 0.01)
    dof_vel_noise : float
        Joint velocity noise std [rad/s] (default: 1.5)
    lin_vel_noise : float
        Linear velocity noise std [m/s] (default: 0.1)
    ang_vel_noise : float
        Angular velocity noise std [rad/s] (default: 0.2)
    gravity_noise : float
        Gravity vector direction noise std (default: 0.05)
    height_noise : float
        Height measurement noise std [m] (default: 0.02)
    enable : bool
        Global enable/disable for noise (default: True)
    """

    def __init__(
        self,
        config: Config,
        dof_pos_noise: float = 0.01,
        dof_vel_noise: float = 1.5,
        lin_vel_noise: float = 0.1,
        ang_vel_noise: float = 0.2,
        gravity_noise: float = 0.05,
        height_noise: float = 0.02,
        enable: bool = True
    ):
        self.dof_pos_noise = dof_pos_noise
        self.dof_vel_noise = dof_vel_noise
        self.lin_vel_noise = lin_vel_noise
        self.ang_vel_noise = ang_vel_noise
        self.gravity_noise = gravity_noise
        self.height_noise = height_noise
        self.enable = enable

    def apply(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply noise to observation vector.

        Parameters
        ----------
        obs : np.ndarray
            Clean observation vector (97,)

        Returns
        -------
        np.ndarray
            Noisy observation vector
        """
        if not self.enable:
            return obs

        noisy_obs = obs.copy()

        # Gravity vector (indices 0-2)
        noisy_obs[0:3] += np.random.normal(0, self.gravity_noise, 3)
        # Renormalize gravity vector
        norm = np.linalg.norm(noisy_obs[0:3])
        if norm > 1e-8:
            noisy_obs[0:3] /= norm

        # Height (index 3)
        noisy_obs[3] += np.random.normal(0, self.height_noise)

        # Linear velocity (indices 4-6)
        noisy_obs[4:7] += np.random.normal(0, self.lin_vel_noise, 3)

        # Angular velocity (indices 7-9)
        noisy_obs[7:10] += np.random.normal(0, self.ang_vel_noise, 3)

        # Joint positions - current (indices 10-21)
        noisy_obs[10:22] += np.random.normal(0, self.dof_pos_noise, 12)

        # Joint velocities - current (indices 22-33)
        noisy_obs[22:34] += np.random.normal(0, self.dof_vel_noise, 12)

        # History positions (indices 34-45 and 58-69)
        noisy_obs[34:46] += np.random.normal(0, self.dof_pos_noise, 12)
        noisy_obs[58:70] += np.random.normal(0, self.dof_pos_noise, 12)

        # History velocities (indices 46-57 and 70-81)
        noisy_obs[46:58] += np.random.normal(0, self.dof_vel_noise, 12)
        noisy_obs[70:82] += np.random.normal(0, self.dof_vel_noise, 12)

        return noisy_obs

    def reset(self):
        """Reset noise state (no-op for stateless Gaussian noise)."""
        pass


@configurable
class CorrelatedNoise:
    """
    Correlated (colored) noise for more realistic sensor simulation.

    Uses Ornstein-Uhlenbeck process for temporally correlated noise.
    This produces noise that changes smoothly over time, better
    simulating real sensor drift and bias.

    Parameters
    ----------
    config : Config
        Central config registry for saving/loading parameters
    size : int
        Dimension of noise vector
    theta : float
        Mean reversion rate (higher = faster return to zero)
    sigma : float
        Volatility (noise amplitude)
    dt : float
        Time step in seconds
    """

    def __init__(
        self,
        config: Config,
        size: int = 97,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1/60
    ):
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.zeros(size)

    def sample(self) -> np.ndarray:
        """Sample next noise value from OU process."""
        dx = -self.theta * self.state * self.dt
        dx += self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.state += dx
        return self.state.copy()

    def reset(self):
        """Reset noise state to zero."""
        self.state = np.zeros(self.size)
