import numpy as np
import pybullet
from scipy.ndimage import gaussian_filter

from src.spotmicro.tools.config import Config

class Terrain:
    def __init__(self, physics_client, config: Config):
        self.physics_client = physics_client
        self._config = config
        self._terrain_id = None
        self._tilt_step = 0

    def generate(self, coefficients: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:    

        self._terrain_id = pybullet.loadURDF(
            "plane.urdf",
            physicsClientId=self.physics_client
        )
        self._generated = 1
        return self._terrain_id

    def reset(self):
        self._tilt_step = 0
        self._tilt_phase = np.random.uniform(0, 2 * np.pi)
    
    def tilt_plane(self):
        
        self._tilt_step += 1
        freq = self._config.rotation_frequency
        max_angle = self._config.max_tilt_angle

        tilt_x = max_angle * np.sin(freq * self._tilt_step + self._tilt_phase)
        tilt_y = max_angle * np.cos(freq * self._tilt_step + self._tilt_phase)

        quat = pybullet.getQuaternionFromEuler([tilt_x, tilt_y, 0])

        pybullet.resetBasePositionAndOrientation(
            self._terrain_id,
            posObj=[0, 0, 0],
            ornObj=quat,
            physicsClientId=self.physics_client
        )

    
    def _gen_pothole_heightfield(self, c_potholes: float):
        rows = cols = self._config.grid_size
        cell = self._config.cell_size
        H = np.zeros((rows, cols), dtype=np.float32)

        if c_potholes <= 0.0:
            return H  # no potholes

        xs = np.arange(cols) * cell
        ys = np.arange(rows) * cell
        X, Y = np.meshgrid(xs, ys)

        rng = np.random.default_rng(42)

        # Scale parameters
        n_pits = int(self._config.n_pits * c_potholes)
        pit_depth = self._config.pit_depth * c_potholes
        pit_radius = max(0.5 * self._config.pit_radius,
                        self._config.pit_radius * (1.0 - 0.5 * c_potholes))

        cx = rng.uniform(xs.min()+pit_radius, xs.max()-pit_radius, size=n_pits)
        cy = rng.uniform(ys.min()+pit_radius, ys.max()-pit_radius, size=n_pits)

        r2 = pit_radius**2
        for x0, y0 in zip(cx, cy):
            d2 = (X - x0)**2 + (Y - y0)**2
            mask = d2 < r2
            w = 1.0 - np.sqrt(d2[mask]) / pit_radius
            H[mask] -= pit_depth * (w**2)

        if self._config.smooth_sigma_cells and self._config.smooth_sigma_cells > 0:
            H = gaussian_filter(H, sigma=self._config.smooth_sigma_cells)

        return H


    def _create_terrain(self, coefficients: tuple[float, float, float]) -> None:
        """coefficients = (c_potholes, c_ridges, c_roughness)"""
        c_potholes, c_ridges, c_roughness = coefficients
        rows = cols = self._config.grid_size
        cell = self._config.cell_size

        print(f"ridges coefficient: {c_ridges}")

        # Ridges
        y = np.linspace(0, cols*cell, cols)
        Y = np.tile(y, (rows, 1))
        ridges_height = self._config.ridges_height * c_ridges
        ridges_distance = max(0.5 * self._config.ridges_distance,
                            self._config.ridges_distance * (1.0 - 0.5 * c_ridges))
        ridges = ridges_height * np.sin(2*np.pi * Y / ridges_distance)

        # Base heightfield
        H = self._gen_pothole_heightfield(c_potholes)
        H += ridges

        # Roughness (Gaussian noise bumps)
        if c_roughness > 0.0:
            noise = np.random.randn(rows, cols) * (self._config.max_height_bumps * 0.2 * c_roughness)
            H += noise

        # Clip and finalize
        H = np.clip(H, self._config.min_height_bumps, self._config.max_height_bumps)
        heightfield_data = H.flatten().tolist()

        shape = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_HEIGHTFIELD,
            meshScale=[cell, cell, 1.0],
            heightfieldTextureScaling=(rows - 1) / 2,
            heightfieldData=heightfield_data,
            numHeightfieldRows=rows,
            numHeightfieldColumns=cols,
            physicsClientId=self.physics_client
        )

        terrain_id = pybullet.createMultiBody(0, shape)
        return terrain_id


    @property
    def terrain_id(self):
        return self._terrain_id

    @property
    def config(self):
        return self._config