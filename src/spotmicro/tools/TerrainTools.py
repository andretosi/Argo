from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, NoiseType, FractalType
import matplotlib.pyplot as plt
from pathlib import Path


class Heightmap:
    def __init__(self, heightmap_data):
        """
        Initializes the Heightmap object.

        Args:
            heightmap_data (np.ndarray): A 2D numpy array representing the heightmap.
        
        Raises:
            TypeError: If heightmap_data is not a numpy array.
        """
        if not isinstance(heightmap_data, np.ndarray):
            raise TypeError("heightmap_data must be a NumPy array.")
        self.data = heightmap_data

    @classmethod
    def from_noise(cls,
                   x=200,
                   y=200,
                   z_max=1.0,
                   scale=100.0,
                   octaves=5,
                   gain=0.5,
                   lacunarity=2.0,
                   seed=None):
        """
        Generates a 2D heightmap using pyfastnoiselite.

        Args:
            x (int): Width of the map (number of columns).
            y (int): Height of the map (number of rows).
            z_max (float, optional): The desired maximum height. 
                                    The final map will be normalized between [0, z_max].
            scale (float, optional): The "scale" of the noise. High values = "zoom in" 
                                    (larger features, low frequency).
            octaves (int, optional): Number of overlapping noise layers 
                                    to add details.
            gain (float, optional): (Ex 'persistence'). Amplitude multiplier 
                                    for each subsequent octave. < 1 = fainter details.
            lacunarity (float, optional): Frequency multiplier for each 
                                        subsequent octave. > 1 = finer details.
            seed (int, optional): Seed for the random generation of the noise.
        """
        if seed is None:
            seed = np.random.randint(0, 10000)

        # --- 1. Set up the Noise object ---
        noise = FastNoiseLite(seed=seed) # The constructor is the same
        noise.noise_type = NoiseType.NoiseType_OpenSimplex2 # Use properties instead of Set*/set_* methods
        noise.frequency = 1.0 / scale # 'scale' is the inverse of frequency.

        # --- 2. Fractal Settings (for octaves) ---
        noise.fractal_type = FractalType.FractalType_FBm
        noise.fractal_octaves = octaves
        noise.fractal_lacunarity = lacunarity
        noise.fractal_gain = gain
        
        # --- 3. Generation ---
        # We need to create the coordinate grid manually.
        # 1. Create the coordinate vectors for each axis
        x_coords = np.arange(x)
        y_coords = np.arange(y)
        
        # 2. Create a 2D coordinate grid (shape: [y, x])
        xx, yy = np.meshgrid(x_coords, y_coords)

        # 3. Generate the noise values in a vectorized way with gen_from_coords (required shape: [2, N], dtype float32)
        coords = np.vstack([xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)])
        flat_noise = noise.gen_from_coords(coords)

        # 4. Reshape the 1D output array into our [y, x] shape
        heightmap = flat_noise.reshape((y, x))

        # --- 4. Normalization ---
        heightmap = (heightmap + 1) / 2 * z_max # The noise is in [-1, 1]. We bring it to [0, z_max]
        return cls(heightmap)

    @classmethod
    def from_stairs(cls, y, x, z_max=1, z_min=0):
        """
        Creates a Heightmap with a staircase pattern.

        Args:
            y (int): Number of rows (height of the map).
            x (int): Number of columns (width of the map).
            z_max (float, optional): Maximum height of the stairs. Defaults to 1.
            z_min (float, optional): Minimum height of the stairs. Defaults to 0.

        Returns:
            Heightmap: A new Heightmap instance with the staircase pattern.
        """
        cols=x
        rows=y
        heightmap_data = np.zeros((rows, cols))
        height_current = z_min
        
        for i in range(0, cols, 2):
            if i < cols:
                heightmap_data[:, i] = height_current
            if i + 1 < cols:
                heightmap_data[:, i+1] = height_current
            
            height_current += (z_max - z_min) / (cols / 2)
        
        return cls(heightmap_data)

    def view(self):
        """
        Displays the 2D heightmap using matplotlib with a "terrain" colormap.

        Args:
            heightmap (np.ndarray): The 2D NumPy array containing the height data.

        Returns:
            tuple: A tuple containing the generated matplotlib objects:
                (fig, ax, im)
                - fig (matplotlib.figure.Figure): The main figure of the plot.
                - ax (matplotlib.axes.Axes): The axes of the plot.
                - im (matplotlib.image.AxesImage): The rendered image object.
        """
        num_rows, num_columns = self.data.shape
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(self.data,
                    cmap="terrain",
                    extent=[0, num_columns, 0, num_rows])
        ax.set_title("2D Visualization of the Heightmap")
        ax.set_xlabel("Columns (y)")
        ax.grid(True)
        ax.set_ylabel("Rows (x)")
        fig.colorbar(im, ax=ax, label="Height")
        plt.show()
        return(fig, ax, im)

@dataclass
class TerrainConfig:
    """
    Contains the terrain parameters.
    generation_params is a flexible dictionary for noise parameters (seed, x, y, etc.).
    """
    method: str  # "mesh" or "heightmap"
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0]) 
    origin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    mesh_path: Optional[str] = None
    generation_params: dict[str, Any] = field(default_factory=dict)


class Terrain:
    def __init__(self, config: TerrainConfig, data=None):
        """
        Initializes the Terrain object.

        Args:
            config (TerrainConfig): Configuration object for the terrain.
            data (Union[Heightmap, Path, None], optional): The terrain data. 
                Can be a Heightmap object or a Path to a mesh file. Defaults to None.
        """
        self.config = config
        self.data = data # Heightmap Object or path
        self.terrain_body_id = None

    @classmethod
    def from_heightmap(cls, heightmap: Heightmap, scale=[1, 1, 1], origin=[0, 0, 0], generation_params=None):
        """
        Creates a Terrain instance from a Heightmap object.

        Args:
            heightmap (Heightmap): The heightmap data.
            scale (list, optional): Scaling factor [x, y, z]. Defaults to [1, 1, 1].
            origin (list, optional): Origin coordinates [x, y, z]. Defaults to [0, 0, 0].
            generation_params (dict, optional): Parameters used to generate the heightmap. Defaults to None.

        Returns:
            Terrain: A Terrain instance configured with the heightmap.
        """
        if generation_params is None:
            generation_params = {}

        config = TerrainConfig(
            method="heightmap", 
            scale=scale, 
            origin=origin,
            generation_params=generation_params
        )
        return cls(config=config, data=heightmap)
    
    @classmethod
    def from_mesh(cls, path: str, scale=[1, 1, 1], origin=[0, 0, 0]):
        """
        Creates a Terrain instance from a mesh file.

        Args:
            path (str): Path to the mesh file (e.g., .obj, .stl).
            scale (list, optional): Scaling factor [x, y, z]. Defaults to [1, 1, 1].
            origin (list, optional): Origin coordinates [x, y, z]. Defaults to [0, 0, 0].

        Raises:
            FileNotFoundError: If the mesh file does not exist.

        Returns:
            Terrain: A Terrain instance configured with the mesh.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"The specified mesh file does not exists. Check the path or the presence of the file.\nPath: '{file_path}'")

        config = TerrainConfig(
            method="mesh",
            mesh_path=str(file_path),
            scale=scale,
            origin=origin
        )
        return cls(config=config, data=file_path)