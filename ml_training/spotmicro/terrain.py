import numpy as np
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, FractalType, NoiseType


class Heightmap:
    def __init__(self, heightmap_data: np.ndarray) -> None:
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
    def from_noise(
        cls,
        x: int = 200,
        y: int = 200,
        z_max: float = 1.0,
        scale: float = 100.0,
        octaves: int = 5,
        gain: float = 0.5,
        lacunarity: float = 2.0,
        seed: int | None = None,
    ) -> "Heightmap":
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
            gain (float, optional): (a.k.a. 'persistence'). Amplitude multiplier
                                    for each subsequent octave. < 1 = fainter details.
            lacunarity (float, optional): Frequency multiplier for each
                                        subsequent octave. > 1 = finer details.
            seed (int, optional): Seed for the random generation of the noise.
        """
        if seed is None:
            seed = np.random.randint(0, 10000)

        # --- 1. Noise generator ---
        noise = FastNoiseLite(seed=seed)
        noise.noise_type = NoiseType.NoiseType_OpenSimplex2
        noise.frequency = 1.0 / scale  # 'scale' is the inverse of the frequency

        # --- 2. Fractal settings (octaves) ---
        noise.fractal_type = FractalType.FractalType_FBm
        noise.fractal_octaves = octaves
        noise.fractal_lacunarity = lacunarity
        noise.fractal_gain = gain

        # --- 3. Generation, on a grid of cell coordinates (shape [y, x]) ---
        xx, yy = np.meshgrid(np.arange(x), np.arange(y))
        # gen_from_coords is vectorized and takes float32 coordinates of shape [2, N]
        coords = np.vstack([xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)])
        flat_noise = noise.gen_from_coords(coords)

        heightmap = flat_noise.reshape((y, x))

        # --- 4. Normalization: the noise is in [-1, 1]; bring it to [0, z_max] ---
        heightmap = (heightmap + 1) / 2 * z_max
        return cls(heightmap)
