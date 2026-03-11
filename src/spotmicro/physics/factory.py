from src.spotmicro.physics.backend import PhysicsBackend


def create_backend(engine: str, use_gui: bool = False, sim_frequency: int = 240) -> PhysicsBackend:
    """Factory that returns the right PhysicsBackend for the given engine name.

    Parameters
    ----------
    engine : str
        ``"pybullet"`` or ``"mujoco"``.
    use_gui : bool
        Whether to open a visualisation window.
    sim_frequency : int
        Simulation timestep frequency in Hz.
    """
    if engine == "pybullet":
        from src.spotmicro.physics.pybullet_backend import PybulletBackend
        return PybulletBackend(use_gui=use_gui, sim_frequency=sim_frequency)
    elif engine == "mujoco":
        from src.spotmicro.physics.mujoco_backend import MujocoBackend
        return MujocoBackend(use_gui=use_gui, sim_frequency=sim_frequency)
    else:
        raise ValueError(f"Unknown physics engine: {engine!r}. Expected 'pybullet' or 'mujoco'.")
