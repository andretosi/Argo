class PhysicsEnv:
    def __init__(self):
        """
        Create the physical engine, set initial parameters
        
        """
        pass

    def attach_terrain(self, heightmap):
        """
        Given an heightmap, create a terrain from it and attach it to the physical simulation
        """
        pass
    
    def close(self):
        """
        Gracefully close the simulation env
        """
        pass

    def step(self):
        """
        Step the simulation forward (may need other parameters?)
        """
        pass
