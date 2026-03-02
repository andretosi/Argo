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

    def get_num_joints(self) -> int:
        """
        Not strictly necessary but... really useful
        """
        pass
    
    def get_joint_info(self, joint_id):
        """
        Just an idea
        """
        pass

    def get_base_info(self):
        """
        Need to get height, orientation? and velocity (linear and angular ig)
        """

    def reset_agent(self):
        """
        Need a way to:
        - Reset base position and orientation
        - Reset base velocity
        - Reset all joints (pos and vel)
        """
        pass

    def apply_action(sellf, action):
        """
        Apply the given action to the agent
        """
        pass

    def get_feet_contacts(self) -> set:
        """
        Obtain a set of feet touching the ground
        """
        pass