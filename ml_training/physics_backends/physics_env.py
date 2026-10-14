from abc import ABC, abstractmethod


class PhysicsEnv(ABC):
    def __init__(self):
        """
        Create the physical engine, set initial parameters
        
        """
        pass

    @classmethod
    def attach_terrain(self, heightmap):
        """
        Given a heightmap, create a terrain from it and attach it to the physical simulation
        """
        pass
    
    @classmethod
    def close(self):
        """
        Gracefully close the simulation env
        """
        pass

    @classmethod
    def step(self, action):
        """
        Observe a discrete moment in time (step) inside the simulation and provide an action to the agent. This method may actively step the simulation forward, or merely observe its state as it independently goes on. 

        Takes an action, returns at least one observation
        """
        pass

    @classmethod
    def get_num_joints(self) -> int:
        """
        Not strictly necessary but... really useful
        """
        pass
    
    @classmethod
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