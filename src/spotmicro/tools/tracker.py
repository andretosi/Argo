from ..agent.input import Input
from .tracker_renderer import TrackerRenderer

#TODO -> specialize for the different backends?

class Tracker():
    """
    The idea for this class are:
    - Make the logic for handling the tracker modular and limited
    - Use just one object to handle different types of physical backend (that are being introduced in the near fututre). "TrackerRenderer" may be polymorphically differentiated in different classes that provide the logic to just draw the tracker in the simulation (eg we may implement one for pybullet, one for mujoco...) -> imma use a factory?
    """
    def __init__(self, trackerRenderer: TrackerRenderer, dt: float):
        self.renderer = trackerRenderer #Make this polymorphic, and keep the rest of the logic env agnostic
        self.renderer.spawn()
        pass

    def reset(self) -> None:
        """
        Put the tracker back into the starting position and reset its properties
        """
        self.renderer.reset()
        pass

    def apply_command(self, i: Input) -> None:
        """
        Move the tracker according to the given command
        """
        self.renderer.update()
        pass
