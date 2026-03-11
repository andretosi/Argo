from src.spotmicro.devices.device import Device
from src.spotmicro.agent.input import Input

class FixedController(Device):
    """
    Very simple controller that can be either used to:
    - Train an agent to stay still (input will hold zeroes)
    - Train an agnt to walk forward at a constant pace (only non-zero input will be vx)
    """

    def __init__(self, mode: str = "walk"):
        if mode in ["still", "walk"]:
            self.mode = mode
        else:
            raise ValueError(f"Unknown mode requested ({mode})")
        
        if self.mode == "still":
            self.i = Input(0, 0, 0)
        elif self.mode == "walk":
            self.i = Input(0.2, 0, 0)
    
    def update(self):
        return

    def read(self) -> Input:
        return self.i

    def reset(self):
        return