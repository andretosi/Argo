import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

#TODO make this class as much agnostic as possible from the use case
@dataclass
class Input:
    """
    This utility class represents a command issued to the controller
    """
    vx: float = 0.0
    vy: float = 0.0
    w: float = 0.0

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy , self.w])
    
    def update(self, vx=None, vy=None, w=None):
        """Overwrite all or some fields."""
        if vx is not None:
            self.vx = vx
        if vy is not None:
            self.vy = vy
        if w is not None:
            self.w = w
        
        #print(f"Updated input: {self.as_array}")
