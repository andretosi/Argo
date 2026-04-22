#Author: Nico
from spotmicro.devices.device import Device
from spotmicro.agent.input import Input
from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable
import numpy as np
import random, os

@configurable
class RandomController(Device):
    """
    A simple device that provides the agent random inputs.
    How the input is sampled, depends on the current state of the device. There are three possible states: Walk, Turn, Still.
    \nThe device stays on a state for a randomly sampled number of steps. 
    In short, the device is modeled as a Markov Chain: the transition from state A to state B happens with a given probability.

    Methods
    --------
    update -> None:
        Update the current state, and transition to a new one if necessary. Should be invoked at every control step
    read -> Input:
        Read the current input. Should be called by the Env to provide the input to the policy and the reward function
    reset -> None:
        Bring the device to the initial conditions, to start a new episode

    """
    def __init__(self, config: Config, 
                p_base2still=0.3, p_base2walk=0.7, p_base2turn=0.0,
                p_still2walk=1.0, p_still2turn=0.0, 
                p_walk2still=0.2, p_walk2turn=0.8,
                p_turn2still=0.1, p_turn2walk=0.9,
                v_mean=(0.23, 0.0), v_var=(0.2, 0.02),
                v_steps_mean=300, v_steps_var=37,
                w_mean=0.0, w_var=0.4,
                w_radius_mean=0.3, w_radius_var=0.1,
                w_steps_mean=100, w_steps_var=13,
                s_steps_mean=50, s_steps_var=10
            ):
        
        self._input = Input()

        #initialize own parameters
        self.p_base2still = p_base2still
        self.p_base2walk = p_base2walk
        self.p_base2turn = p_base2turn
        self.p_still2walk = p_still2walk
        self.p_still2turn = p_still2turn
        self.p_walk2still = p_walk2still
        self.p_walk2turn = p_walk2turn
        self.p_turn2still = p_turn2still
        self.p_turn2walk = p_turn2walk
        self.v_mean = v_mean
        self.v_var = v_var
        self.v_steps_mean = v_steps_mean
        self.v_steps_var = v_steps_var
        self.w_mean = w_mean
        self.w_var = w_var
        self.w_radius_mean = w_radius_mean
        self.w_radius_var = w_radius_var
        self.w_steps_mean = w_steps_mean
        self.w_steps_var = w_steps_var
        self.s_steps_mean = s_steps_mean
        self.s_steps_var = s_steps_var

        if self.p_base2still + self.p_base2walk + self.p_base2turn != 1.0:
            raise ValueError("Sum of probabilities outgoing of base state must be exactly 1.0")
        if self.p_still2walk + self.p_still2turn != 1.0:
            raise ValueError("Sum of probabilities outgoing of still state must be exactly 1.0")
        if self.p_walk2still + self.p_walk2turn != 1.0:
            raise ValueError("Sum of probabilities outgoing of walk state must be exaclty 1.0")
        if self.p_turn2still + self.p_turn2walk != 1.0:
            raise ValueError("Sum of probabilities outgoing of turn state must be exaclty 1.0")

        #Initialize states        
        self.walk_state = WalkState(self)
        self.turn_state = TurnState(self)
        self.still_state = StillState(self)
        self.base_state = BaseState(self)

        self._state = self.base_state
     

    def update(self) -> None:
        """
        Update the internal state of the controller.
        May randomly change the input
        """
        next_state = self._state.update()
        if self._state is not next_state: #Works only because there are no self-loops
            self._state = next_state
            #print(self._state)
            self._state.enter()

    def read(self) -> Input:
        """
        Obtain the current input to give to the agent
        """
        return self._input
    
    def reset(self) -> None:
        self._state.reset()
        self._state = self.base_state

class State:
    """
    Serves as the initial state for this class, and the default one
    """
    def __init__(self, controller: RandomController):
        self.controller = controller
        self.remaining_steps = 0
    
    def __str__(self):
        return "Base state"

    def update(self):
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            return self._next_state()
        else:
            return self
    
    def _next_state(self):
        """
        Randomly chose a state from the given ones assigning to each a certain probability.
        \nState in position 0 will have a probabilities[0] probability of being returned, and so on.
        \nThis method should be overridden by each subclass
        """
        return self._map_state(random.choices(list(self.transitions.keys()), list(self.transitions.values()))[0])
    
    def _map_state(self, tag: str):
        if tag == "still":
            #print(f"Transitioning to {self.controller.still_state}")
            return self.controller.still_state
        elif tag == "walk":
            #print(f"Transitioning to {self.controller.walk_state}")
            return self.controller.walk_state
        elif tag == "turn":
            #print(f"Transitioning to {self.controller.turn_state}")
            return self.controller.turn_state

    def reset(self):
        self.remaining_steps = 0

class BaseState(State):
    """
    SPECS
    -----
    **REQUIRES**: State implements an update method 
    **SIGNALS**: cannot be entered
    **ENSURES**: remaining_steps <= 0
    """
    def __init__(self, controller: RandomController):
        super().__init__(controller)
        self.transitions = {
            "still": self.controller.p_base2still,
            "turn": self.controller.p_base2turn,
            "walk": self.controller.p_base2walk
        }
        print("Initializing base state")

    def __str__(self):
        return "Base state"
    
    def enter(self):
        raise NotImplementedError("Base state does not implement the enter method, since it shpuld not be entered")
    
    def _IR(self) -> bool:
        return (self.remaining_steps <= 0)
    
class TurnState(State):
    def __init__(self, controller: RandomController):
        super().__init__(controller)
        self.transitions = {
            "still": self.controller.p_turn2still,
            "walk": self.controller.p_turn2walk
        }

    def __str__(self):
        return "Turning state"

    def enter(self):
        vx, vy, w = self._sample_command()
        self.controller._input.update(vx=vx, vy=vy, w=w)
        self.remaining_steps = int(np.random.normal(
            self.controller.w_steps_mean,
            self.controller.w_steps_var
        ))
    
    def _sample_command(self):
        w = np.clip(np.random.normal(self.controller.w_mean, self.controller.w_var), -1.0, 1.0)
        R = np.clip(np.random.normal(self.controller.w_radius_mean, self.controller.w_radius_var), -1.0, 1.0) #Should be normalized.. does not cause any trouble for now
        vx = w*R

        return vx, 0.0, w

class WalkState(State):
    def __init__(self, controller: RandomController):
        super().__init__(controller)
        self.transitions = {
            "still": self.controller.p_walk2still,
            "turn": self.controller.p_walk2turn
        }

    def __str__(self):
        return "Walk state"

    def enter(self):
        vx, vy, w = self._sample_command()
        self.controller._input.update(vx=vx, vy=vy, w=w)
        self.remaining_steps = int(np.random.normal(
            self.controller.v_steps_mean,
            self.controller.v_steps_var
        ))
    
    def _sample_command(self):
        vx, vy = np.clip(tuple(np.random.normal(self.controller.v_mean, self.controller.v_var)), (-1.0, -1.0), (1.0, 1.0))
        return vx, vy, 0.0

class StillState(State):
    def __init__(self, controller: RandomController):
        super().__init__(controller)
        self.transitions = {
            "walk": self.controller.p_still2walk,
            "turn": self.controller.p_still2turn
        }

    def __str__(self):
        return "Still state"

    def enter(self):
        self.controller._input.update(vx=0.0, vy=0.0, w=0.0)
        self.remaining_steps = int(np.random.normal(
            self.controller.s_steps_mean,
            self.controller.s_steps_var
        ))