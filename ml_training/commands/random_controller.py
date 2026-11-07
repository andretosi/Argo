# Author: Nico
import random
from dataclasses import dataclass
from typing import NoReturn

import numpy as np

from commands.device import Device, DeviceConfig, Input


class RandomController(Device):
    """
    A simple device that gives the agent random commands.
    How a command is sampled depends on the current state of the device: Walk, Turn or Still.
    The device stays in a state for a randomly sampled number of steps.
    In short, the device is a Markov chain: the transition from state A to state B happens
    with a given probability.

    Methods
    --------
    update -> None:
        Update the current state, and transition to a new one if necessary. Should be invoked at every control step
    read -> Input:
        Read the current input. Should be called by the Env to provide the input to the policy and the reward function
    reset -> None:
        Bring the device to the initial conditions, to start a new episode

    """

    name = "random"

    @dataclass
    class Params(DeviceConfig):
        p_base2still: float = 0.3
        p_base2walk: float = 0.7
        p_base2turn: float = 0.0
        p_still2walk: float = 1.0
        p_still2turn: float = 0.0
        p_walk2still: float = 0.2
        p_walk2turn: float = 0.8
        p_turn2still: float = 0.1
        p_turn2walk: float = 0.9
        v_mean: tuple[float, float] = (0.23, 0.0)
        v_var: tuple[float, float] = (0.2, 0.02)
        v_steps_mean: float = 300
        v_steps_var: float = 37
        w_mean: float = 0.0
        w_var: float = 0.4
        w_radius_mean: float = 0.3
        w_radius_var: float = 0.1
        w_steps_mean: float = 100
        w_steps_var: float = 13
        s_steps_mean: float = 50
        s_steps_var: float = 10

    def __init__(self, params: "RandomController.Params | None" = None) -> None:
        self.params = params if params is not None else self.Params()
        self._input = Input()

        # Copy the parameters onto the device
        for name, value in vars(self.params).items():
            setattr(self, name, value)

        if self.p_base2still + self.p_base2walk + self.p_base2turn != 1.0:
            raise ValueError("The probabilities out of the base state must sum to 1.0")
        if self.p_still2walk + self.p_still2turn != 1.0:
            raise ValueError("The probabilities out of the still state must sum to 1.0")
        if self.p_walk2still + self.p_walk2turn != 1.0:
            raise ValueError("The probabilities out of the walk state must sum to 1.0")
        if self.p_turn2still + self.p_turn2walk != 1.0:
            raise ValueError("The probabilities out of the turn state must sum to 1.0")

        # Initialize states
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
        if self._state is not next_state:  # Works only because there are no self-loops
            self._state = next_state
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
    """A state of the RandomController; subclasses set ``transitions`` and the command."""

    transitions: dict[str, float]  # next-state tag -> probability, set by each subclass

    def __init__(self, controller: RandomController) -> None:
        self.controller = controller
        self.remaining_steps = 0

    def __str__(self) -> str:
        return "Base state"

    def update(self) -> "State":
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            return self._next_state()
        else:
            return self

    def _next_state(self) -> "State":
        """
        Pick the next state at random, using the probabilities in ``transitions``
        (next-state tag -> probability).
        """
        return self._map_state(
            random.choices(list(self.transitions.keys()), list(self.transitions.values()))[0]
        )

    def _map_state(self, tag: str) -> "State":
        if tag == "still":
            return self.controller.still_state
        elif tag == "walk":
            return self.controller.walk_state
        elif tag == "turn":
            return self.controller.turn_state
        raise ValueError(f"Unknown state tag: {tag!r}")

    def reset(self) -> None:
        self.remaining_steps = 0


class BaseState(State):
    """
    SPECS
    -----
    **REQUIRES**: State implements an update method
    **SIGNALS**: cannot be entered
    **ENSURES**: remaining_steps <= 0
    """

    def __init__(self, controller: RandomController) -> None:
        super().__init__(controller)
        self.transitions = {
            "still": self.controller.p_base2still,
            "turn": self.controller.p_base2turn,
            "walk": self.controller.p_base2walk,
        }
        print("Initializing base state")

    def __str__(self) -> str:
        return "Base state"

    def enter(self) -> NoReturn:
        raise NotImplementedError(
            "Base state does not implement the enter method, since it shpuld not be entered"
        )


class TurnState(State):
    def __init__(self, controller: RandomController) -> None:
        super().__init__(controller)
        self.transitions = {
            "still": self.controller.p_turn2still,
            "walk": self.controller.p_turn2walk,
        }

    def __str__(self) -> str:
        return "Turning state"

    def enter(self) -> None:
        vx, vy, w = self._sample_command()
        self.controller._input.update(vx=vx, vy=vy, w=w)
        self.remaining_steps = int(
            np.random.normal(self.controller.w_steps_mean, self.controller.w_steps_var)
        )

    def _sample_command(self) -> tuple[float, float, float]:
        w = np.clip(np.random.normal(self.controller.w_mean, self.controller.w_var), -1.0, 1.0)
        R = np.clip(
            np.random.normal(self.controller.w_radius_mean, self.controller.w_radius_var), -1.0, 1.0
        )  # turning radius, normalized
        vx = w * R  # both factors are in [-1, 1], so vx is too

        return vx, 0.0, w


class WalkState(State):
    def __init__(self, controller: RandomController) -> None:
        super().__init__(controller)
        self.transitions = {
            "still": self.controller.p_walk2still,
            "turn": self.controller.p_walk2turn,
        }

    def __str__(self) -> str:
        return "Walk state"

    def enter(self) -> None:
        vx, vy, w = self._sample_command()
        self.controller._input.update(vx=vx, vy=vy, w=w)
        self.remaining_steps = int(
            np.random.normal(self.controller.v_steps_mean, self.controller.v_steps_var)
        )

    def _sample_command(self) -> tuple[float, float, float]:
        vx, vy = np.clip(
            tuple(np.random.normal(self.controller.v_mean, self.controller.v_var)),
            (-1.0, -1.0),
            (1.0, 1.0),
        )
        return vx, vy, 0.0


class StillState(State):
    def __init__(self, controller: RandomController) -> None:
        super().__init__(controller)
        self.transitions = {
            "walk": self.controller.p_still2walk,
            "turn": self.controller.p_still2turn,
        }

    def __str__(self) -> str:
        return "Still state"

    def enter(self) -> None:
        self.controller._input.update(vx=0.0, vy=0.0, w=0.0)
        self.remaining_steps = int(
            np.random.normal(self.controller.s_steps_mean, self.controller.s_steps_var)
        )
