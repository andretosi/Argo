"""Drive a policy with a gamepad, through SDL's game controller API (pygame._sdl2.controller).

The game controller API gives every supported gamepad the same layout (left stick, right
stick, ...). Gamepads SDL does not know can be described in an SDL mapping file
(``db_path``, e.g. the community gamecontrollerdb.txt).
"""

import os
from dataclasses import dataclass

import numpy as np
import pygame
from pygame._sdl2 import controller as sdl_controller

from commands.device import Device, DeviceConfig, Input


class Joystick(Device):
    name = "joystick"

    @dataclass
    class Params(DeviceConfig):
        db_path: str = "gamecontrollerdb.txt"  # SDL controller mappings, loaded if present

    def __init__(self, params: "Joystick.Params | None" = None) -> None:
        db_path = (params if params is not None else self.Params()).db_path
        # SDL reads extra mappings from this file when the controller subsystem starts
        if os.path.exists(db_path):
            os.environ.setdefault("SDL_GAMECONTROLLERCONFIG_FILE", os.path.abspath(db_path))
        pygame.init()
        sdl_controller.init()

        self.controller: sdl_controller.Controller | None = None
        self._input_vector = np.zeros(3, dtype=np.float32)

        # Connect right away if a gamepad is already plugged in
        for index in range(sdl_controller.get_count()):
            if sdl_controller.is_controller(index):
                self._connect(index)
                break

    def _connect(self, index: int) -> None:
        self.controller = sdl_controller.Controller(index)
        print(f" Controller connected: {self.controller.name}")

    @staticmethod
    def apply_deadzone(val: float, threshold: float = 0.1) -> float:
        if abs(val) < threshold:
            return 0.00
        return (val - np.sign(val) * threshold) / (1.0 - threshold)

    def update(self) -> None:
        """Handle (dis)connection events and read the sticks."""
        for event in pygame.event.get():
            # Auto-reconnect: take a new gamepad only if none is active
            if event.type == pygame.CONTROLLERDEVICEADDED and self.controller is None:
                self._connect(event.device_index)
            elif (
                event.type == pygame.CONTROLLERDEVICEREMOVED
                and self.controller is not None
                and not self.controller.attached()
            ):
                print(" Controller disconnected! Waiting for re-connection...")
                self.controller = None
                self.reset()  # zero the command if the connection is lost

        if self.controller is not None:
            # Axes are integers in [-32768, 32767]; stick up/left is negative.
            # Robot frame: x towards the head, y to the left, yaw rate counter-clockwise.
            def axis(a: int) -> float:
                return float(np.clip(self.controller.get_axis(a) / 32767.0, -1.0, 1.0))

            vel_x = -axis(pygame.CONTROLLER_AXIS_LEFTY)
            vel_y = -axis(pygame.CONTROLLER_AXIS_LEFTX)
            vel_w = -axis(pygame.CONTROLLER_AXIS_RIGHTX)

            self._input_vector[0] = self.apply_deadzone(vel_x)
            self._input_vector[1] = self.apply_deadzone(vel_y)
            self._input_vector[2] = self.apply_deadzone(vel_w)

    def read(self) -> Input:
        vx, vy, w = (float(v) for v in self._input_vector)
        return Input(vx, vy, w)

    def reset(self) -> None:
        self._input_vector = np.zeros(3, dtype=np.float32)
