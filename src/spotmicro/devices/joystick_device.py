import pygame
import os
import numpy as np
from spotmicro.devices.device import Device
from spotmicro.agent.input import Input

class Joystick(Device):
    def __init__(self, db_path: str = "gamecontrollerdb.txt"):
        pygame.init()
        pygame.joystick.init()
        
        # 1. Database Mapping Loading
        if os.path.exists(db_path):
            try:
                pygame.controller.load_mappings(db_path)
                print(f" Database loaded: {db_path}")
            except Exception as e:
                print(f" Errore database loading: {e}")

        # Initialize variables
        self.controller = None
        self._input_vector = np.array([0.00, 0.00, 0.00], dtype=np.float32)
        
        # Try to connect if a controller is already ploughed at the start
        self._check_for_existing_controller()

    def _check_for_existing_controller(self):
        """Loock for an aviable controller at start."""
        if pygame.joystick.get_count() > 0:
            for i in range(pygame.joystick.get_count()):
                if pygame.joystick.is_gamepad(i):
                    self.controller = pygame.controller.Controller(i)
                    print(f" Controller Found at start: {self.controller.get_name()}")
                    break

    def apply_deadzone(self, val, threshold=0.1):
        if abs(val) < threshold:
            return 0.00
        return (val - np.sign(val) * threshold) / (1.0 - threshold)

    def update(self) -> None:
        """Manage connection/disconnection events and uppdate the data."""
        for event in pygame.event.get():
            # AUTO-RECONNECT MANAGING
            if event.type == pygame.CONTROLLERDEVICEADDED:
                if self.controller is None: # If there is not an already active controller
                    self.controller = pygame.controller.Controller(event.device_index)
                    print(f" Connesso: {self.controller.get_name()}")

            if event.type == pygame.CONTROLLERDEVICEREMOVED:
                if self.controller and event.instance_id == self.controller.get_instance_id():
                    print(" Controller disconnected! Waiting for re-connection...")
                    self.controller = None
                    self.reset()        # Reset the motors if connection is lost

        # INPUT READ (Only if the contoller is active)
        if self.controller:
            # Use standard constants of pygame. controller for the axis
            # universal mappings grant that LEFTX remain the always the left stick
            vel_x = -self.controller.get_axis(pygame.CONTROLLER_AXIS_LEFTY)
            vel_y = -self.controller.get_axis(pygame.CONTROLLER_AXIS_LEFTX)
            vel_w = -self.controller.get_axis(pygame.CONTROLLER_AXIS_RIGHTX)

            self._input_vector[0] = self.apply_deadzone(vel_x)
            self._input_vector[1] = self.apply_deadzone(vel_y)
            self._input_vector[2] = self.apply_deadzone(vel_w)

    def read(self) -> Input:
        return self._input_vector

    def reset(self) -> None:
        self._input_vector = np.array([0.00, 0.00, 0.00], dtype=np.float32)