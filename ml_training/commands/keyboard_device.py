# Assignee: virgina


import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

from commands.device import Device, DeviceConfig, Input


class Keyboard(Device):
    name = "keyboard"

    # --- Settings ---
    VAL_SPEED = 1
    VAL_TURN = 1  # to be tuned to match the robot's turning speed
    # How long (in seconds) a key stays active after it was pressed
    KEY_TIMEOUT = 0.1

    @dataclass
    class Params(DeviceConfig):
        """No parameters."""

    def __init__(self, params: "Keyboard.Params | None" = None) -> None:
        super().__init__()
        self._enabled = False
        self._orig_settings = None

        # --- Internal state ---
        # When each axis last received a key press
        self.last_time_x = 0.0
        self.last_time_y = 0.0
        self.last_time_w = 0.0

        # Current values to send
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_w = 0.0

    def update(self) -> None:
        if not self._enabled:
            self._orig_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._enabled = True

    def read(self) -> Input:
        now = time.time()

        if not self._enabled:
            return Input(0.0, 0.0, 0.0)

        # 1. Read every key typed since the last call, to catch quick combinations. Straight
        # from the file descriptor: sys.stdin's own buffer would hide keys from select()
        fd = sys.stdin.fileno()
        typed = ""
        while select.select([fd], [], [], 0)[0]:
            chunk = os.read(fd, 1024)
            if not chunk:
                break  # end of input
            typed += chunk.decode(errors="ignore")

        for char in typed.lower():
            # Update the axis value and its timestamp
            if char == "w":  # forward (head first)
                self.current_x = self.VAL_SPEED
                self.last_time_x = now
            elif char == "s":  # backward
                self.current_x = -self.VAL_SPEED
                self.last_time_x = now
            elif char == "q":  # turn left
                self.current_w = self.VAL_TURN
                self.last_time_w = now
            elif char == "e":  # turn right
                self.current_w = -self.VAL_TURN
                self.last_time_w = now
            elif char == "a":  # step left
                self.current_y = self.VAL_SPEED
                self.last_time_y = now
            elif char == "d":  # step right
                self.current_y = -self.VAL_SPEED
                self.last_time_y = now

        # 2. Decay: zero an axis whose key has not been seen for KEY_TIMEOUT

        if now - self.last_time_x > self.KEY_TIMEOUT:
            self.current_x = 0.0

        if now - self.last_time_y > self.KEY_TIMEOUT:
            self.current_y = 0.0

        if now - self.last_time_w > self.KEY_TIMEOUT:
            self.current_w = 0.0

        # Robot frame: x towards the head, y to the left, yaw rate positive counter-clockwise
        vx = max(min(self.current_x, 1.0), -1.0)
        vy = max(min(self.current_y, 1.0), -1.0)
        vw = max(min(self.current_w, 1.0), -1.0)

        return Input(vx, vy, vw)

    def reset(self) -> None:
        self.last_time_x = 0.0
        self.last_time_y = 0.0
        self.last_time_w = 0.0

        # Current values to send
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_w = 0.0

    def __del__(self) -> None:
        if self._enabled and self._orig_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_settings)
