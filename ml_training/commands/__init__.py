"""Command sources: where the velocity command the policy must follow comes from.

Each device has a ``Params`` dataclass; a run config picks one under ``command:``.
"""

from typing import Any

from commands.device import Device, DeviceConfig
from commands.fixed_controller import FixedController
from commands.random_controller import RandomController


def _keyboard() -> type[Device]:
    from commands.keyboard_device import Keyboard

    return Keyboard


def _joystick() -> type[Device]:
    from commands.joystick_device import Joystick

    return Joystick


class _Lazy:
    """Registry entry for a device whose module has heavy or terminal-specific imports."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    @property
    def Params(self) -> type:
        return self._loader().Params

    def __call__(self, params: Any) -> Device:
        return self._loader()(params)


# By each device's ``name``; keyboard and joystick are imported only when used
COMMANDS: dict[str, Any] = {
    RandomController.name: RandomController,  # training: random walk/turn/stand-still commands
    FixedController.name: FixedController,  # constant command: "still" or "walk" (0.2 m/s)
    "keyboard": _Lazy(_keyboard),  # drive a policy from the terminal
    "joystick": _Lazy(_joystick),  # drive a policy with a gamepad (pygame)
}


def make_command(section: dict[str, Any]) -> Device:
    """Build a command device from a one-key config section, e.g. ``{"random": {...}}``."""
    from training.config import build_component

    return build_component(section, COMMANDS, "command").make()


__all__ = ["COMMANDS", "Device", "DeviceConfig", "make_command"]
