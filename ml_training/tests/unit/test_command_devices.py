"""Command devices: registry, fixed/random validation, keyboard (on a pseudo-terminal) and
joystick (with a fake gamepad)."""

import os
import pty
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # pygame without a display
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from commands import COMMANDS, make_command
from commands.fixed_controller import FixedController
from commands.random_controller import RandomController


class RegistryTestCase(unittest.TestCase):
    def test_every_device_builds_from_its_params(self) -> None:
        for name in ("fixed", "random", "joystick"):
            with self.subTest(device=name):
                device = COMMANDS[name](COMMANDS[name].Params())
                device.update()
                self.assertEqual(len(device.read().as_array), 3)
        keyboard = COMMANDS["keyboard"](COMMANDS["keyboard"].Params())  # no update: no terminal
        self.assertEqual(tuple(keyboard.read().as_array), (0.0, 0.0, 0.0))

    def test_make_command_from_config_section(self) -> None:
        device = make_command({"fixed": {"mode": "walk"}})
        self.assertEqual(device.read().vx, 0.2)


class FixedAndRandomTestCase(unittest.TestCase):
    def test_fixed_modes(self) -> None:
        self.assertEqual(FixedController(FixedController.Params("still")).read().vx, 0.0)
        self.assertEqual(FixedController("walk").read().vx, 0.2)
        with self.assertRaises(ValueError):
            FixedController("run")

    def test_random_rejects_probabilities_not_summing_to_one(self) -> None:
        for field in ("p_base2walk", "p_still2walk", "p_walk2turn", "p_turn2walk"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                RandomController(RandomController.Params(**{field: 0.5}))

    def test_random_states_are_named_and_tags_checked(self) -> None:
        device = RandomController()
        names = {str(s) for s in (device.base_state, device.walk_state, device.turn_state)}
        names.add(str(device.still_state))
        self.assertEqual(len(names), 4)
        with self.assertRaises(ValueError):
            device.base_state._map_state("jump")
        with self.assertRaises(NotImplementedError):
            device.base_state.enter()


class KeyboardTestCase(unittest.TestCase):
    def setUp(self) -> None:
        from commands import keyboard_device

        self.master, slave = pty.openpty()
        self.stdin = os.fdopen(slave, "r")
        self.patch = mock.patch.object(keyboard_device.sys, "stdin", self.stdin)
        self.patch.start()
        self.keyboard = keyboard_device.Keyboard()

    def tearDown(self) -> None:
        del self.keyboard  # restores the terminal settings
        self.patch.stop()
        self.stdin.close()
        os.close(self.master)

    def press(self, keys: str) -> tuple[float, float, float]:
        os.write(self.master, keys.encode())
        time.sleep(0.02)
        command = self.keyboard.read()
        return command.vx, command.vy, command.w

    def test_disabled_until_the_first_update(self) -> None:
        self.assertEqual(self.press("w"), (0.0, 0.0, 0.0))

    def test_keys_map_to_the_robot_frame(self) -> None:
        self.keyboard.update()
        self.assertEqual(self.press("w"), (1, 0.0, 0.0))  # forward = head first
        self.keyboard.reset()
        self.assertEqual(self.press("s"), (-1, 0.0, 0.0))
        self.keyboard.reset()
        self.assertEqual(self.press("a"), (0.0, 1, 0.0))  # left = +y
        self.keyboard.reset()
        self.assertEqual(self.press("d"), (0.0, -1, 0.0))
        self.keyboard.reset()
        self.assertEqual(self.press("q"), (0.0, 0.0, 1))  # turn left = +w
        self.keyboard.reset()
        self.assertEqual(self.press("Ex"), (0.0, 0.0, -1))  # case-insensitive, others ignored

    def test_keys_combine_and_expire(self) -> None:
        self.keyboard.update()
        self.assertEqual(self.press("wq"), (1, 0.0, 1))
        time.sleep(self.keyboard.KEY_TIMEOUT + 0.05)
        self.assertEqual(self.press(""), (0.0, 0.0, 0.0))


class FakeGamepad:
    """Stands in for pygame._sdl2.controller.Controller."""

    name = "fake pad"

    def __init__(self, axes: dict[int, int]) -> None:
        self.axes = axes
        self.plugged = True

    def get_axis(self, axis: int) -> int:
        return self.axes.get(axis, 0)

    def attached(self) -> bool:
        return self.plugged


SDL = "pygame._sdl2.controller"


class JoystickTestCase(unittest.TestCase):
    def test_sticks_map_to_the_robot_frame_with_a_deadzone(self) -> None:
        import pygame

        from commands.joystick_device import Joystick

        pad = FakeGamepad(
            {
                pygame.CONTROLLER_AXIS_LEFTY: -32768,  # stick up: forward
                pygame.CONTROLLER_AXIS_LEFTX: 1000,  # inside the deadzone
                pygame.CONTROLLER_AXIS_RIGHTX: 32767,  # right: turn right (negative yaw rate)
            }
        )
        added = SimpleNamespace(type=pygame.CONTROLLERDEVICEADDED, device_index=0)
        removed = SimpleNamespace(type=pygame.CONTROLLERDEVICEREMOVED, instance_id=3)
        with (
            mock.patch(f"{SDL}.get_count", return_value=0),
            mock.patch(f"{SDL}.Controller", return_value=pad),
            mock.patch("pygame.event.get", side_effect=[[added], [removed], [removed]]),
        ):
            joystick = Joystick()
            self.assertIsNone(joystick.controller)
            joystick.update()  # the gamepad connects
            command = joystick.read()
            self.assertAlmostEqual(command.vx, 1.0)
            self.assertEqual(command.vy, 0.0)
            self.assertAlmostEqual(command.w, -1.0)
            joystick.update()  # another gamepad was removed: this one stays
            self.assertIs(joystick.controller, pad)
            pad.plugged = False
            joystick.update()  # this one was removed: the command drops to zero
            self.assertIsNone(joystick.controller)
            self.assertEqual(tuple(joystick.read().as_array), (0.0, 0.0, 0.0))

    def test_gamepad_present_at_start_and_mapping_file(self) -> None:
        from commands.joystick_device import Joystick

        with (
            tempfile.NamedTemporaryFile("w", suffix=".txt") as db,
            mock.patch.dict(os.environ),
            mock.patch(f"{SDL}.get_count", return_value=2),
            mock.patch(f"{SDL}.is_controller", side_effect=[False, True]),
            mock.patch(f"{SDL}.Controller", return_value=FakeGamepad({})) as open_pad,
        ):
            os.environ.pop("SDL_GAMECONTROLLERCONFIG_FILE", None)
            joystick = Joystick(Joystick.Params(db_path=db.name))
            self.assertEqual(os.environ["SDL_GAMECONTROLLERCONFIG_FILE"], db.name)
        self.assertIsNotNone(joystick.controller)
        open_pad.assert_called_once_with(1)  # the first index that is a gamepad


if __name__ == "__main__":
    unittest.main()
