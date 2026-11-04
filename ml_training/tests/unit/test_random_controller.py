import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from commands.random_controller import (
    BaseState,
    RandomController,
)


def fake_normal(
    mean: float | tuple[float, ...] | list[float] | np.ndarray,
    var: float | tuple[float, ...] | list[float] | np.ndarray,
) -> float | np.ndarray:
    """Deterministic stand-in for np.random.normal: returns the mean."""
    if isinstance(mean, (tuple, list, np.ndarray)):
        return np.array(mean)
    return mean


class TestRandomController(unittest.TestCase):
    def setUp(self) -> None:
        self.ctrl = RandomController()

    # ------------------------------------------------------------------
    # Constructor / configuration validation
    # ------------------------------------------------------------------

    def test_invalid_base_probabilities_raise(self) -> None:
        with self.assertRaises(ValueError):
            RandomController(
                RandomController.Params(p_base2still=0.5, p_base2walk=0.5, p_base2turn=0.2)
            )

    def test_invalid_walk_probabilities_raise(self) -> None:
        with self.assertRaises(ValueError):
            RandomController(RandomController.Params(p_walk2still=0.3, p_walk2turn=0.3))

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def test_initial_state_is_base(self) -> None:
        self.assertIsInstance(self.ctrl._state, BaseState)

    # ------------------------------------------------------------------
    # Reset semantics
    # ------------------------------------------------------------------

    def test_reset_returns_to_base_state(self) -> None:
        self.ctrl._state = self.ctrl.walk_state
        self.ctrl.reset()
        self.assertIs(self.ctrl._state, self.ctrl.base_state)

    # ------------------------------------------------------------------
    # State transition mechanics
    # ------------------------------------------------------------------

    @patch("random.choices")
    def test_update_triggers_state_transition(self, mock_choices: MagicMock) -> None:
        # Force BaseState -> StillState
        mock_choices.return_value = ["still"]

        self.ctrl.update()

        self.assertIs(self.ctrl._state, self.ctrl.still_state)

    @patch("random.choices")
    def test_enter_called_on_transition(self, mock_choices: MagicMock) -> None:
        mock_choices.return_value = ["still"]

        with patch.object(self.ctrl.still_state, "enter") as enter_mock:
            self.ctrl.update()
            enter_mock.assert_called_once()

    # ------------------------------------------------------------------
    # State-specific behavior
    # ------------------------------------------------------------------

    @patch("numpy.random.normal", return_value=10)
    def test_still_state_outputs_zero_command(self, _: MagicMock) -> None:
        self.ctrl._state = self.ctrl.still_state
        self.ctrl._state.enter()

        inp = self.ctrl.read()
        self.assertEqual(inp.vx, 0.0)
        self.assertEqual(inp.vy, 0.0)
        self.assertEqual(inp.w, 0.0)

    @patch("numpy.random.normal", side_effect=fake_normal)
    def test_walk_state_has_zero_angular_velocity(self, _: MagicMock) -> None:
        self.ctrl._state = self.ctrl.walk_state
        self.ctrl._state.enter()

        inp = self.ctrl.read()
        self.assertEqual(inp.w, 0.0)

    @patch("numpy.random.normal", return_value=0.5)
    def test_turn_state_has_zero_lateral_velocity(self, _: MagicMock) -> None:
        self.ctrl._state = self.ctrl.turn_state
        self.ctrl._state.enter()

        inp = self.ctrl.read()
        self.assertEqual(inp.vy, 0.0)

    # ------------------------------------------------------------------
    # Remaining-steps logic
    # ------------------------------------------------------------------

    def test_state_does_not_transition_if_steps_remaining(self) -> None:
        state = self.ctrl.walk_state
        self.ctrl._state = state
        state.remaining_steps = 5

        next_state = state.update()

        self.assertIs(next_state, state)

    def test_state_transitions_when_steps_exhausted(self) -> None:
        state = self.ctrl.walk_state
        self.ctrl._state = state
        state.remaining_steps = 0

        with patch("random.choices", return_value=["still"]):
            next_state = state.update()
            self.assertIs(next_state, self.ctrl.still_state)
