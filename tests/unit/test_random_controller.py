import unittest
from unittest.mock import patch
import numpy as np

from src.spotmicro.tools.config import Config
from src.spotmicro import (
    RandomController,
    BaseState,
)

class TestRandomController(unittest.TestCase):

    def setUp(self):
        self.cfg = Config()
        self.ctrl = RandomController(self.cfg)

    # ------------------------------------------------------------------
    # Constructor / configuration validation
    # ------------------------------------------------------------------

    def test_invalid_base_probabilities_raise(self):
        with self.assertRaises(ValueError):
            RandomController(
                self.cfg,
                p_base2still=0.5,
                p_base2walk=0.5,
                p_base2turn=0.2,
            )

    def test_invalid_walk_probabilities_raise(self):
        with self.assertRaises(ValueError):
            RandomController(
                self.cfg,
                p_walk2still=0.3,
                p_walk2turn=0.3,
            )

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def test_initial_state_is_base(self):
        self.assertIsInstance(self.ctrl._state, BaseState)

    # ------------------------------------------------------------------
    # Reset semantics
    # ------------------------------------------------------------------

    def test_reset_returns_to_base_state(self):
        self.ctrl._state = self.ctrl.walk_state
        self.ctrl.reset()
        self.assertIs(self.ctrl._state, self.ctrl.base_state)

    # ------------------------------------------------------------------
    # State transition mechanics
    # ------------------------------------------------------------------

    @patch("random.choices")
    def test_update_triggers_state_transition(self, mock_choices):
        # Force BaseState -> StillState
        mock_choices.return_value = ["still"]

        self.ctrl.update()

        self.assertIs(self.ctrl._state, self.ctrl.still_state)

    @patch("random.choices")
    def test_enter_called_on_transition(self, mock_choices):
        mock_choices.return_value = ["still"]

        with patch.object(self.ctrl.still_state, "enter") as enter_mock:
            self.ctrl.update()
            enter_mock.assert_called_once()

    # ------------------------------------------------------------------
    # State-specific behavior
    # ------------------------------------------------------------------

    @patch("numpy.random.normal", return_value=10)
    def test_still_state_outputs_zero_command(self, _):
        self.ctrl._state = self.ctrl.still_state
        self.ctrl._state.enter()

        inp = self.ctrl.read()
        self.assertEqual(inp.vx, 0.0)
        self.assertEqual(inp.vy, 0.0)
        self.assertEqual(inp.w, 0.0)
    
    #Helper function for the following test
    def fake_normal(mean, var):
        if isinstance(mean, (tuple, list, np.ndarray)):
            return np.array(mean)
        return mean

    @patch("numpy.random.normal", side_effect=fake_normal)
    def test_walk_state_has_zero_angular_velocity(self, _):
        self.ctrl._state = self.ctrl.walk_state
        self.ctrl._state.enter()

        inp = self.ctrl.read()
        self.assertEqual(inp.w, 0.0)

    @patch("numpy.random.normal", return_value=0.5)
    def test_turn_state_has_zero_lateral_velocity(self, _):
        self.ctrl._state = self.ctrl.turn_state
        self.ctrl._state.enter()

        inp = self.ctrl.read()
        self.assertEqual(inp.vy, 0.0)

    # ------------------------------------------------------------------
    # Remaining-steps logic
    # ------------------------------------------------------------------

    def test_state_does_not_transition_if_steps_remaining(self):
        state = self.ctrl.walk_state
        self.ctrl._state = state
        state.remaining_steps = 5

        next_state = state.update()

        self.assertIs(next_state, state)

    def test_state_transitions_when_steps_exhausted(self):
        state = self.ctrl.walk_state
        self.ctrl._state = state
        state.remaining_steps = 0

        with patch("random.choices", return_value=["still"]):
            next_state = state.update()
            self.assertIs(next_state, self.ctrl.still_state)
