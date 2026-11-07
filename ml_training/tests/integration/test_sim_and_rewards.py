"""Every reward on the real environment, the simulation's terrain and viewer paths (viewer
mocked: no display), and the environment's input checks."""

import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from commands import FixedController, RandomController
from commands.device import Device, Input
from spotmicro.agent import AgentParams, Joint
from spotmicro.env import EnvParams, SpotmicroEnv
from spotmicro.sim import MujocoSim
from spotmicro.terrain import Heightmap
from training.rewards import REWARDS, make_reward
from training.rewards.base import Reward

MODEL = Path(__file__).resolve().parents[2] / "robot_models" / "spotmicroai.mujoco.xml"


class RewardsTestCase(unittest.TestCase):
    def test_every_reward_gives_finite_values_early_and_late_in_training(self) -> None:
        rng = np.random.default_rng(0)
        for name in REWARDS:
            for steps_done in (0, 100_000_000):  # before and after every fade-in
                with self.subTest(reward=name, steps_done=steps_done):
                    env = SpotmicroEnv(RandomController(), make_reward({name: {}}))
                    env.set_num_steps(steps_done)
                    env.reset(seed=0)
                    for _ in range(150):
                        noise = 0.3 * rng.uniform(-1, 1, 12)
                        action = np.clip(env.agent.default_actions + noise, -1, 1)
                        _, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
                        self.assertTrue(np.isfinite(reward))
                        self.assertTrue(
                            all(np.isfinite(v) for v in env.episode_reward_info[-1].values())
                        )
                        if terminated or truncated:
                            env.reset()
                    env.close()

    def test_rewards_must_return_a_dict_of_components(self) -> None:
        class Broken(Reward):
            def compute(self, env: SpotmicroEnv, action: np.ndarray) -> list:
                return [1.0]

        env = SpotmicroEnv(FixedController("still"), Broken())
        env.reset(seed=0)
        with self.assertRaises(TypeError):
            env.step(env.agent.default_actions.astype(np.float32))
        env.close()


class EnvChecksTestCase(unittest.TestCase):
    def test_invalid_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple"):
            SpotmicroEnv(
                FixedController(), make_reward({"standing": {}}), EnvParams(sim_frequency=250)
            )
        with self.assertRaisesRegex(ValueError, "at least 5"):
            SpotmicroEnv(
                FixedController(),
                make_reward({"standing": {}}),
                agent_params=AgentParams(joint_history_maxlen=3),
            )
        with self.assertRaisesRegex(ValueError, "invalid limits"):
            Joint("front_left_leg", 0, 0, "leg", (1.0, -1.0), 6.5, *[0.1] * 9)

    def test_commands_outside_the_unit_range_are_rejected(self) -> None:
        class TooFast(Device):
            def update(self) -> None: ...
            def read(self) -> Input:
                return Input(vx=2.0)

            def reset(self) -> None: ...

        env = SpotmicroEnv(TooFast(), make_reward({"robust_walking": {}}))
        with self.assertRaisesRegex(ValueError, "normalized"):
            env.reset(seed=0)
        env.close()


class SimTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = MujocoSim()
        self.sim.load_model(str(MODEL), np.zeros(3), np.array([1.0, 0, 0, 0]))

    def test_terrain_spawn_clip_and_remove(self) -> None:
        heights = Heightmap.from_noise(x=64, y=64, z_max=2.0).data  # random seed
        handle = self.sim.spawn_terrain(heights, (0.02, 0.02, 1.0), (0, 0, 0))
        m = self.sim.model
        elevation = m.hfield_size[handle][2]
        self.assertAlmostEqual(m.hfield_data.max(), 1.0, places=5)  # clipped to the elevation
        self.assertAlmostEqual(m.geom_pos[0][2], -elevation / 2)
        self.sim.remove_terrain(handle)
        self.assertEqual(m.hfield_data.max(), 0.0)
        np.testing.assert_array_equal(m.geom_pos[0], [0, 0, 0])
        with self.assertRaises(TypeError):
            Heightmap([[0.0]])

    def test_helpers_and_unloaded_model(self) -> None:
        q = self.sim.quaternion_from_euler(0.1, -0.2, 0.3)
        np.testing.assert_allclose(self.sim.euler_from_quaternion(q), [0.1, -0.2, 0.3])
        self.assertAlmostEqual(np.linalg.det(self.sim.get_rotation_matrix(q)), 1.0)
        empty = MujocoSim()
        with self.assertRaisesRegex(RuntimeError, "load_model"):
            empty.spawn_terrain(np.zeros((4, 4)), (1, 1, 1), (0, 0, 0))
        empty.remove_terrain(0)  # nothing to do
        empty.close()

    def test_model_without_floor_or_terrain_is_rejected(self) -> None:
        xml = MODEL.read_text()
        with tempfile.TemporaryDirectory() as tmp:
            meshes = MODEL.parent / "meshes"
            no_floor = xml.replace('name="floor"', 'name="ground"').replace(
                'file="meshes/', f'file="{meshes}/'
            )
            path = Path(tmp) / "no_floor.xml"
            path.write_text(no_floor)
            with self.assertRaisesRegex(RuntimeError, "floor"):
                MujocoSim().load_model(str(path), np.zeros(3), np.array([1.0, 0, 0, 0]))
            no_hfield = (
                xml.replace('hfield="terrain"', 'hfield="flat"')
                .replace('<hfield name="terrain"', '<hfield name="flat"')
                .replace('file="meshes/', f'file="{meshes}/')
            )
            path.write_text(no_hfield)
            sim = MujocoSim()
            sim.load_model(str(path), np.zeros(3), np.array([1.0, 0, 0, 0]))
            with self.assertRaisesRegex(RuntimeError, "terrain"):
                sim.spawn_terrain(np.zeros((4, 4)), (1, 1, 1), (0, 0, 0))

    def test_viewer_is_synced_updated_and_waited_for(self) -> None:
        with mock.patch("mujoco.viewer.launch_passive") as launch:
            sim = MujocoSim(use_gui=True)
            sim.load_model(str(MODEL), np.zeros(3), np.array([1.0, 0, 0, 0]))
        viewer = launch.return_value
        sim.sync_viewer()
        handle = sim.spawn_terrain(np.zeros((8, 8)), (0.02, 0.02, 1.0), (0, 0, 0))
        viewer.update_hfield.assert_called_with(handle)
        self.assertGreaterEqual(viewer.sync.call_count, 2)

        # close() waits for the viewer's render thread to finish
        render = threading.Thread(target=time.sleep, args=(0.2,), name="_launch_internal")
        render.start()
        sim.close()
        viewer.close.assert_called_once()
        self.assertFalse(render.is_alive())
        sim.close()  # closing twice is harmless


class EpisodeEndTestCase(unittest.TestCase):
    def test_episode_endings_are_scored_by_the_reward(self) -> None:
        """The env decides how an episode ends; the reward's own parameters score it."""
        params = replace(EnvParams(), max_episode_len=5)
        reward = make_reward(
            {"standing": {"fall_penalty": -7.0, "tipping_penalty": -5.0, "survival_bonus": 11.0}}
        )
        env = SpotmicroEnv(FixedController(), reward, params)
        stand = env.agent.default_actions.astype(np.float32)
        ending = ("fall_penalty", "tipping_penalty", "survival_bonus")

        def last_step() -> tuple[float, bool, bool, dict[str, float]]:
            _, total, terminated, truncated, _ = env.step(stand)
            components = env.episode_reward_info[-1]
            self.assertAlmostEqual(total, sum(components.values()))
            return total, terminated, truncated, {k: components[k] for k in ending}

        env.reset(seed=0)
        for _ in range(4):
            _, terminated, truncated, end = last_step()
            self.assertFalse(terminated or truncated)
            self.assertEqual(end, dict.fromkeys(ending, 0.0))
        _, terminated, truncated, end = last_step()
        self.assertTrue(truncated and not terminated)
        self.assertEqual(env.termination, None)
        self.assertEqual(end, {**dict.fromkeys(ending, 0.0), "survival_bonus": 11.0})

        env.reset(seed=0)
        self.assertFalse(env.truncated)
        env.sim.data.qpos[2] = 1.0  # far above the allowed height
        _, terminated, _, end = last_step()
        self.assertTrue(terminated)
        self.assertEqual(env.termination, "fall")
        self.assertEqual(end, {**dict.fromkeys(ending, 0.0), "fall_penalty": -7.0})

        env.reset(seed=0)
        env.sim.data.qpos[3:7] = [np.cos(0.75), np.sin(0.75), 0.0, 0.0]  # rolled by 1.5 rad
        _, terminated, _, end = last_step()
        self.assertTrue(terminated)
        self.assertEqual(env.termination, "tipping")
        self.assertEqual(end, {**dict.fromkeys(ending, 0.0), "tipping_penalty": -5.0})
        env.close()


if __name__ == "__main__":
    unittest.main()
