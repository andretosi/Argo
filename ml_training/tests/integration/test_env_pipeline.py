"""The MuJoCo model loads as expected, and the policy gets correct observations and actions.

Every value the environment reports is compared with an independent reading from MuJoCo:
qpos/qvel looked up by joint name, the XML's own base sensors, and the body rotation matrix.
"""

import unittest

import mujoco
import numpy as np

from commands import FixedController, RandomController
from spotmicro.env import SpotmicroEnv
from training.rewards import make_reward

# MuJoCo body frame -> robot frame (x towards the head, y left): a half turn about z
FLIP_XY = np.diag([-1.0, -1.0, 1.0])


def make_env(command: object = None, reward: str = "robust_walking") -> SpotmicroEnv:
    return SpotmicroEnv(command or RandomController(), make_reward({reward: {}}))


def normalized(env: SpotmicroEnv, positions: np.ndarray) -> np.ndarray:
    lo = np.array([j.limits[0] for j in env.agent.motor_joints])
    hi = np.array([j.limits[1] for j in env.agent.motor_joints])
    return 2 * (positions - lo) / (hi - lo) - 1


class TestModelLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_env(FixedController("still"), "standing")
        self.m = self.env.sim.model

    def tearDown(self) -> None:
        self.env.close()

    def test_dimensions(self) -> None:
        self.assertEqual((self.m.nq, self.m.nv, self.m.nu), (19, 18, 12))
        self.assertEqual(self.m.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, 1), "base_link")

    def test_each_joint_drives_its_own_actuator(self) -> None:
        joints = self.env.agent.motor_joints
        self.assertEqual(len(joints), 12)
        self.assertEqual(len({j.id for j in joints}), 12)
        for j in joints:
            with self.subTest(joint=j.name):
                self.assertEqual(self.m.actuator_trnid[j.id, 0], j.link_id)
                name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, j.id)
                self.assertEqual(name, f"{j.name}_ctrl")
                np.testing.assert_allclose(self.m.jnt_range[j.link_id], j.limits)
                np.testing.assert_allclose(self.m.actuator_ctrlrange[j.id], j.limits)
                self.assertTrue(self.m.actuator_forcelimited[j.id])

    def test_physical_mass_and_inertia(self) -> None:
        mass = self.m.body_mass[1:].sum()
        self.assertGreater(mass, 2.0)
        self.assertLess(mass, 3.0)
        # A link of ~0.1 kg and ~0.1 m has an inertia of ~1e-4 kg m^2, not 0.05 placeholders
        self.assertLess(self.m.body_inertia[2:].max(), 0.005)

    def test_stands_still_at_homing(self) -> None:
        env, d = self.env, self.env.sim.data
        env.reset(seed=0)
        for _ in range(180):
            env.step(env.agent.default_actions.astype(np.float32))
        self.assertLess(np.abs(d.qvel).max(), 0.05)
        self.assertLess(np.abs(env.agent.state.roll_pitch_yaw[:2]).max(), 0.1)
        # The feet are the toes, on the ground
        np.testing.assert_allclose(env.agent.get_feet_positions()[:, 2], 0.02, atol=0.005)
        self.assertAlmostEqual(env.agent.get_body_to_feet_height_projected(), 0.205, delta=0.02)
        # Holding the robot up takes torque, and the state reports it
        efforts = env.agent.state.joint_efforts
        np.testing.assert_allclose(
            efforts, [d.actuator_force[j.id] for j in env.agent.motor_joints]
        )
        self.assertGreater(np.abs(efforts).max(), 0.1)

    def test_friction_reaches_the_contacts(self) -> None:
        env, d = self.env, self.env.sim.data
        env.reset(seed=0)
        for _ in range(30):  # land on the feet
            env.step(env.agent.default_actions.astype(np.float32))
        for friction in (0.3, 1.4):
            env.sim.set_friction(friction)
            env.step(env.agent.default_actions.astype(np.float32))
            self.assertGreater(d.ncon, 0)
            np.testing.assert_allclose(d.contact.friction[: d.ncon, 0], friction)

    def test_timestep_follows_sim_frequency(self) -> None:
        env, d = self.env, self.env.sim.data
        env.reset(seed=0)
        t0 = d.time
        env.step(env.agent.default_actions.astype(np.float32))
        self.assertAlmostEqual(d.time - t0, 1 / env.control_frequency, places=9)


class TestActions(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_env(FixedController("still"), "standing")
        self.joints = self.env.agent.motor_joints

    def tearDown(self) -> None:
        self.env.close()

    def test_action_reaches_the_right_actuator(self) -> None:
        a = np.random.default_rng(0).uniform(-1, 1, 12).astype(np.float32)
        self.env.agent.apply_action(a)
        ctrl = [self.env.sim.data.ctrl[j.id] for j in self.joints]
        expected = [j.from_action_to_position(a[i]) for i, j in enumerate(self.joints)]
        np.testing.assert_allclose(ctrl, expected)

    def test_action_range_maps_to_joint_limits(self) -> None:
        for j in self.joints:
            self.assertAlmostEqual(j.from_action_to_position(-1), j.limits[0])
            self.assertAlmostEqual(j.from_action_to_position(1), j.limits[1])

    def test_default_actions_are_the_normalized_homing_pose(self) -> None:
        agent = self.env.agent
        np.testing.assert_allclose(
            agent.default_actions, normalized(self.env, agent.homing_positions)
        )
        back = [
            j.from_action_to_position(agent.default_actions[i]) for i, j in enumerate(self.joints)
        ]
        np.testing.assert_allclose(back, agent.homing_positions)

    def test_action_map_is_identity_outside_the_deadzone(self) -> None:
        # Away from the homing deadzone, the normalized position of a target is (close to) its action
        for j in self.joints:
            target = j.homing_position + 0.25 * (j.limits[1] - j.homing_position)
            a = (2 * target - j.limits[0] - j.limits[1]) / (j.limits[1] - j.limits[0])
            self.assertNotAlmostEqual(j.from_action_to_position(a), j.homing_position, places=3)


class TestObservation(unittest.TestCase):
    def test_observation_matches_mujoco(self) -> None:
        env = make_env()
        m, d, agent = env.sim.model, env.sim.data, env.agent
        joints = agent.motor_joints
        qadr = [m.jnt_qposadr[j.link_id] for j in joints]
        vadr = [m.jnt_dofadr[j.link_id] for j in joints]

        def sensor(name: str) -> np.ndarray:
            adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)]
            return d.sensordata[adr : adr + 3].copy()

        rng = np.random.default_rng(1)
        env.reset(seed=1)
        positions, velocities, actions = [], [], []
        for t in range(200):
            a = np.clip(0.3 * rng.uniform(-1, 1, 12) + agent.default_actions, -1, 1)
            a = a.astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(a)
            mujoco.mj_forward(m, d)  # fresh sensors; the state itself does not change
            r_robot = d.xmat[1].reshape(3, 3) @ FLIP_XY
            q, qd = d.qpos[qadr].copy(), d.qvel[vadr].copy()
            positions.append(q)
            velocities.append(qd)
            actions.append(a)

            np.testing.assert_allclose(obs[0:3], r_robot.T @ [0, 0, -1], atol=1e-5)
            height = (d.qpos[2] - env.target_body_to_feet_height) / agent.max_norm_height
            self.assertAlmostEqual(obs[3], height, places=5)
            lin = FLIP_XY @ sensor("base_link_site_vel")  # velocimeter: body frame
            np.testing.assert_allclose(obs[4:7], lin / agent.max_linear_velocity, atol=1e-5)
            ang = r_robot.T @ sensor("base_link_site_angvel")  # world frame
            np.testing.assert_allclose(obs[7:10], ang / agent.max_angular_velocity, atol=1e-5)
            np.testing.assert_allclose(obs[10:22], normalized(env, q), atol=1e-5)
            np.testing.assert_allclose(
                obs[22:34], np.tanh(qd / agent.max_joint_velocity), atol=1e-5
            )
            if len(positions) >= 5:
                np.testing.assert_allclose(obs[34:46], normalized(env, positions[-2]), atol=1e-5)
                np.testing.assert_allclose(
                    obs[46:58], np.tanh(velocities[-2] / agent.max_joint_velocity), atol=1e-5
                )
                np.testing.assert_allclose(obs[58:70], normalized(env, positions[-5]), atol=1e-5)
                np.testing.assert_allclose(
                    obs[70:82], np.tanh(velocities[-5] / agent.max_joint_velocity), atol=1e-5
                )
            np.testing.assert_allclose(obs[82:94], a)
            np.testing.assert_allclose(obs[94:97], agent.command_input.as_array, atol=1e-6)
            self.assertEqual(obs.dtype, np.float32)
            self.assertLess(np.abs(obs).max(), 1.5)
            if terminated or truncated:
                env.reset()
                positions, velocities, actions = [], [], []
        env.close()


class TestRobotFrame(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_env(FixedController("still"), "standing")

    def tearDown(self) -> None:
        self.env.close()

    def _settle(self) -> None:
        self.env.reset(seed=0)
        for _ in range(30):
            self.env.step(self.env.agent.default_actions.astype(np.float32))

    def _head_direction_world(self) -> np.ndarray:
        m, d = self.env.sim.model, self.env.sim.data
        joints = self.env.agent.motor_joints
        front = [m.jnt_bodyid[j.link_id] for j in joints if j.frontback == "front"]
        rear = [m.jnt_bodyid[j.link_id] for j in joints if j.frontback == "rear"]
        v = d.xpos[front].mean(axis=0) - d.xpos[rear].mean(axis=0)
        v[2] = 0
        return v / np.linalg.norm(v)

    def test_spawn_heading_is_zero(self) -> None:
        self._settle()
        yaw = self.env.agent.state.roll_pitch_yaw[2]
        self.assertLess(abs(yaw), 0.05)
        np.testing.assert_allclose(self._head_direction_world()[:2], [1, 0], atol=0.05)

    def test_moving_head_first_is_positive_vx(self) -> None:
        self._settle()
        head = self._head_direction_world()
        for _ in range(10):
            self.env.sim.apply_external_force(30 * head)
            self.env.step(self.env.agent.default_actions.astype(np.float32))
        vx, vy, _ = self.env.agent.state.linear_velocity
        self.assertGreater(vx, 0.05)
        self.assertLess(abs(vy), abs(vx))

    def test_left_is_positive_vy(self) -> None:
        self._settle()
        head = self._head_direction_world()
        left = np.cross([0, 0, 1], head)
        for _ in range(10):
            self.env.sim.apply_external_force(30 * left)
            self.env.step(self.env.agent.default_actions.astype(np.float32))
        _, vy, _ = self.env.agent.state.linear_velocity
        self.assertGreater(vy, 0.05)


class TestPPOGetsTheEnvData(unittest.TestCase):
    def test_rollout_buffer(self) -> None:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        check_env(make_env(FixedController("still"), "standing"), warn=False)

        log: list[tuple[str, np.ndarray, tuple | None]] = []

        class Recorder(SpotmicroEnv):
            def reset(self, **kwargs: object) -> tuple:
                obs, info = super().reset(**kwargs)
                log.append(("reset", obs.copy(), None))
                return obs, info

            def step(self, action: np.ndarray) -> tuple:
                obs, r, te, tr, info = super().step(action)
                log.append(("step", obs.copy(), (np.array(action), r, te or tr)))
                return obs, r, te, tr, info

        env = Recorder(RandomController(), make_reward({"robust_walking": {}}))
        model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0, verbose=0)
        model.learn(64)
        buf = model.rollout_buffer
        # The obs the policy acted on: the reset obs, then each step's (after an episode end,
        # the next reset's)
        seen = [o for kind, o, extra in log if kind == "reset" or not extra[2]][:64]
        np.testing.assert_allclose(buf.observations.reshape(64, 97), seen)
        steps = [extra for kind, _, extra in log if kind == "step"][:64]
        np.testing.assert_allclose(
            [s[0] for s in steps], np.clip(buf.actions.reshape(64, 12), -1, 1)
        )
        np.testing.assert_allclose([s[1] for s in steps], buf.rewards.reshape(64), atol=1e-5)
        env.close()


if __name__ == "__main__":
    unittest.main()
