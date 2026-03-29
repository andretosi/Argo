import unittest
import tempfile
import yaml
import os

from spotmicro.env.spotmicro_env_mujoco import SpotmicroEnv
from .data.mock_rw_fn import reward_function, RewardState
from spotmicro import RandomController
from spotmicro.tools.config import Config

#TODO: add more tests to explore other behaviours. what happens to overrides? pay special attention to thew "matrioska" classes,: Agent and RandomController. How do you override their parameters?
class TestConfigIntegration(unittest.TestCase):
    def test_config_persistence(self):
        # Load original config
        WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        TEST_DIR = "tests/integration"
        cfg_path = os.path.join(WORKSPACE_ROOT, TEST_DIR, "data/cfgPersistence.yaml")
        cfg = Config(cfg_path)

        # Initialize environment and device
        dev = RandomController(cfg)
        env = SpotmicroEnv(dev, cfg, reward_function, reward_state=RewardState())
        obs, _ = env.reset()

        # Take one step (optional for integration)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Save config to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp_file:
            temp_path = tmp_file.name
        cfg.save(temp_path)

        # Compare original and saved YAML files
        with open(cfg_path, "r") as f:
            original_yaml = yaml.safe_load(f)
        with open(temp_path, "r") as f:
            saved_yaml = yaml.safe_load(f)

        # Test that they are identical
        self.assertEqual(original_yaml, saved_yaml)

        # Clean up temp file
        os.remove(temp_path)

    def test_config_nested_overrides(self):
        WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        TEST_DIR = "tests/integration"
        src_cfg_path = os.path.join(WORKSPACE_ROOT, TEST_DIR, "data/cfgOverridesSRC.yaml")
        src_cfg = Config(src_cfg_path)

        #Initializing components with overrides
        dev = RandomController(src_cfg, p_base2still=0.1, p_base2walk=0.9, p_base2turn=0)
        env = SpotmicroEnv(dev, src_cfg, reward_function, RewardState(), max_height=0.6)

        obv, _ = env.reset()

        # Save config to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp_file:
            temp_path = tmp_file.name
        src_cfg.save(temp_path)

        # Compare target and saved YAML files
        dst_cfg_path = os.path.join(WORKSPACE_ROOT, TEST_DIR, "data/cfgOverridesDST.yaml")
        with open(dst_cfg_path, "r") as f:
            target_yaml = yaml.safe_load(f)
        with open(temp_path, "r") as f:
            saved_yaml = yaml.safe_load(f)

        # Test that they are identical
        self.assertEqual(target_yaml, saved_yaml)
        self.assertEqual(env.max_height, 0.6)
        self.assertEqual(env.agent.homing_pitch, 0.4)

        # Clean up temp file
        os.remove(temp_path)