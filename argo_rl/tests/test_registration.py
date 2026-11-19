# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

import argo_rl.tasks  # noqa: F401


def test_task_registrations():
    """The project tasks must expose valid environment and agent entry points."""
    expected = {
        "ArgoRl-Balance-Cartpole": {
            "entry_point": "argo_rl.tasks.balance.config.cartpole.env:BalanceEnv",
            "env_cfg_entry_point": "argo_rl.tasks.balance.config.cartpole.env_cfg:BalanceEnvCfg",
        },
        "ArgoRl-Velocity-Flat-UnitreeGo2": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "argo_rl.tasks.velocity.config.go2.flat_env_cfg:UnitreeGo2FlatEnvCfg",
            "default_agent": "rsl_rl",
        },
        "ArgoRl-Velocity-Rough-UnitreeGo2": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "argo_rl.tasks.velocity.config.go2.rough_env_cfg:UnitreeGo2RoughEnvCfg",
            "default_agent": "rsl_rl",
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]
