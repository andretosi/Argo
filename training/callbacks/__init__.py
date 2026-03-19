"""
Training Callbacks
==================

Custom callbacks for Stable-Baselines3 training.

Atomic Curriculum Callbacks (recommended):
- TerrainCurriculumCallbackV2: Terrain difficulty curriculum
- ForceCurriculumCallback: Push force perturbation curriculum
- FrictionCurriculumCallback: Friction randomization curriculum
- MotorNoiseCurriculumCallback: Motor/actuator noise curriculum
- SensorNoiseCurriculumCallback: Sensor observation noise curriculum

Legacy Callbacks:
- TerrainChangeCallback: Changes terrain every N episodes
- CurriculumTerrainCallback: Gradually increases terrain difficulty
- PerturbationCallback: Applies push forces and friction randomization
- CurriculumCallback: Unified curriculum (deprecated, use atomic callbacks)
"""

# Base class
from training.callbacks.base_curriculum import BaseCurriculumCallback

# Atomic curriculum callbacks (recommended)
from training.callbacks.terrain_curriculum import TerrainCurriculumCallbackV2
from training.callbacks.force_curriculum import ForceCurriculumCallback
from training.callbacks.friction_curriculum import FrictionCurriculumCallback
from training.callbacks.motor_noise_curriculum import MotorNoiseCurriculumCallback
from training.callbacks.sensor_noise_curriculum import SensorNoiseCurriculumCallback

# Legacy callbacks (for backwards compatibility)
from training.callbacks.terrain_callback import (
    TerrainChangeCallback,
    CurriculumTerrainCallback,
)
from training.callbacks.perturbation_callback import PerturbationCallback
from training.callbacks.curriculum_callback import CurriculumCallback, CurriculumStage

__all__ = [
    # Base
    "BaseCurriculumCallback",
    # Atomic curriculum callbacks (recommended)
    "TerrainCurriculumCallbackV2",
    "ForceCurriculumCallback",
    "FrictionCurriculumCallback",
    "MotorNoiseCurriculumCallback",
    "SensorNoiseCurriculumCallback",
    # Legacy
    "TerrainChangeCallback",
    "CurriculumTerrainCallback",
    "PerturbationCallback",
    "CurriculumCallback",
    "CurriculumStage",
]
