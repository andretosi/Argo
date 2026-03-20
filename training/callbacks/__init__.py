"""Callback exports for training.

Active curriculum callbacks live in ``curriculum_*`` modules.
Older callbacks are still exported from ``legacy_*`` modules for compatibility.
"""

from training.callbacks.curriculum_base import BaseCurriculumCallback
from training.callbacks.curriculum_competence import CompetenceTrackerCallback
from training.callbacks.curriculum_force import ForceCurriculumCallback
from training.callbacks.curriculum_friction import FrictionCurriculumCallback
from training.callbacks.curriculum_motor_noise import MotorNoiseCurriculumCallback
from training.callbacks.curriculum_sensor_noise import SensorNoiseCurriculumCallback
from training.callbacks.curriculum_terrain import TerrainCurriculumCallbackV2
from training.callbacks.legacy_curriculum import CurriculumCallback, CurriculumStage
from training.callbacks.legacy_perturbation import PerturbationCallback
from training.callbacks.legacy_terrain import (
    CurriculumTerrainCallback,
    TerrainChangeCallback,
)

# Readable aliases for the active curriculum stack. The underlying class names
# stay unchanged so older saved YAML snapshots keep loading correctly.
CompetenceCurriculumCallback = CompetenceTrackerCallback
TerrainCurriculumCallback = TerrainCurriculumCallbackV2

__all__ = [
    "BaseCurriculumCallback",
    "CompetenceTrackerCallback",
    "CompetenceCurriculumCallback",
    "TerrainCurriculumCallbackV2",
    "TerrainCurriculumCallback",
    "ForceCurriculumCallback",
    "FrictionCurriculumCallback",
    "MotorNoiseCurriculumCallback",
    "SensorNoiseCurriculumCallback",
    "TerrainChangeCallback",
    "CurriculumTerrainCallback",
    "PerturbationCallback",
    "CurriculumCallback",
    "CurriculumStage",
]
