"""
Training Callbacks
==================

Custom callbacks for Stable-Baselines3 training.

Available callbacks:
- TerrainChangeCallback: Changes terrain every N episodes
- CurriculumTerrainCallback: Gradually increases terrain difficulty
"""

from training.callbacks.terrain_callback import (
    TerrainChangeCallback,
    CurriculumTerrainCallback,
)

__all__ = [
    "TerrainChangeCallback",
    "CurriculumTerrainCallback",
]
