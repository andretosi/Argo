"""
Training Utilities
==================

Domain randomization and noise injection utilities for robust training.
All classes are @configurable for parameter saving/loading.

Modules:
- domain_randomization: Push perturbations, friction randomization
- noise: Sensor noise injection for sim-to-real transfer
"""

from training.utils.domain_randomization import (
    PushPerturbation,
    PushPerturbationConfig,
    FrictionRandomizer,
    FrictionConfig,
)

from training.utils.noise import (
    SensorNoise,
    NoiseConfig,
    CorrelatedNoise,
)

__all__ = [
    # Domain randomization
    "PushPerturbation",
    "PushPerturbationConfig",
    "FrictionRandomizer",
    "FrictionConfig",
    # Noise
    "SensorNoise",
    "NoiseConfig",
    "CorrelatedNoise",
]
