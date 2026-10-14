# Training Examples

This directory contains example training scripts that demonstrate Fucina's features.

## Dynamic Terrain Training

The main example shows how to train a walking policy with terrain that changes during training:

```bash
# Basic training with PyBullet (fastest)
python train_with_terrain.py --engine pybullet

# Training with MuJoCo (same code, different physics!)
python train_with_terrain.py --engine mujoco

# With curriculum learning (gradually harder terrain)
python train_with_terrain.py --engine pybullet --terrain-type curriculum

# Debug with GUI
python train_with_terrain.py --engine pybullet --gui --total-steps 10000
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--engine` | Physics backend (`pybullet` or `mujoco`) | `pybullet` |
| `--total-steps` | Total training timesteps | 5,000,000 |
| `--terrain-change-episodes` | Change terrain every N episodes | 50 |
| `--terrain-type` | `fixed`, `random`, or `curriculum` | `random` |
| `--terrain-difficulty` | Max terrain height in meters | 0.3 |
| `--gui` | Show visualization | False |
| `--run-name` | Name for this training run | `terrain_walk` |

## Terrain Types

1. **Fixed**: Flat ground, no changes
2. **Random**: Random terrain every N episodes (same difficulty)
3. **Curriculum**: Starts easy (10% difficulty), gradually increases to 100%

## Output

Training creates:
```
runs/<run_name>_<engine>/
├── checkpoints/          # Model checkpoints
├── logs/                 # TensorBoard logs
└── ppo_<run_name>_final  # Final trained model
```

View training progress:
```bash
tensorboard --logdir runs/
```
