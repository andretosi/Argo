# Development image for ml_training.
# Only the dependencies are baked in: the source code is mounted at /workspace
# at runtime (see run_container.sh), so edits on the host show up immediately.

FROM python:3.12-slim

# Build tools for packages without wheels (pybullet, pyfastnoiselite) and the
# OpenGL/X11 libraries the PyBullet and MuJoCo GUIs need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libegl1 \
        libglfw3 \
        libosmesa6 \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch comes in through stable-baselines3 as the default CUDA build, so the
# same image uses a GPU when one is available and falls back to CPU otherwise.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Run as a user with the host's UID/GID so files created in the mounted
# folder (logs, checkpoints, plots) are owned by you, not root.
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid ${GROUP_ID} dev \
    && useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --shell /bin/bash dev
USER dev

# Top-level modules (environments, robot, training, ...) import from anywhere in the container.
ENV PYTHONPATH=/workspace \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=glfw

WORKDIR /workspace
CMD ["bash"]
