# Development image for ml_training.
# Only the dependencies are baked in: the source code is mounted at /workspace
# at runtime (see run_container.sh), so edits on the host show up immediately.

FROM python:3.12-slim

# Build tools for packages without wheels (pyfastnoiselite) and the
# OpenGL/X11 libraries the MuJoCo viewer needs.
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

# PyTorch: the CPU build by default. MuJoCo and this small policy run on the CPU, and the
# CUDA build brings the Triton compiler, whose bundled LLVM collides with the LLVM of the
# software OpenGL the MuJoCo viewer uses in the container: the process segfaults.
# For an NVIDIA GPU anyway: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url "$TORCH_INDEX" \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Run as a user with the host's UID/GID so files created in the mounted
# folder (logs, checkpoints, plots) are owned by you, not root.
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid ${GROUP_ID} dev \
    && useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --shell /bin/bash dev
USER dev

# GIT_PYTHON_REFRESH: there is no git executable here; keeps GitPython (used by MLflow)
# from printing a long warning about it.
# Top-level modules (spotmicro, commands, training, ...) import from anywhere in the container.
ENV PYTHONPATH=/workspace \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=glfw \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /workspace
CMD ["bash"]
