# Self-contained image for deployment to Pascal-class GPUs (e.g. Titan X),
# without gsplat. gsplat's kernels use cooperative_groups features gated
# behind __CUDA_ARCH__ >= 700 (Volta+), so it hard-fails to compile for
# sm_61/sm_52 - see Models/mega-sam/LINUX_SETUP_CHANGELOG.md. MegaSaM's own
# CUDA code (droid_backends/lietorch_backends) doesn't use cooperative_groups
# and compiles fine for sm_61 (base/setup.py has compute_61 enabled), so this
# variant just drops the gsplat install and keeps everything else from
# Dockerfile.standalone.
#
# Build (context must be the repo root, so Models/mega-sam is in scope; see
# the repo-root .dockerignore for what's actually sent to the daemon):
#   docker build -f .devcontainer/Dockerfile.standalone.pascal -t msc2-image:pascal .
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

WORKDIR /workspace

RUN rm -f /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh
ENV PATH=$CONDA_DIR/envs/Msc2/bin:$CONDA_DIR/bin:$PATH

COPY .devcontainer/environment_linux.yml /tmp/environment_linux.yml
RUN conda env create -f /tmp/environment_linux.yml && conda clean -afy

RUN conda run -n Msc2 python -m ipykernel install --user --name Msc2 --display-name Msc2

# Bake in MegaSaM: its own conda env (CUDA 11.8/torch 2.0.1), compiled
# droid_backends/lietorch_backends extensions, and checkpoints/demo data.
# setup_linux_env.sh is idempotent and skips anything already present in the
# copied tree (checkpoints, demo data) - it only needs to create the env and
# (re)compile the CUDA extensions against base/setup.py's gencode list,
# which now includes compute_61/sm_61 for Pascal.
COPY Models/mega-sam /workspace/Models/mega-sam
RUN bash /workspace/Models/mega-sam/setup_linux_env.sh

CMD ["/bin/bash"]
