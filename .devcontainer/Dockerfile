FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

WORKDIR /workspace

RUN rm -f /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*