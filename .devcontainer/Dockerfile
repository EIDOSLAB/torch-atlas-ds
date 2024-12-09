FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV USERNAME=user
ARG USER_UID
ARG USER_GID=$USER_UID

RUN apt update \
    && apt install -y git

RUN pip install poetry

RUN mkdir -p /workspaces/torch-atlas-ds
WORKDIR /workspaces/torch-atlas-ds

COPY pyproject.toml ./pyproject.toml
COPY poetry.lock ./poetry.lock

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction \
    && rm -rf /root/.cache/pypoetry

ENV CUDA_HOME=/usr/local/cuda
ENV HF_HOME=/workspaces/data/hf-cache/

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
RUN poetry config virtualenvs.create false