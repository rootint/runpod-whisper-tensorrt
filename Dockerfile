# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags

# FROM runpod/base:0.6.2-cuda12.1.0
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends sudo ca-certificates git wget curl bash libgl1 libx11-6 software-properties-common libsndfile1 ffmpeg build-essential -y &&\
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# install_tensorrt.sh
RUN apt-get install -y build-essential ca-certificates ccache cmake gnupg2 wget curl gdb || sudo apt-get update && sudo apt-get install -y build-essential ca-certificates ccache cmake gnupg2 wget curl gdb
RUN apt-get -y install openmpi-bin libopenmpi-dev || sudo apt-get update && sudo apt-get -y install openmpi-bin libopenmpi-dev

RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install python3.10-dev python3.10-venv python3-pip -y --no-install-recommends && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* 

COPY builder/install_mpi4py.sh /install_mpi4py.sh
RUN bash /install_mpi4py.sh && rm /install_mpi4py.sh
RUN python3 -m pip install --no-cache-dir -U torch==2.1.2
RUN python3 -m pip install --no-cache-dir tensorrt_llm==0.8.0.dev2024012301 --extra-index-url https://pypi.nvidia.com


# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt
RUN python3 -m pip install git+https://github.com/rootint/whisper-tensorrt.git@3be97fd5ff54966f0c96b4f89d40b371ab661a78

COPY builder/cache /root/.cache

# Add src files (Worker Template)
ADD src .

CMD python3 -u /rp_handler.py
