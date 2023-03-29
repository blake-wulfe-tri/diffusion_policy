FROM nvidia/cuda:11.6.2-base-ubuntu20.04

ARG python=3.9
ENV PYTORCH_VERSION=1.12.1
ENV TORCHVISION_VERSION=0.13.1

ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Temporary fix for invalid GPG key see
# https://forums.developer.nvidia.com/t/gpg-error-http-developer-download-nvidia-com-compute-cuda-repos-ubuntu1804-x86-64/212904
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  libosmesa6-dev \
  libgl1-mesa-glx \
  libglfw3 \
  patchelf \
  libgtk2.0-dev \
  libjpeg-dev \
  libpng-dev \
  wget \
  tmux \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-dev \
  && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
  python3 get-pip.py && \
        rm get-pip.py
# The above appears to install pip into /usr/local/, but some tools expect /usr/bin/.
RUN ln -sf /usr/local/bin/pip /usr/bin/pip
RUN ln -sf /usr/local/bin/pip3 /usr/bin/pip3

# Install Pytorch
RUN pip install --no-cache-dir \
  torch==${PYTORCH_VERSION} \
  torchvision==${TORCHVISION_VERSION} \
  -f https://download.pytorch.org/whl/${PYTORCH_VERSION/*+/}/torch_stable.html

ARG WORKSPACE=/home/diffusion_policy
WORKDIR ${WORKSPACE}

# Need to add miniconda to the path manually.
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
# Install miniconda and mamba.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
  && mkdir /root/.conda \
  && bash ./Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -b \
  && rm Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
  && conda install mamba -n base -c conda-forge \
  && conda init bash

COPY conda_environment.yaml .
RUN mamba env create -f conda_environment.yaml

# # Settings for S3
# RUN aws configure set default.s3.max_concurrent_requests 100 && \
#   aws configure set default.s3.max_queue_size 10000

# TODO
# 1. install aws cli before setting the s3 settings


