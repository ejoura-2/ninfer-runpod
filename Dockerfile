# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG NINFER_REF=a99407c63fc5bbd25d9fb597cbb8ab352bdb01ef

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        cmake \
        git \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git init \
    && git remote add origin https://github.com/Neroued/ninfer.git \
    && git fetch --depth 1 origin "${NINFER_REF}" \
    && git checkout --detach FETCH_HEAD

COPY patches/ninfer-target-sm-count.patch /tmp/ninfer-target-sm-count.patch
RUN git apply --check /tmp/ninfer-target-sm-count.patch \
    && git apply /tmp/ninfer-target-sm-count.patch

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=120a \
        -DNINFER_TARGET_SM_COUNT=170 \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer-serve

# NInfer links CUDA kernels into one very large executable. Remove ELF symbol
# tables before copying it into the runtime image so Serverless workers have
# substantially less data to pull during cold starts.
RUN strip --strip-unneeded /build/apps/ninfer-serve

FROM build AS debug

# Runpod injects the real host driver in these directories. CUDA 13.1 also
# ships a forward-compat libcuda which cannot initialize GeForce GPUs.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        curl \
        openssh-server \
        python3 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /root/.ssh /models

COPY debug_entrypoint.py /opt/ninfer-runpod/debug_entrypoint.py

EXPOSE 22
STOPSIGNAL SIGTERM
ENTRYPOINT ["python3", "/opt/ninfer-runpod/debug_entrypoint.py"]

FROM ubuntu:24.04

# The host injects libcuda. NInfer only dynamically links CUDA's small runtime
# library, so copying it directly avoids shipping the 1.2 GB CUDA runtime image.
COPY --from=build /usr/local/cuda-13.1/targets/x86_64-linux/lib/libcudart.so.13* /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libswscale7 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN ldconfig

COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve
COPY entrypoint.py /opt/ninfer-runpod/entrypoint.py
COPY queue_entrypoint.py /opt/ninfer-runpod/queue_entrypoint.py
COPY queue_handler.py /opt/ninfer-runpod/queue_handler.py
COPY requirements.txt /opt/ninfer-runpod/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir \
        --requirement /opt/ninfer-runpod/requirements.txt

ENV MODEL_REPO_ID=lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4 \
    MODEL_FILENAME=qwen3_8_27b_nvfp4.ninfer \
    MODEL_ID=qwen3.8-27b-huihui-abliterated \
    MAX_CONTEXT=204800 \
    KV_CAPACITY=204800 \
    KV_DTYPE=int8 \
    MAX_CONCURRENCY=1 \
    PREFILL_CHUNK=4096 \
    DEFAULT_MAX_TOKENS=32768 \
    DRAFT_TOKENS=3 \
    PORT=8080 \
    PORT_HEALTH=8081 \
    RUNPOD_INIT_TIMEOUT=1200 \
    REQUEST_TIMEOUT=3600

EXPOSE 8080 8081
STOPSIGNAL SIGTERM
ENTRYPOINT ["python3", "/opt/ninfer-runpod/queue_entrypoint.py"]
