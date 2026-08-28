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

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=120a \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer-serve

FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

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
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve
COPY entrypoint.py /opt/ninfer-runpod/entrypoint.py

ENV MODEL_REPO_ID=lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4 \
    MODEL_FILENAME=qwen3_8_27b_nvfp4.ninfer \
    MODEL_ID=qwen3.8-27b-huihui-abliterated \
    MAX_CONTEXT=262144 \
    KV_CAPACITY=262144 \
    KV_DTYPE=fp8 \
    MAX_CONCURRENCY=1 \
    PREFILL_CHUNK=4096 \
    DEFAULT_MAX_TOKENS=32768 \
    DRAFT_TOKENS=3 \
    PORT=8080 \
    PORT_HEALTH=8080 \
    NINFER_PORT=8082

EXPOSE 8080
STOPSIGNAL SIGTERM
ENTRYPOINT ["python3", "/opt/ninfer-runpod/entrypoint.py"]
