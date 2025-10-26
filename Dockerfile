# syntax=docker/dockerfile:1.6

FROM python:3.10-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKDIR=/workspace

WORKDIR ${WORKDIR}

# --- system deps ----------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        wget \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- python deps ----------------------------------------------------------
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- external repos -------------------------------------------------------
RUN mkdir -p ${WORKDIR}/external && \
    cd ${WORKDIR}/external && \
    git clone --depth=1 https://github.com/LiheYoung/Depth-Anything.git && \
    git clone --depth=1 https://github.com/ibaiGorordo/CREStereo-Pytorch.git && \
    git clone --depth=1 https://github.com/autonomousvision/DEFOM-Stereo.git && \
    git clone --depth=1 https://github.com/princeton-vl/RAFT-Stereo.git

# --- project files --------------------------------------------------------
COPY . ${WORKDIR}

ENV PYTHONPATH="${WORKDIR}:${WORKDIR}/external/Depth-Anything:${WORKDIR}/external/CREStereo-Pytorch:${WORKDIR}/external/DEFOM-Stereo:${WORKDIR}/external/RAFT-Stereo"

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
