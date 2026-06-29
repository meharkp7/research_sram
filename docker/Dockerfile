FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    nano \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-venv \
    ngspice \
    xschem \
    magic \
    netgen \
    klayout \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]