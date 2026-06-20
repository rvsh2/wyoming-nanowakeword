ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:bookworm
FROM ${BUILD_FROM}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /usr/src
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        netcat-traditional \
        python3 \
        python3-pip \
    && pip3 install --break-system-packages --no-cache-dir -U \
        setuptools \
        wheel \
    && rm -rf /var/lib/apt/lists/*

COPY . /usr/src/wyoming-nanowakeword
RUN pip3 install --no-cache-dir /usr/src/wyoming-nanowakeword

WORKDIR /
COPY nanowakeword/rootfs /

HEALTHCHECK --start-period=30s --interval=30s --timeout=5s --retries=3 \
    CMD echo '{ "type": "describe" }' \
        | nc -w 2 localhost 10400 \
        | grep -iq "nanowakeword" \
        || exit 1
