FROM python:3.10-slim-bookworm

ARG ACTIONS_RUNNER_VERSION=2.337.0
ARG ACTIONS_RUNNER_SHA256=70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        coreutils \
        curl \
        ffmpeg \
        findutils \
        gawk \
        git \
        grep \
        gzip \
        jq \
        libcurl4 \
        libgomp1 \
        libicu72 \
        libkrb5-3 \
        liblttng-ust1 \
        libssl3 \
        libstdc++6 \
        procps \
        rsync \
        sed \
        tar \
        time \
        util-linux \
        xz-utils \
        zlib1g \
        zstd \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /usr/local/bin/uv

RUN groupadd --gid 1001 cb16-runner \
    && useradd --uid 1001 --gid 1001 --create-home --shell /bin/bash cb16-runner \
    && mkdir -p /opt/actions-runner-dist \
    && curl -fsSL \
      "https://github.com/actions/runner/releases/download/v${ACTIONS_RUNNER_VERSION}/actions-runner-linux-x64-${ACTIONS_RUNNER_VERSION}.tar.gz" \
      -o /tmp/actions-runner.tar.gz \
    && echo "${ACTIONS_RUNNER_SHA256}  /tmp/actions-runner.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/actions-runner.tar.gz -C /opt/actions-runner-dist \
    && rm -f /tmp/actions-runner.tar.gz \
    && mkdir -p \
        /cb16/raw \
        /cb16/g0 \
        /cb16/venv \
        /cb16/uv-cache \
        /cb16/worker \
        /cb16/runtime/r104 \
        /cb16/r2-native \
        /cb16/r2-authority \
        /cb16/package \
        /cb16/parent-r101 \
        /cb16/parents \
    && chown -R 1001:1001 /opt/actions-runner-dist /cb16

COPY --chown=1001:1001 ci/shanxi_runner_pre_job_gate.py /opt/cb16/ci/shanxi_runner_pre_job_gate.py
COPY --chown=1001:1001 infra/docker/runner/entrypoint.sh /opt/cb16/entrypoint.sh

RUN chmod 0555 /opt/cb16/entrypoint.sh /opt/cb16/ci/shanxi_runner_pre_job_gate.py

ENV PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/cb16/uv-cache \
    CB16_RAW_ROOT=/cb16/raw \
    CB16_G0_ROOT=/cb16/g0 \
    CB16_R11_G0_ROOT=/cb16/g0 \
    CB16_R104_ROOT=/cb16/runtime/r104 \
    CB16_VERIFIED_VENV=/cb16/venv \
    CB16_UV_CACHE_DIR=/cb16/uv-cache \
    CB16_CI_WORKER_ROOT=/cb16/worker \
    CB16_PROVISION_ENV=/run/secrets/cb16-provision.env \
    CB16_PACKAGE_ROOT=/cb16/package \
    CB16_PARENT_R101_ROOT=/cb16/parent-r101 \
    CB16_PARENT_G0=/cb16/parents/r10_1.pt \
    ACTIONS_RUNNER_HOOK_JOB_STARTED=/opt/cb16/ci/shanxi_runner_pre_job_gate.py \
    RUNNER_STATE_ROOT=/cb16/worker/actions-runner \
    RUNNER_WORK_DIR=/cb16/worker/_work \
    RUNNER_LABELS=shanxi,shanxi-docker-r11 \
    RUNNER_DISABLE_UPDATE=true

USER 1001:1001
WORKDIR /cb16/worker
ENTRYPOINT ["/opt/cb16/entrypoint.sh"]
