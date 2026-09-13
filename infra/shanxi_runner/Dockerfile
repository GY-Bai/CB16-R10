# CB16 Shanxi Docker runner image — versioned reconstruction V1
#
# Provenance:
#   - Reconstructed from `docker history --no-trunc cb16-runner-r11:latest`
#     as recorded in docs/infra/SHANXI_DOCKER_RUNNER_CONTRACT.md §1.1.
#   - Cross-checked against authority/infra/SHANXI_DOCKER_RUNNER_SNAPSHOT_V1.json.
#   - The original Dockerfile, compose file or docker-run script does not
#     exist in the repository or in the readable host paths.
#
# This file is a versioned definition of the live runner's image build
# contract, not a claim that a rebuild is bit-identical to the live image ID.
# The live image identity and the uv build-context pin are recorded in
# infra/shanxi_runner/runner_launch_spec_v1.json and
# infra/shanxi_runner/build_context_v1.json.

FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates coreutils curl findutils git grep gawk gzip \
      libgomp1 libicu72 libstdc++6 procps rsync sed tar time util-linux xz-utils \
    && rm -rf /var/lib/apt/lists/*

COPY uv /usr/local/bin/uv

RUN addgroup --gid 1001 cb16-runner \
 && useradd -m -u 1001 -g 1001 cb16-runner \
 && mkdir -p /cb16/raw /cb16/g0 /cb16/venv /cb16/uv-cache /cb16/worker \
             /cb16/runtime/r104 /cb16/r2-native /cb16/package /cb16/parent-r101 /cb16/parents

ENV UV_CACHE_DIR=/cb16/uv-cache
ENV PYTHONUNBUFFERED=1
USER cb16-runner
