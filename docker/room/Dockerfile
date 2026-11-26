# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Agent Hardener room: agent-hardener, its garak venv, the OpenShell gateway and a private
# Docker engine in one image. Sandboxes run on the room's own dockerd, so 127.0.0.1:17670 and
# :8000 belong to this container and many rooms can run side by side on one host.
#
# The build context is this repository's root; the room's own scripts come from docker/ as the
# named context `ahd` (docker/build.sh does both):
#   docker buildx build -f docker/room/Dockerfile --build-context ahd=docker -t agent-hardener-room:0.0.19 --load .
# Run (privileged: the room starts its own dockerd):
#   docker run -d --name ah-room --privileged \
#     -v "$PWD/keys.env:/run/secrets/keys.env:ro" agent-hardener-room:0.0.19
#   docker exec -it ah-room ah-cycle
FROM python:3.12-bookworm

ARG OPENSHELL_VERSION=v0.1.4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HOME=/root \
    PATH="/opt/agent-hardener/src/.venv/bin:/root/.local/bin:${PATH}" \
    OPENSHELL_VERSION=${OPENSHELL_VERSION}

# Docker CE, not Debian's docker.io: 20.10 resolves the sandbox's "999:999" archive user as a
# literal name and fails every sandbox create with BoundaryConfigWriteFailed. The sandbox user
# comes first: docker-ce otherwise takes GID 999 for its own `docker` group.
RUN groupadd --gid 999 sandbox \
 && useradd --uid 999 --gid 999 --system --create-home --home-dir /home/sandbox --shell /usr/sbin/nologin sandbox \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git gnupg iptables tini xz-utils \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
 && chmod a+r /etc/apt/keyrings/docker.asc \
 && printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable\n' \
      "$(dpkg --print-architecture)" > /etc/apt/sources.list.d/docker.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce docker-ce-cli containerd.io docker-buildx-plugin \
 && rm -rf /var/lib/apt/lists/* \
 && dockerd --version

# The package install tries a user systemd unit, which a container does not have; the
# entrypoint starts the gateway instead.
RUN curl -fsSL "https://raw.githubusercontent.com/NVIDIA/OpenShell/${OPENSHELL_VERSION}/install.sh" -o /tmp/install-openshell.sh \
 && OPENSHELL_VERSION="${OPENSHELL_VERSION}" sh /tmp/install-openshell.sh \
 && rm /tmp/install-openshell.sh \
 && command -v openshell \
 && command -v openshell-gateway

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && command -v uv

WORKDIR /opt/agent-hardener/src
COPY . /opt/agent-hardener/src
# NeMo Helix's plugin looks for the CLI at ~/.agent-hardener/venv; link it so a Helix image
# built on this one passes `nemo agent-hardener doctor` without a second install.
RUN uv sync --frozen --no-dev --python 3.12 \
 && agent-hardener setup \
 && test -x /root/.agent-hardener/garak-venv/bin/python \
 && ln -sfn /opt/agent-hardener/src/.venv /root/.agent-hardener/venv

COPY --from=ahd room/entrypoint.sh /usr/local/bin/ah-room-entrypoint
COPY --from=ahd room/models.sh /usr/local/lib/ah-room/models.sh
COPY --from=ahd room/keys.py /usr/local/lib/ah-room/keys.py
COPY --from=ahd room/ah-cycle /usr/local/bin/ah-cycle
RUN chmod 755 /usr/local/bin/ah-room-entrypoint /usr/local/bin/ah-cycle \
 && mkdir -p /etc/ah-room/start.d /work

WORKDIR /work
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/ah-room-entrypoint"]
CMD ["sleep", "infinity"]
