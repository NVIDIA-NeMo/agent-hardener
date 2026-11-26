# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NeMo Helix + Agent Hardener in one container: the image handed to a user. It layers the Helix
# CLI and services on the Agent Hardener room (agent-hardener, garak, the OpenShell gateway and a
# private dockerd), so `nemo agent-hardener ...` builds and attacks the user's agent inside it.
#
# The build context is a NeMo Helix checkout; the room's scripts come from agent-hardener's docker/
# as the named context `ahd`. Build the room (room/Dockerfile) and the Studio UI assets first;
# docker/build.sh does all three:
#   docker buildx build -f docker/helix-room/Dockerfile --build-context ahd=docker \
#     --build-arg AH_ROOM_IMAGE=agent-hardener-room:0.0.19 --build-arg AH_ROOM_BUILD="$(versions.py build ...)" \
#     -t nemo-agent-hardener-room:dev --load <nemo-helix>
# Run with the launcher (starts the room, waits, opens Studio):
#   docker run --rm --entrypoint cat nemo-agent-hardener-room:dev /usr/local/share/ah-room/ah-room > ah-room
#   chmod +x ah-room && ./ah-room up --keys keys.env --agent ./my-agent
ARG AH_ROOM_IMAGE=agent-hardener-room:0.0.19
ARG STUDIO_UI_IMAGE=nhx-studio-ui:dev
# Only the static bundle is copied out, so the native build serves every CPU.
FROM --platform=$BUILDPLATFORM ${STUDIO_UI_IMAGE} AS studio-ui

FROM ${AH_ROOM_IMAGE}

# Same reason as nhx-api: annoy otherwise compiles CPU-specific instructions that SIGILL on
# AVX2-only hosts at Guardrails inference time.
ENV ANNOY_COMPILER_ARGS="-D_CRT_SECURE_NO_WARNINGS,-DANNOYLIB_MULTITHREADED_BUILD" \
    NHX_BASE_URL=http://127.0.0.1:8080 \
    PATH="/opt/nemo-helix/.venv/bin:${PATH}"

WORKDIR /opt/nemo-helix
COPY . /opt/nemo-helix
RUN uv python install 3.13 \
 && uv sync --python 3.13 --frozen --only-group functional-services --no-editable \
 && nemo agent-hardener --help >/dev/null \
 && test -f plugins/nemo-agent-hardener/examples/langgraph-victim/Dockerfile

# Where Studio looks for its bundle in NeMo Helix images (StudioConfig.static_files_path).
COPY --from=studio-ui /artifacts /static/studio
RUN test -f /static/studio/index.html

COPY --from=ahd helix-room/start-helix.sh /etc/ah-room/start.d/50-helix
COPY --from=ahd helix-room/ah-cycle /usr/local/bin/ah-cycle
# The host launcher travels in the image so the image is the only thing a user is handed.
COPY --from=ahd helix-room/ah-room /usr/local/share/ah-room/ah-room
RUN chmod 755 /etc/ah-room/start.d/50-helix /usr/local/bin/ah-cycle /usr/local/share/ah-room/ah-room

# Last, so a new record (each build has one) rebuilds only this layer.
COPY --from=ahd helix-room/versions.py /usr/local/lib/ah-room/versions.py
ARG AH_ROOM_BUILD
RUN test -n "${AH_ROOM_BUILD}" || { echo 'AH_ROOM_BUILD is unset: build with docker/build.sh' >&2; exit 1; } \
 && /usr/local/bin/python3 /usr/local/lib/ah-room/versions.py image "${AH_ROOM_BUILD}" > /usr/local/share/ah-room/versions.json

WORKDIR /work
