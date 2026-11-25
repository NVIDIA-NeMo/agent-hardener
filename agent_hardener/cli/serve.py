# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``serve`` command: expose the benign-suite synth over HTTP for a UI or the NeMo plugin."""

from __future__ import annotations

from pathlib import Path

import typer

from agent_hardener.cli._app import ENV_FILE_OPTION, app, load_cli_env_file
from agent_hardener.display.console import info, section


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind (localhost by default)."),
    port: int = typer.Option(8100, "--port", help="Port to bind."),
    env_file: Path | None = ENV_FILE_OPTION,
) -> None:
    """Serve the benign-suite synth over HTTP (interview + review) for a UI or the NeMo plugin.

    The synth graph runs in this process (its checkpointer holds each run's state across interrupt/resume),
    so a client drives it via POST /synth -> /synth/{id}/answers -> /synth/{id}/suite. Synth probes the live
    victim, so its `base_url` must be reachable.
    """
    from agent_hardener.cli._errors import cli_error_boundary
    from agent_hardener.model_check import preflight_configured_models
    from agent_hardener.serve import run_server

    with cli_error_boundary():
        load_cli_env_file(env_file)
        # Synth uses the analysis LLM; fail fast on a missing credential instead of erroring per request.
        preflight_configured_models()
        section("Agent Hardener synth service")
        info(f"Serving on http://{host}:{port}")
        run_server(host=host, port=port)
