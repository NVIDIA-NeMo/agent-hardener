# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-cutting runtime exceptions + the structured error dump.

Every fatal war-game failure raises an :class:`AgentHardenerError` carrying a machine-readable
``category`` and an operator-facing ``remediation``. :func:`emit_run_error` serializes the active
error to ``$AGENT_HARDENER_ERROR_FILE`` (a JSON file the caller reads back) so a run that fails inside
agent-hardener's own venv surfaces a *classified* cause upstream — not just a non-zero exit code and a
log tail the caller has to guess from. The NeMo Platform plugin sets that env var and reads the
file; an interactive ``agent-hardener run`` leaves it unset and relies on the console message alone.

This module is dependency-light (stdlib only) because runtime adapters import it; the CLI boundary
that catches these errors and exits cleanly lives in :mod:`agent_hardener.cli._errors`.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

# Path the caller sets for a structured failure dump; unset for an interactive run.
ERROR_FILE_ENVVAR = "AGENT_HARDENER_ERROR_FILE"


class AgentHardenerError(RuntimeError):
    """Base for a classified, operator-facing war-game failure.

    ``category`` groups the failure for upstream handling/telemetry; ``remediation`` is a short,
    actionable next step shown to the operator. Subclasses set ``category`` and a default
    ``remediation``; either can be overridden per instance.
    """

    category: str = "unexpected"
    default_remediation: str = ""

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = self.default_remediation if remediation is None else remediation


class SandboxProvisioningError(AgentHardenerError):
    """The OpenShell/Docker sandbox lifecycle (up/down/status) reported a failure."""

    category = "sandbox"
    default_remediation = "Check the Docker daemon and the OpenShell gateway on the host, then retry."


class VictimBuildError(AgentHardenerError):
    """The victim image could not be staged or built (missing project dir or Dockerfile)."""

    category = "victim_unavailable"
    default_remediation = "Verify the agent's project_dir, workflow path, and Dockerfile, then retry."


class VictimUnavailableError(AgentHardenerError):
    """The victim agent (or its sandbox) is unreachable mid-run — the run cannot continue.

    Raised on a transport-level failure talking to the victim (connection refused, server disconnected,
    read timeout): the sandbox or the agent has crashed — commonly because a malformed guardrail set or
    policy was deployed and the agent failed to load it. Fail loudly here rather than let the pipeline
    replay attacks and benign requests against a dead victim and report meaningless "0 blocked" results.
    """

    category = "victim_unavailable"
    default_remediation = "Inspect the victim agent log; a malformed workflow or policy often stops it loading."


class VictimNotInstrumentedError(AgentHardenerError):
    """The victim is not NeMo Relay-connected, so the war-game cannot observe or guard it.

    Agent Hardener reads the victim's tool calls from Relay's ATOF stream and delivers guardrails as Relay
    plugin config. Both need the agent to attach ``NemoRelayMiddleware`` *and* call
    ``nemo_relay.plugin.initialize(...)`` at startup — nothing activates a delivered
    ``/etc/nemo-relay/plugins.toml`` without that call.

    Fail before the first attack rather than mid-run: an uninstrumented victim accepts every attack
    and produces no ATOF, so the run reads as a working demo of a guardrail that was never installed.
    """

    category = "victim_not_instrumented"
    default_remediation = (
        "Attach NeMo Relay to the agent (middleware + nemo_relay.plugin.initialize() at startup) "
        "and confirm it writes events.atof.jsonl to the configured artifacts directory."
    )


class AttackerError(AgentHardenerError):
    """An attacker agent did not complete (e.g. timed out), so the run has no valid attack corpus.

    A partial or failed attacker produces no scored hits — reporting the run as "0 hits" would be a
    false negative (it looks like the victim resisted when the scan never finished). Fail loudly instead.
    """

    category = "attacker_failed"
    default_remediation = (
        "The attacker did not finish (often a timeout on a heavy agent). Re-run; for large agents raise "
        "the attacker timeout (garak.timeout_s) or lower attack_intensity."
    )


class BenignSuiteError(AgentHardenerError):
    """A smart-benign validator has no usable explicit request suite."""

    category = "benign_suite_failed"
    default_remediation = "Generate a suite with `agent-hardener synth-benign`, then pass it to `agent-hardener run` with `--benign-suite PATH`."


class ModelUnavailableError(AgentHardenerError):
    """A configured model can't be reached: wrong endpoint/key, or a model name the endpoint doesn't serve.

    Raised by the pre-run model preflight so a mistyped model or a bad credential fails in seconds — with
    the list of models the credentials *can* reach — instead of surfacing minutes later as an opaque
    attacker/validator error mid-run.
    """

    category = "model_unavailable"
    default_remediation = "Check the model name, endpoint URL, and API key; the error lists the reachable models."


def write_run_error(category: str, message: str, *, remediation: str = "", stack: str = "") -> None:
    """Write a structured failure to ``$AGENT_HARDENER_ERROR_FILE`` as ``{category, message, remediation, stack}``.

    A no-op when the env var is unset (an interactive run needs only the console message) or when the
    write fails (the non-zero exit + captured log remain the fallback channel). Use this for a failure
    surfaced as data rather than an exception (e.g. a lifecycle result with ``ok=False``);
    :func:`emit_run_error` is the exception-typed wrapper.
    """
    path = os.environ.get(ERROR_FILE_ENVVAR)
    if not path:
        return
    payload = {"category": category, "message": message or category, "remediation": remediation, "stack": stack}
    try:
        Path(path).write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        logger.warning("failed to write agent-hardener run-error file at %s", path, exc_info=True)


def emit_run_error(exc: BaseException) -> None:
    """Serialize *exc* to ``$AGENT_HARDENER_ERROR_FILE`` (category/remediation read off :class:`AgentHardenerError`)."""
    write_run_error(
        getattr(exc, "category", "unexpected"),
        str(exc) or exc.__class__.__name__,
        remediation=getattr(exc, "remediation", ""),
        stack="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )
