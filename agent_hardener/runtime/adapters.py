# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent runners (invoke agents) and uploaders (deploy defender changes to the victim)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import httpx
import yaml

from agent_hardener.errors import VictimUnavailableError
from agent_hardener.openshell.lifecycle import OpenShellCommandResult, OpenShellConfig, OpenShellLifecycle
from agent_hardener.rate_limits import raise_for_rate_limit_response, rate_limit_error_from_exception
from agent_hardener.runtime.agent_loader import (
    failure_output,
    invoke_run_callable,
    load_run_callable,
    normalize_agent_output,
)

if TYPE_CHECKING:
    from agent_hardener.models import AgentConfig, AgentRunInput, AgentRunOutput, VictimControlConfig
    from agent_hardener.runtime.run_context import RunContext


class AgentRunner(Protocol):
    """Invoker contract used by the orchestrator."""

    async def run(self, agent: AgentConfig, request: AgentRunInput, ctx: RunContext | None = None) -> AgentRunOutput:
        """Run one configured agent."""


class InProcessAgentRunner:
    """Run configured implementations in the orchestrator process."""

    async def run(self, agent: AgentConfig, request: AgentRunInput, ctx: RunContext | None = None) -> AgentRunOutput:
        try:
            run_callable = load_run_callable(agent.implementation)
            output = await asyncio.wait_for(
                invoke_run_callable(run_callable, request, agent, ctx),
                timeout=agent.timeout_seconds,
            )
            return normalize_agent_output(agent, request, output)
        except Exception as exc:
            rate_limit = rate_limit_error_from_exception(exc, source=f"agent {agent.name}")
            if rate_limit is not None:
                raise rate_limit from exc
            if isinstance(exc, VictimUnavailableError):
                raise  # infra-down is fatal: propagate to abort, not a soft ok=False replayed against a dead victim
            return failure_output(agent, request, exc)


class HTTPAgentRunner:
    """Run configured wrappers through POST /run."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient()

    async def run(self, agent: AgentConfig, request: AgentRunInput, ctx: RunContext | None = None) -> AgentRunOutput:
        try:
            response = await self._client.post(
                _run_url(agent), json=request.model_dump(mode="json"), timeout=agent.timeout_seconds
            )
            raise_for_rate_limit_response(response, source=f"agent service {agent.name}")
            response.raise_for_status()
            return normalize_agent_output(agent, request, response.json())
        except Exception as exc:
            rate_limit = rate_limit_error_from_exception(exc, source=f"agent service {agent.name}")
            if rate_limit is not None:
                raise rate_limit from exc
            if isinstance(exc, VictimUnavailableError):
                raise  # infra-down is fatal: propagate to abort, not a soft ok=False replayed against a dead victim
            return failure_output(agent, request, exc)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


class RoutingAgentRunner:
    """Route to HTTP when service_url is configured, otherwise run in-process."""

    def __init__(
        self,
        http_adapter: AgentRunner | None = None,
        in_process_adapter: AgentRunner | None = None,
    ) -> None:
        self.http_adapter = http_adapter or HTTPAgentRunner()
        self.in_process_adapter = in_process_adapter or InProcessAgentRunner()

    async def run(self, agent: AgentConfig, request: AgentRunInput, ctx: RunContext | None = None) -> AgentRunOutput:
        if agent.service_url:
            return await self.http_adapter.run(agent, request, ctx)
        return await self.in_process_adapter.run(agent, request, ctx)

    async def close(self) -> None:
        """Close adapters that hold resources."""
        for adapter in (self.http_adapter, self.in_process_adapter):
            close = getattr(adapter, "close", None)
            if close is not None:
                await close()


class Uploader(Protocol):
    """Deploys one defender artifact (a YAML file) to the running victim.

    Each uploader knows how to push exactly one kind of file in its own way; the orchestrator
    decides *whether* to call it via a visible ``if`` on the defenders' patches (see
    :func:`openshell_policy_patch`/:func:`relay_guardrail_patch`). The uploader itself holds no branching.
    """

    async def upload(self, yaml_path: Path, *, recreate: bool = False) -> OpenShellCommandResult:
        """Deploy ``yaml_path`` to the victim; ``recreate`` requests a full rebuild when supported."""


class OpenshellUploader:
    """Deploys an OpenShell policy YAML: recreate the sandbox when asked, else apply in place.

    After a successful apply, verifies the sandbox's active policy matches the candidate so a silent
    no-op apply is reported as a failure.
    """

    def __init__(self, lifecycle: OpenShellLifecycle) -> None:
        self.lifecycle = lifecycle

    async def upload(self, yaml_path: Path, *, recreate: bool = False) -> OpenShellCommandResult:
        policy_path = Path(str(yaml_path))
        apply = self._recreate if recreate else self.lifecycle.apply_policy
        applied = await asyncio.to_thread(apply, policy_path)
        if not applied.ok:
            return applied
        return await asyncio.to_thread(self._verify, policy_path)

    def _recreate(self, policy_path: Path) -> OpenShellCommandResult:
        # restart(policy_path=…) already applies the new policy via up(); no need to clone the lifecycle
        # (cloning also dropped run-scoped state like log_dir, splitting logs across two dirs).
        return self.lifecycle.restart(policy_path=policy_path)

    def _verify(self, policy_path: Path) -> OpenShellCommandResult:
        active = self.lifecycle.active_policy()
        if not active.ok:
            return active
        try:
            candidate_digest = _policy_digest(policy_path.read_text(encoding="utf-8"))
            active_digest = _policy_digest(active.output)
        except Exception as exc:  # malformed YAML on either side
            return OpenShellCommandResult(ok=False, command=active.command, output=f"policy compare failed: {exc}")
        matches = candidate_digest == active_digest
        return OpenShellCommandResult(
            ok=matches,
            command=active.command,
            output="active policy matches candidate" if matches else "active policy does not match candidate",
        )


class RelayPolicyUploader:
    """Deploys the hardened Relay guardrail set: upload ``plugins.toml`` then restart the victim.

    The agent runs inside the OpenShell sandbox, so this composes the same lifecycle. It is the
    "quick" path — no sandbox rebuild. Falls back to a full recreate when no upload destination is
    configured.

    The restart is what applies the guardrail: Relay layers the discovered
    ``/etc/nemo-relay/plugins.toml`` in when the agent calls ``nemo_relay.plugin.initialize()`` at
    startup. Note this no longer depends on *how* the agent is launched — under NAT the start command
    also had to serve the uploaded path, and a hand-written one that did not silently discarded every
    round's hardening.
    """

    def __init__(self, lifecycle: OpenShellLifecycle) -> None:
        self.lifecycle = lifecycle

    async def upload(self, yaml_path: Path, *, recreate: bool = False) -> OpenShellCommandResult:
        plugins_path = Path(str(yaml_path))
        destination = self._destination(plugins_path)
        if destination is None:
            return await asyncio.to_thread(self.lifecycle.restart, None, True)
        uploaded = await asyncio.to_thread(self.lifecycle.upload_file, plugins_path, destination)
        if not uploaded.ok:
            return uploaded
        return await asyncio.to_thread(self.lifecycle.restart_victim)

    def _destination(self, plugins_path: Path) -> str | None:
        # Match by filename: the file we upload is the active-state copy, which shares its name with
        # the configured upload source (a different directory), so exact-path won't match.
        for upload in self.lifecycle.config.uploads:
            local, separator, destination = upload.partition(":")
            if separator and destination and Path(local).name == plugins_path.name:
                return destination
        return None


class NoopUploader:
    """Default uploader for non-OpenShell victims: records intent, deploys nothing."""

    async def upload(self, yaml_path: Path, *, recreate: bool = False) -> OpenShellCommandResult:
        return OpenShellCommandResult(ok=True, command=[], output=f"noop uploader: skipped {yaml_path}")


def build_uploaders(
    config: VictimControlConfig, lifecycle: OpenShellLifecycle | None = None
) -> tuple[Uploader, Uploader]:
    """Build ``(openshell_uploader, relay_uploader)`` sharing one lifecycle.

    A provided ``lifecycle`` means the caller already prepared an OpenShell sandbox, so ``config`` is
    not inspected. Otherwise the uploaders are built from ``config`` (non-OpenShell victims get no-ops).
    """
    if lifecycle is not None:
        return OpenshellUploader(lifecycle), RelayPolicyUploader(lifecycle)
    if config.type == "openshell":
        shared = OpenShellLifecycle(OpenShellConfig.from_mapping(config.config))
        return OpenshellUploader(shared), RelayPolicyUploader(shared)
    noop = NoopUploader()
    return noop, noop


def openshell_policy_patch(patches: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the changed OpenShell policy patch (with a candidate path) to deploy, or ``None``.

    ``requires_recreate`` is OR-ed across every matching patch rather than taken from the first.
    The defender emits one patch per attack it analysed and each flag describes only that attack —
    network edits hot-reload, filesystem and process edits do not — but all of them name the same
    candidate file, which holds the union of their changes. Taking the first patch's flag deploys
    an all-attacks payload on a one-attack decision, so a run whose network attack happened to sort
    first applied filesystem edits live and OpenShell rejected the file:
    ``filesystem read_write path '/sandbox' cannot be removed on a live sandbox``. Which patch
    sorts first is arbitrary, so the same run succeeded or failed on ordering alone.

    The returned patch is a copy, so each defender's own artifact still records what it asked for.
    """
    matching = [
        patch
        for patch in patches
        if patch.get("type") == "openshell_policy_candidate"
        and patch.get("changed") is not False
        and patch.get("candidate_policy_path")
    ]
    if not matching:
        return None
    return {**matching[0], "requires_recreate": any(patch.get("requires_recreate") for patch in matching)}


def relay_guardrail_patch(patches: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the Relay guardrail patch (with candidate + target paths) to deploy, or ``None``."""
    for patch in patches:
        if patch.get("type") == "relay_plugins_candidate" and patch.get("target_relay_plugins_path"):
            return patch
    return None


def _policy_digest(policy_text: str) -> str:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", policy_text)
    docs = [doc for doc in yaml.safe_load_all(clean) if doc is not None]
    if not docs:
        raise ValueError("active OpenShell policy output did not contain YAML")
    policy = _normalize_policy_for_digest(docs[-1])
    normalized = json.dumps(policy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_policy_for_digest(value: Any) -> Any:
    """Normalize OpenShell policy fields that are omitted after load/apply."""
    if isinstance(value, dict):
        normalized = {}
        for key, child in value.items():
            normalized_child = _normalize_policy_for_digest(child)
            if key == "endpoints" and normalized_child == []:
                continue
            normalized[key] = normalized_child
        return normalized
    if isinstance(value, list):
        return [_normalize_policy_for_digest(child) for child in value]
    return value


def _run_url(agent: AgentConfig) -> str:
    service_url = agent.service_url
    if service_url is None:
        msg = "agent service_url is required for HTTP execution"
        raise ValueError(msg)
    if service_url.rstrip("/").endswith("/run"):
        return service_url.rstrip("/")
    return f"{service_url.rstrip('/')}/run"
