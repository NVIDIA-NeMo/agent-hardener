# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment-agnostic run engine.

``run_mission`` is the shared core the CLI (standalone OSS) and a future NeMo entry-point both drive:
bring up host backends → stand the victim up once (build/reuse sandbox, health, pre-flight synth) →
run the orchestrator once (it owns the hardening rounds) → final log → cleanup. The runner manages no
rounds. It depends only on the session config and reports progress by emitting events on the run's event
bus (``output``/``status_*`` + the orchestrator's phase events), so it is decoupled from Typer — a single
``on_event`` subscriber renders them. The garak agent_breaker attacker spawns the garak CLI itself, so the
runner no longer manages a garak service. OS-specific prep (docker host, dotenv) stays in the caller.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from agent_hardener.agents.validators.smart_benign import synthesize
from agent_hardener.agents.victims.openshell_victim import contract_for
from agent_hardener.atof import load_records, requested_tools
from agent_hardener.bundle import write_hardened_bundle
from agent_hardener.display.console import line_text
from agent_hardener.display.final.render_plain import render_final_run_log
from agent_hardener.display.final.render_rich import print_final_summary
from agent_hardener.errors import AtofSyncError, BenignSuiteError, VictimUnavailableError
from agent_hardener.events import EventType
from agent_hardener.final_log.mitigations import build_mitigations
from agent_hardener.final_log.validation_summary import build_validation
from agent_hardener.ids import generate_round_id
from agent_hardener.loggers import FILE_ONLY, EventBus, capture_run_log, emit_event, using_event_sink
from agent_hardener.models import AgentRunInput, RelayVictimSpec
from agent_hardener.openshell.lifecycle import OpenShellLifecycle, SubprocessCommandRunner
from agent_hardener.openshell.relay_victim import RELAY_PLUGINS_UPLOAD_DEST
from agent_hardener.preflight.relay import (
    RELAY_PROBE_PROMPT,
    RELAY_PROBE_TIMEOUT_SECONDS,
    check_relay_instrumentation,
    check_tool_path,
)
from agent_hardener.providers.backends import BackendManager
from agent_hardener.relay_plugin.victim import SANDBOX_ARTIFACTS_DIR
from agent_hardener.runtime.adapters import build_uploaders
from agent_hardener.runtime.event_sink import make_http_event_sink
from agent_hardener.runtime.orchestrator import Orchestrator
from agent_hardener.swarm_tracker import (
    OPENSHELL_LOGS,
    TRACES,
    RunLayout,
    agent_fingerprint_path,
    read_victim_image,
    victim_active_state_dir,
)
from agent_hardener.tools.openshell import (
    openshell_config,
    prepare_relay_victim,
    startup_error,
    target_health_url,
    wait_for_health,
)
from agent_hardener.traces import build_trace_provider

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from agent_hardener.models import AgentConfig, AttackRecord, RoundReport, SessionConfig
    from agent_hardener.openshell.lifecycle import OpenShellConfig
    from agent_hardener.traces import TraceProvider


_BACKEND_REMEDIATION = "Check the Docker daemon and the backends' compose services (`docker compose logs`), then retry."


async def _no_interview_answers(_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Default answer provider: supply nothing (non-interactive runs never interrupt anyway)."""
    return []


@dataclass
class MissionResult:
    """Outcome of a mission run."""

    reports: list[RoundReport]
    success: bool


# Lines of a failed build/compose blob shown on the console; the whole blob is in the log file.

SMART_BENIGN_IMPL_PREFIX = "agent_hardener.agents.validators.smart_benign"

_events_logger = logging.getLogger("agent_hardener.events")

#: One benign question, chosen to provoke a normal turn without asking for any privileged action.


def _log_event(event: str, payload: dict[str, Any]) -> None:
    """Event-bus subscriber that echoes each event into ``agent-hardener.log`` as a concise human line.

    Only scalar payload fields are rendered, so big serialized blobs (``attack_summary`` etc.) stay
    out of the text transcript — their full form lives in ``events.jsonl``. Routes through the
    ``agent_hardener.events`` logger, which the run-log bus captures.
    """
    if event == EventType.OUTPUT:  # ``line`` is already the line as the console shows it (glyph, elapsed)
        _events_logger.info("%s", payload.get("line", ""))
        return
    fields = " ".join(f"{k}={v}" for k, v in payload.items() if isinstance(v, (str, int, float, bool)))
    _events_logger.info("%s %s", event, fields)


def _is_smart_benign(agent: AgentConfig) -> bool:
    """True when a benign validator is backed by the smart benign package."""
    return (agent.implementation or "").startswith(SMART_BENIGN_IMPL_PREFIX)


def recorded_victim_image(session_config: SessionConfig, config: OpenShellConfig) -> str | None:
    """The image recorded for this sandbox, when no image is configured and the agent has not changed since.

    A ``--reuse`` of a Ready sandbox adopts it instead of building, so a host without Docker can reuse a
    sandbox it created from a pre-built image.
    """
    if config.image is not None:
        return None
    cwd = config.cwd or Path.cwd()
    recorded = read_victim_image(cwd, config.sandbox)
    if recorded is None:
        return None
    fingerprint = _agent_fingerprint(session_config, replace(config, image=recorded))
    return recorded if _read_fingerprint(agent_fingerprint_path(cwd, config.sandbox)) == fingerprint else None


def _agent_fingerprint(session_config: SessionConfig, openshell_config: OpenShellConfig) -> str:
    """Deterministic identity of the agent a sandbox should be running.

    Combines the victim's workflow config contents with the sandbox name, start
    command, build context, and pre-built image. Recorded when we build a sandbox and re-checked
    on ``--reuse`` so a different agent under the same sandbox name is detected.
    """
    h = hashlib.sha256()
    parts = [openshell_config.sandbox, openshell_config.start_command, str(openshell_config.build_context or "")]
    if openshell_config.image:
        parts.append(openshell_config.image)
    for part in parts:
        h.update((part or "").encode("utf-8"))
        h.update(b"\x00")
    workflow = session_config.target.agent_relay_plugins
    if workflow:
        with contextlib.suppress(OSError):
            h.update(Path(workflow).read_bytes())
    return h.hexdigest()[:16]


def _read_fingerprint(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _write_fingerprint(path: Path, fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fingerprint + "\n", encoding="utf-8")


#: Ways a run ends other than Ctrl-C: ``kill`` or a supervisor's stop (TERM), a closed terminal or dropped
#: SSH session (HUP). SIGHUP does not exist on Windows.
TERMINATION_SIGNALS = tuple(sig for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)) if sig is not None)


@contextlib.contextmanager
def teardown_on_termination() -> Iterator[None]:
    """Make SIGTERM and SIGHUP end the run the way Ctrl-C does, so its ``finally`` still tears down.

    Python's default for both is to exit on the spot, skipping every ``finally``: the sandbox, its port
    forward and the host backends were left running, and the forward kept the port bound. Raising
    ``KeyboardInterrupt`` reuses the one path a run already handles. Only the first signal raises; later
    ones are ignored, so a repeated stop cannot abort the teardown the first one started — SIGKILL still
    ends the process outright.

    A no-op off the main thread, where Python cannot install signal handlers.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _interrupt(signum: int, _frame: object) -> None:
        for sig in TERMINATION_SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        name = signal.Signals(signum).name
        logging.getLogger("agent_hardener.run").warning("received %s; tearing down", name)
        raise KeyboardInterrupt(name)

    previous = {sig: signal.signal(sig, _interrupt) for sig in TERMINATION_SIGNALS}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@dataclass
class MissionSpec:
    """The repeatable inputs to a mission, grouped so the runner and its phases share one ``ctx``."""

    session_config: SessionConfig
    rounds: int = 1
    verbose: bool = False
    mission_id: str | None = None
    skip_initial_up: bool = False
    no_cleanup: bool = False
    on_event: Callable[[str, dict[str, Any]], None] | None = None


def run_mission(
    session_config: SessionConfig,
    *,
    rounds: int = 1,
    verbose: bool = False,
    mission_id: str | None = None,
    skip_initial_up: bool = False,
    no_cleanup: bool = False,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> MissionResult:
    """Run one attack/defend/validate mission (thin wrapper over :class:`Mission`)."""
    return Mission(
        MissionSpec(
            session_config,
            rounds=rounds,
            verbose=verbose,
            mission_id=mission_id,
            skip_initial_up=skip_initial_up,
            no_cleanup=no_cleanup,
            on_event=on_event,
        )
    ).run()


class Mission:
    """Runs one attack/defend/validate mission against the configured victim.

    Holds the mission inputs (``ctx``) plus derived run-scoped state (lifecycle, uploaders, log/event bus)
    — formerly locals + closures in ``run_mission`` — so the phases read as a sequence: bring the victim up
    once at the baseline, synth the benign suite, run the orchestrator's hardening rounds, report, tear
    down. Presentation flows through the run's event bus (``ctx.on_event`` is the single subscriber — the
    CLI passes a console renderer, a future UI its own). Use the ``run_mission`` wrapper as the entry point.
    """

    def __init__(self, ctx: MissionSpec) -> None:
        self.ctx = ctx
        #: Scope ids in the victim's ATOF stream once the startup probe passed: the tool-path check
        #: counts only the others as attack traffic.
        self._probe_scopes: frozenset[str] = frozenset()
        self._prepare()

    def run(self) -> MissionResult:
        """Open the log/event sinks, run the phases in order, and always tear down at the end."""
        with contextlib.ExitStack() as sinks:
            sinks.enter_context(teardown_on_termination())
            self._open_sinks(sinks)
            self._infrastructure_started = False
            try:
                self._load_benign_suite()
                self._infrastructure_started = True
                self._announce("Sandbox", level="section")
                self._start_backends()
                self._bring_up_victim()
                reports = self._harden()
                return MissionResult(reports=reports, success=all(r.success for r in reports))
            finally:
                self._teardown()

    def _prepare(self) -> None:
        """Resolve the victim build config, run layout, lifecycle, uploaders, and host backends."""
        cfg = self.ctx.session_config
        self._base_config = prepare_relay_victim(openshell_config(cfg.victim_control))
        self._adopted_image = self._recorded_victim_image() if self.ctx.skip_initial_up else None
        if self._adopted_image:
            self._base_config = replace(self._base_config, image=self._adopted_image)
        # Seed the run from the inferred (egress-injected) policy, not the raw template, so defenders start
        # from the deployed policy (the workflow is already seeded via storage).
        if self._base_config.policy_path is not None:
            cfg.storage.victim_policy_path = self._base_config.policy_path

        # Resolve run_id once; agent-hardener.log + subprocess logs live under run-logs/<run_id>/. The
        # orchestrator reuses this run_id; layout dir creation is idempotent.
        self._run_id = cfg.storage.run_id or generate_round_id()
        run_layout = RunLayout(cfg.storage.root_dir, self._run_id)
        self._log_path = run_layout.run_dir / "agent-hardener.log"

        # Build uploaders on the PREPARED lifecycle so a defender recreate keeps the staged build_context +
        # egress-injected policy (raw config has build_context=None, which would destroy the sandbox), and
        # re-uploads the run's hardened guardrails rather than the seed.
        self._lifecycle = OpenShellLifecycle(
            self._base_config,
            log_dir=run_layout.dir_for(OPENSHELL_LOGS),
            active_state_dir=victim_active_state_dir(run_layout.run_dir),
        )
        self._openshell_uploader, self._relay_uploader = build_uploaders(cfg.victim_control, self._lifecycle)
        self._backend_specs = list(self._base_config.relay_victim.backends) if self._base_config.relay_victim else []
        self._backend_cwd = self._base_config.cwd or Path.cwd()
        # `compose up --build` is bounded by the same command_timeout_s as the sandbox image build.
        self._backend_manager = (
            BackendManager(SubprocessCommandRunner(self._base_config.command_timeout_s))
            if self._backend_specs
            else None
        )

    def _recorded_victim_image(self) -> str | None:
        """The pre-built image the running sandbox was created from, when ``--victim-image`` is not repeated.

        Without it a ``--reuse`` after a ``--victim-image`` run would see a fingerprint mismatch, tear the good
        sandbox down, and fall into a local ``docker build`` on the very host that has no Docker. Adopted only
        while the recorded fingerprint still matches, so a changed agent is rebuilt from the configured source.
        """
        return recorded_victim_image(self.ctx.session_config, self._base_config)

    def _open_sinks(self, sinks: contextlib.ExitStack) -> None:
        """Wire the per-run agent-hardener.log capture and the multicast structured-event bus onto *sinks*."""
        # Every agent_hardener.* record flows to the file subscriber under one handler lock; console stays clean.
        sinks.enter_context(capture_run_log(self._log_path))
        self._run_log = logging.getLogger("agent_hardener.run")
        # Ambient event sink for the whole run so the orchestrator's loggers and pre-flight synth emit
        # through it. Events persist per round in round_<N>/events.jsonl; the bus writes no whole-run file.
        self._event_bus = EventBus()
        sinks.enter_context(using_event_sink(self._event_bus.emit))
        if self.ctx.on_event is not None:
            self._event_bus.subscribe(self.ctx.on_event)
        self._event_bus.subscribe(_log_event)  # echo each event into agent-hardener.log as a concise human line
        http_sink = make_http_event_sink()  # POST events to the plugin's SSE ingest when configured (else None)
        if http_sink is not None:
            self._event_bus.subscribe(http_sink)

    # --- presentation (formerly nested closures) ---

    def _detail(self, text: str, *, console: bool = True) -> None:
        """Record a run-log entry (heavy build/compose blobs). The console shows it only with ``--debug``.

        ``console=False`` keeps it out of the console even then — for the final report, which the console
        already renders as the summary.
        """
        if text:
            self._run_log.info(text, extra=None if console else FILE_ONLY)

    def _announce(self, line: str, *, level: str = "info", elapsed: float | None = None) -> None:
        """Emit a console line as an ``output`` event — rendered by the subscriber and logged.

        ``level`` picks how the console renders it: ``section``, ``step``, ``ok``, ``warn``, ``fail``, ``hint``,
        ``detail`` (``--debug`` only), ``info``, or ``output`` for verbatim subprocess output. ``elapsed`` (seconds)
        is shown after an ``ok`` line as ``(12.4s)``.

        The payload's ``line`` is the line as the console shows it (``✓ Sandbox ready (12.4s)``), so a subscriber
        that renders only ``line`` (Helix Studio's feed) keeps the marker; ``message``, ``level`` and ``elapsed``
        are its parts, for subscribers that style lines themselves.
        """
        payload: dict[str, Any] = {"line": line_text(level, line, elapsed), "message": line, "level": level}
        if elapsed is not None:
            payload["elapsed"] = round(elapsed, 3)
        emit_event(EventType.OUTPUT, payload)

    @contextlib.contextmanager
    def _status(self, label: str) -> Iterator[None]:
        """Bracket a blocking step with ``status_started``/``status_completed`` events (a spinner)."""
        emit_event(EventType.STATUS_STARTED, {"label": label})
        try:
            yield
        finally:
            emit_event(EventType.STATUS_COMPLETED, {"label": label})

    @contextlib.contextmanager
    def _timed(self) -> Iterator[Callable[[], float]]:
        """Yield a clock reading the seconds since the block started, for the ``✓ … (12.4s)`` lines."""
        started = time.monotonic()
        yield lambda: time.monotonic() - started

    # --- phases ---

    def _start_backends(self) -> None:
        """Start the compose-managed backends the victim's tools call and wait for every backend's health."""
        if not self._backend_manager:
            return
        with self._timed() as elapsed, self._status("Starting backends"):
            backend_up = self._backend_manager.up(self._backend_specs, self._backend_cwd)
        if not backend_up.ok:
            self._detail(backend_up.output, console=False)
            raise startup_error("backend startup failed", backend_up.output, remediation=_BACKEND_REMEDIATION)
        self._detail(backend_up.output)
        self._announce("Backends ready", level="ok", elapsed=elapsed())

    def _bring_up_victim(self) -> None:
        """Stand the victim up once at the baseline: build/reuse the sandbox and gate health (synth is separate)."""
        cfg, base_config = self.ctx.session_config, self._base_config
        # --reuse only sets skip_initial_up, which matches by sandbox NAME. Gate reuse on an agent
        # fingerprint (workflow + start command + build context + image) so a different agent under the same
        # name forces a rebuild instead of probing the wrong victim.
        fp = _agent_fingerprint(cfg, base_config)
        fp_path = agent_fingerprint_path(base_config.cwd or Path.cwd(), base_config.sandbox)
        if self._adopted_image:
            self._announce(f"Reusing the pre-built victim image recorded for this sandbox: {self._adopted_image}")
        fingerprint_ok = _read_fingerprint(fp_path) == fp
        # A rotated API key leaves the provider (and every sandbox started under it) holding the old one.
        credentials_drift = self._lifecycle.credentials_drift()
        reuse_ok = self.ctx.skip_initial_up and fingerprint_ok and credentials_drift is None
        if self.ctx.skip_initial_up and not fingerprint_ok:
            self._announce(
                "Running sandbox does not match the configured agent (fingerprint mismatch or "
                "unknown); rebuilding instead of reusing",
                level="warn",
            )
        if credentials_drift:
            self._announce(f"{credentials_drift}; refreshing the provider and rebuilding the sandbox", level="warn")
        if not reuse_ok:
            self._lifecycle.down()
            with self._timed() as elapsed, self._status("Building and starting sandbox"):
                result = self._lifecycle.up(None)
            if not result.ok:
                self._detail(result.output, console=False)
                raise startup_error("the victim sandbox failed to start", result.output)
            self._detail(result.output)
            self._announce("Sandbox ready", level="ok", elapsed=elapsed())
            self._announce(f"sandbox {self._base_config.sandbox}", level="detail")
            _write_fingerprint(fp_path, fp)
        else:
            # ``up`` is what starts the forward, so a reused sandbox checks it holds its port: otherwise the
            # health check and Relay probe below reach whatever else does, and pass or fail on its behalf.
            result = self._lifecycle.ensure_forward()
            if not result.ok:
                self._detail(result.output, console=False)
                raise startup_error("the reused sandbox's port forward failed", result.output)
            self._announce("Reusing the running sandbox (agent fingerprint matches)", level="ok")
        drift = self._lifecycle.image_context_drift()
        if drift:
            self._announce(drift, level="warn")
        # Ensure the victim is serving before synth probes it (wait_for_health is idempotent).
        health_url = target_health_url(cfg.target.base_url)
        if health_url:
            try:
                with self._timed() as elapsed, self._status("Waiting for the victim to become healthy"):
                    wait_for_health(health_url, base_config.health_timeout)
            except TimeoutError as exc:
                raise VictimUnavailableError(
                    f"victim did not become healthy at {health_url} within {base_config.health_timeout}s"
                ) from exc
            self._announce("Victim healthy", level="ok", elapsed=elapsed())
            self._announce(health_url, level="detail")
        self._check_relay_instrumentation()

    def _check_relay_instrumentation(self) -> None:
        """Prove the victim emits Relay telemetry before a single attack is sent.

        Load-bearing rather than a nicety: an uninstrumented victim answers every attack and emits
        nothing, so the run looks like a working demo of a guardrail that was never installed. One
        probe up front is cheap; discovering it from a clean-looking report is not.

        Skipped for a plain HTTP target with no relay-victim block — there is no sandbox to read
        telemetry out of.
        """
        spec = self._base_config.relay_victim
        if spec is None:
            return
        atof_path = self._host_atof_path(spec)
        contract = contract_for(self.ctx.session_config.target.base_url, self.ctx.session_config.victim)

        async def probe() -> Any:
            async with httpx.AsyncClient(timeout=RELAY_PROBE_TIMEOUT_SECONDS) as client:
                return await client.post(
                    contract.url, json=contract.request_payload(RELAY_PROBE_PROMPT), headers=contract.headers()
                )

        with self._status("Checking Relay instrumentation"):
            result = asyncio.run(
                check_relay_instrumentation(
                    probe=probe,
                    atof_path=atof_path,
                    sync=lambda: self._lifecycle.fetch_file(f"{SANDBOX_ARTIFACTS_DIR}/{atof_path.name}", atof_path),
                )
            )
        records = load_records(atof_path) if atof_path.exists() else []
        self._probe_scopes = frozenset(str(record["uuid"]) for record in records if "uuid" in record)
        self._announce("Relay instrumentation confirmed", level="ok")
        self._announce(f"{result.records_observed} Relay event(s) observed from the probe", level="detail")
        if result.quarantined:
            # Not fatal: the run still produces findings, but the trajectory is untrustworthy and a
            # report that does not say so reads as more certain than it is.
            self._announce("Relay reported a dirty scope stack; the recorded trajectory may be wrong", level="warn")

    def _host_atof_path(self, spec: RelayVictimSpec) -> Path:
        """Where the victim's ATOF log lands on *this* machine."""
        cwd = self._base_config.cwd or Path.cwd()
        return cwd / spec.atof_path if not spec.atof_path.is_absolute() else spec.atof_path

    def _verify_tool_path(self, attacks: list[AttackRecord] | None = None) -> None:
        """Prove the victim's tool calls pass through Relay, from the first round's attack traffic.

        The startup probe proves Relay is *attached*; it cannot prove tool calls *reach* it, because a
        greeting provokes no tool. Garak does, so this runs once the attacks have — no probe tool to
        declare, no synthetic call, and no side effect Agent Hardener caused itself.

        The copy is fetched strictly here: the startup probe already read the stream, so a fetch that
        fails now is a sync fault, and judging a stale copy would blame the victim's wiring for it.

        Skipped for a plain HTTP target with no relay-victim block, matching the startup probe: there
        is no sandbox to read telemetry out of.

        Args:
            attacks: The live attackers' records, when they produced the traffic; their targets are
                checked against the tools the victim actually used. ``None`` after a replay.
        """
        spec = self._base_config.relay_victim
        if spec is None:
            return
        atof_path = self._host_atof_path(spec)
        self._sync_atof(strict=True)
        records = load_records(atof_path) if atof_path.exists() else []
        tools = check_tool_path(records, before=self._probe_scopes)
        self._announce(f"Tool calls reach Relay ({', '.join(tools)})", level="ok")
        if attacks:
            self._warn_untouched_targets(attacks, {*tools, *requested_tools(records)})

    def _warn_untouched_targets(self, attacks: list[AttackRecord], used: set[str]) -> None:
        """Name the attack targets the victim never asked for nor ran.

        Not fatal: the run is valid, but attempts spent on a tool the victim does not have (garak has
        targeted ``multi_tool_use.parallel``, an artifact of OpenAI's tool-call format) are attacks that
        never tested anything, and a report that does not say so overstates the coverage.
        """
        targets = {str(target) for attack in attacks for target in attack.metadata.get("attack_targets") or []}
        untouched = sorted(targets - used)
        if untouched:
            self._announce(
                f"attacks targeted {len(untouched)} tool(s) the victim never asked for or ran "
                f"({', '.join(untouched)}) — they may not exist, so those attempts tested nothing",
                level="warn",
            )

    def _sync_atof(self, *, strict: bool = False) -> None:
        """Copy the victim's ATOF stream out of the sandbox, best-effort unless ``strict``.

        Separate from :meth:`_verify_tool_path` because collecting the telemetry and asserting
        something about it are different jobs: the assertion only makes sense when live attackers
        produced the traffic, while the collection is worth doing for every round.

        Raises:
            AtofSyncError: ``strict`` and the stream could not be copied.
        """
        spec = self._base_config.relay_victim
        if spec is None:
            return
        atof_path = self._host_atof_path(spec)
        remote = f"{SANDBOX_ARTIFACTS_DIR}/{atof_path.name}"
        if not strict:
            with contextlib.suppress(Exception):
                self._lifecycle.fetch_file(remote, atof_path)
            return
        try:
            fetched = self._lifecycle.fetch_file(remote, atof_path)
        except Exception as exc:
            raise AtofSyncError(f"could not copy {remote} out of the victim's sandbox: {exc}") from exc
        if not fetched:
            raise AtofSyncError(
                f"could not copy {remote} out of the victim's sandbox: the copy came back empty, although "
                "the startup probe read it, so the victim's telemetry is unreadable rather than absent"
            )

    def _load_benign_suite(self) -> None:
        """Pre-flight: seed the supplied benign suite for each smart-benign validator.

        ``run`` is a pure consumer — it never synthesizes. Every configured smart-benign validator
        requires a suite generated out of band (``agent-hardener synth-benign`` or the serve HITL) and
        supplied via ``--benign-suite``. Missing, empty, or unreadable suites fail before infrastructure
        startup so validation can never fall back to stale cached requests.
        """
        cfg = self.ctx.session_config
        validators = [v for v in cfg.benign_validators if _is_smart_benign(v)]
        if not validators:
            return
        self._announce("Benign suite", level="section")
        if cfg.benign_suite_path is None:
            raise BenignSuiteError("smart-benign validation requires an explicit benign suite, but none was supplied")
        context = {"storage_root": str(cfg.storage.root_dir)}
        for agent in validators:
            request = AgentRunInput(
                round_id="preflight-load", target=cfg.target, context=context, validator_kind="benign"
            )
            try:
                with self._status(f"Loading benign suite ({agent.name})"):
                    result = asyncio.run(
                        synthesize(
                            request,
                            agent,
                            answer_provider=_no_interview_answers,
                            source_suite=cfg.benign_suite_path,
                        )
                    )
            except Exception as exc:
                raise BenignSuiteError(
                    f"failed to load benign suite {cfg.benign_suite_path} for {agent.name}: {exc}"
                ) from exc
            if not result.viable:
                raise BenignSuiteError(
                    f"benign suite {cfg.benign_suite_path} contains no replayable requests for {agent.name}"
                )
            self._announce(f"Benign suite loaded: {result.request_count} request(s) for {agent.name}", level="ok")
            self._announce(f"from {cfg.benign_suite_path}", level="detail")

    def _harden(self) -> list[RoundReport]:
        """Run the orchestrator's hardening rounds and persist the final report."""
        # The orchestrator owns the loop: one run_rounds runs the configured rounds, re-running the
        # attackers and carrying each round's deployed policy forward. Events flow through the ambient sink.
        orchestrator = Orchestrator(
            self.ctx.session_config,
            openshell_uploader=self._openshell_uploader,
            relay_uploader=self._relay_uploader,
            verify_tool_path=self._verify_tool_path,
            sync_atof=self._sync_atof,
            trace_provider=self._build_trace_provider(),
        )
        reports = asyncio.run(
            orchestrator.run_rounds(rounds=self.ctx.rounds, mission_id=self.ctx.mission_id, run_id=self._run_id)
        )
        # Always persist the full report to the run log (--verbose only controls inline detail). The
        # console summary is rendered here (not via an event) as a deliberate, bounded exception: it is a
        # one-shot terminal render that needs the full RoundReport objects, which don't fit the
        # serializable-event model the rest of the run uses.
        mission_ids = [self.ctx.mission_id] if self.ctx.mission_id else None
        self._detail("\n" + render_final_run_log(reports, mission_ids=mission_ids), console=False)
        self._announce("Report", level="section")
        print_final_summary(reports, verbose=self.ctx.verbose)
        cfg = self.ctx.session_config
        # mitigations.json: the run's before/after policy+workflow — the NeMo plugin saves it for the Studio
        # Mitigations view. Its policy/workflow names come from the configured victim-control paths.
        self._write_run_artifact(
            "mitigations.json",
            lambda run_dir: build_mitigations(
                run_dir,
                reports,
                policy_name=cfg.storage.victim_policy_path.name if cfg.storage.victim_policy_path else None,
                guardrails_name=cfg.storage.victim_relay_plugins_path.name
                if cfg.storage.victim_relay_plugins_path
                else None,
            ),
        )
        # validation.json: the final per-item attack/benign results — drives the Studio sanity-check
        # scorecard. Written for every run that ran validators, including frozen validate-only runs.
        self._write_run_artifact("validation.json", lambda _run_dir: build_validation(reports))
        self._write_hardened_bundle()
        return reports

    def _build_trace_provider(self) -> TraceProvider | None:
        """The configured trace provider, saving raw data under ``<run_dir>/traces/``.

        ``None`` only for a plain HTTP target with no relay-victim block: there is no sandbox to read the
        victim's telemetry out of, so the guardrails defender writes no rails (each attack fails, saying why).
        """
        traces = self.ctx.session_config.traces
        spec = self._base_config.relay_victim
        provider = build_trace_provider(
            traces,
            relay_spec=spec,
            atof_path=self._host_atof_path(spec) if spec is not None else None,
            sync=self._sync_atof,
            raw_dir=RunLayout(self.ctx.session_config.storage.root_dir, self._run_id).dir_for(TRACES),
        )
        if provider is not None:
            self._announce(f"execution traces from {traces.provider}", level="detail")
        return provider

    def _write_hardened_bundle(self) -> None:
        """Assemble the run's deployable artifact: the image the run validated plus its guardrails.

        Best-effort. The bundle is the *useful* output, but a packaging failure must not discard a run
        whose report is already written — the operator is told, and the evidence survives either way.
        """
        spec = self._base_config.relay_victim
        build_context, image = self._base_config.build_context, self._base_config.image
        if spec is None or (build_context is None and image is None):
            return
        run_dir = RunLayout(self.ctx.session_config.storage.root_dir, self._run_id).run_dir
        active_state = victim_active_state_dir(run_dir)
        plugins_toml = active_state / RELAY_PLUGINS_UPLOAD_DEST.rsplit("/", 1)[-1]
        if not plugins_toml.is_file():
            return
        # The run hardens two things; ship both. The policy is the half docker cannot apply.
        configured_policy = self.ctx.session_config.storage.victim_policy_path
        policy = active_state / configured_policy.name if configured_policy else None
        try:
            bundle = write_hardened_bundle(
                # build_context is the staged *Dockerfile*; the bundle is its whole directory.
                build_root=build_context.parent if build_context else None,
                plugins_toml=plugins_toml,
                destination=run_dir / "hardened-image",
                agent=self.ctx.session_config.target.name,
                run_id=self._run_id,
                policy=policy,
                image=image,
            )
        except Exception:
            self._run_log.warning("could not assemble the hardened image bundle", exc_info=True)
            return
        self._announce(f"hardened image bundle: {bundle}", level="hint")

    def _write_run_artifact(self, filename: str, build: Callable[[Path], Any]) -> None:
        """Persist a best-effort JSON run artifact next to the run log; skip a falsy payload, never raise."""
        try:
            run_dir = RunLayout(self.ctx.session_config.storage.root_dir, self._run_id).run_dir
            payload = build(run_dir)
            if payload:
                (run_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            self._run_log.debug("failed to write %s", filename, exc_info=True)

    def _save_victim_logs(self) -> None:
        """Keep the victim's in-sandbox logs before the sandbox is deleted, and say where. Best-effort.

        A victim that starts healthy and then fails its requests leaves its traceback only there.
        """
        try:
            run_dir = RunLayout(self.ctx.session_config.storage.root_dir, self._run_id).run_dir
            output = self._lifecycle.collect_victim_diagnostics().output
            if not output:
                return
            saved = run_dir / "victim-diagnostics.log"
            saved.write_text(output, encoding="utf-8")
        except Exception:
            self._run_log.debug("failed to save victim logs", exc_info=True)
            return
        self._announce(f"victim log: {saved}", level="hint")

    def _teardown(self) -> None:
        """Tear the sandbox + backends down, unless --no-cleanup left them running.

        A run that failed before the sandbox stage started nothing, so it still clears any sandbox an
        earlier run left behind, but only into the log: the console never claims a teardown it did not need.
        """
        if not self._infrastructure_started:
            if not self.ctx.no_cleanup:
                self._detail(self._lifecycle.down().output, console=False)
            return
        self._announce("Cleanup", level="section")
        if self.ctx.no_cleanup:
            self._announce("Cleanup skipped: the sandbox and backends are still running", level="warn")
            self._announce("tear them down with: agent-hardener down", level="hint")
            return
        with self._timed() as elapsed, self._status("Tearing down the sandbox"):
            self._save_victim_logs()
            self._detail(self._lifecycle.down().output)
            if self._backend_manager:
                self._detail(self._backend_manager.down(self._backend_specs, self._backend_cwd).output)
        self._announce("Sandbox torn down", level="ok", elapsed=elapsed())
