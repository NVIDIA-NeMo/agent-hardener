# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registration and config validation for the Agent Hardener guardrail plugin.

Deliberately thin. All the logic that decides whether a call is refused lives in
:mod:`agent_hardener.relay_plugin.policy`, which imports no Relay at all; this module only turns a
config table into registered guardrails.

Each guardrail is installed as a Relay *tool execution intercept*, which wraps the tool callback:
it decides before calling downstream, so a refusal means the tool never ran — not that it ran and
was reported.

A refusal also *latches* the request: see :func:`_latch`. Once any rail refuses, no further tool in
that request runs and the agent's next model call is answered with a fixed message instead of being
sent to the provider, so the turn ends there.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING, Any

import nemo_relay
from pydantic import ValidationError

from .blocked import blocked_response, blocked_stream_chunks
from .config import PLUGIN_KIND, Guardrail, GuardrailsComponentConfig
from .judge import LlmSafetyJudge
from .latch import BlockedRequests
from .policy import SafetyJudge, decide

logger = logging.getLogger("agent_hardener.relay_plugin")

#: Name of the latch's three intercepts. Relay namespaces registrations per family, so one name is
#: enough for the tool, LLM and streaming registrations.
_LATCH_NAME = "agent_hardener.latch"

#: Priority of the latch. Lower runs first, and this has to be first twice over: ahead of the rails'
#: ``100 + index``, so a blocked request never reaches the judge again and stops paying for
#: inference; and ahead of Relay's own Adaptive component, whose LLM execution intercept defaults to
#: priority 50. Losing that second race is not a missed optimisation — Adaptive would call the
#: provider with the refusal already in the conversation, and the turn fails instead of declining.
_LATCH_PRIORITY = 0

#: Leave the latch out entirely when set to ``1``. This exists to attribute a result: the latch ends
#: the turn by answering the model call locally, which also means a request that would have failed at
#: the provider never gets sent — so with the latch on, a broken refusal looks like a working one.
#: Turning it off is the only way to see whether the guardrail blocks on the strength of the refusal
#: itself. Read from the environment because the plugin is copied into the victim image, where the
#: only thing that reaches it is the process environment.
_LATCH_DISABLED = os.environ.get("AGENT_HARDENER_DISABLE_LATCH") == "1"

if TYPE_CHECKING:
    from collections.abc import Callable

    # Relay's own types, so mypy checks this class against the real Plugin protocol rather than a
    # local approximation of it.
    from nemo_relay import JsonObject
    from nemo_relay.plugin import ConfigDiagnostic, PluginContext


class AgentHardenerGuardrailPlugin:
    """The ``agent_hardener.pre_tool_verifier`` plugin kind."""

    def validate(self, plugin_config: JsonObject) -> list[ConfigDiagnostic] | None:
        """Reject a malformed guardrail set at initialisation, with a diagnostic.

        An error diagnostic blocks Relay's ``initialize()``, which is what we want: a victim that
        silently started with an unparseable guardrail set would answer every attack and look
        hardened in the report.
        """
        try:
            GuardrailsComponentConfig.model_validate(plugin_config)
        except ValidationError as error:
            diagnostic: ConfigDiagnostic = {
                "level": "error",
                "code": "agent_hardener.invalid_guardrails",
                "message": f"invalid Agent Hardener guardrail config: {error}",
            }
            return [diagnostic]
        return None

    def register(self, plugin_config: JsonObject, context: PluginContext) -> None:
        """Install one execution intercept per configured rail."""
        _require_context_intercepts()
        config = GuardrailsComponentConfig.model_validate(plugin_config)
        judge = LlmSafetyJudge(config.model)
        for index, rail in enumerate(config.guardrails):
            # Lower priority runs first; keep the defender's authoring order so a report reading
            # "custom_guardrail_1 refused this" matches what actually ran first.
            context.register_tool_execution_intercept(
                rail.name, 100 + index, _intercept(rail, judge, config.blocked_message)
            )
        if not _LATCH_DISABLED:
            _register_latch(context, config.blocked_message)


def _require_context_intercepts() -> None:
    """Fail startup, with the cause named, on a Relay that cannot report the managed tool call id.

    ``ToolExecutionContext`` (NVIDIA/NeMo-Relay#1029) is what hands an intercept the ``tool_call_id`` Relay recorded for the call.
    Without it a refusal has to invent one, and an invented id matches no ``tool_calls`` entry in the
    preceding assistant message — LangChain and LangGraph both append the refusal to the transcript
    anyway, and the *next* provider request is the one that fails, as an HTTP 500 that names neither
    this plugin nor the tool.

    Raised rather than worked around: the fallback would be exactly the bug this plugin was changed
    to fix, and a victim that refuses to start is easier to diagnose than one that answers 500 only
    once a guardrail fires. The check is by attribute, not by version, because the plugin runs inside
    the user's image and does not get to choose the Relay version there.
    """
    if not hasattr(nemo_relay, "ToolExecutionContext"):
        raise RuntimeError(
            "agent-hardener needs nemo-relay 0.9 or newer, whose tool execution intercepts carry a "
            "ToolExecutionContext with the managed tool_call_id; this victim's Relay predates it. "
            "Refusals would return an unmatched tool_call_id and fail the next model call with "
            "HTTP 500."
        )


def _register_latch(context: PluginContext, blocked_message: str) -> None:
    """Install the three intercepts that end a request once a rail has refused it.

    Registered here, for the process, rather than from inside the refusing call. Relay does offer
    scope-local registration, and binding the latch to the refusing scope would give it a lifetime
    for free — but a scope-local intercept loses to a globally registered one from another plugin,
    and Relay's own Adaptive component registers globally. With Adaptive loaded, a scope-local latch
    never runs: Adaptive reaches the provider first, sends the conversation with the refusal in it,
    and the turn fails.

    The cost of registering once is that every call now asks whether its request is blocked, so the
    answer lives in :mod:`agent_hardener.relay_plugin.latch` instead of in the registration itself.
    """
    context.register_tool_execution_intercept(_LATCH_NAME, _LATCH_PRIORITY, _latched_tool(blocked_message))
    context.register_llm_execution_intercept(_LATCH_NAME, _LATCH_PRIORITY, _latched_llm(blocked_message))
    context.register_llm_stream_execution_intercept(_LATCH_NAME, _LATCH_PRIORITY, _latched_llm_stream(blocked_message))


def _intercept(rail: Guardrail, judge: SafetyJudge, blocked_message: str) -> Callable[..., Any]:
    """Wrap one tool call, refusing it by *returning* the refusal.

    An execution intercept rather than a conditional-execution guardrail. A conditional-execution
    guardrail signals a refusal by returning a rejection string, and Relay turns that into a raised
    ``RuntimeError`` at the ``tools.execute`` boundary; nothing in Relay's LangChain integration
    catches it, so the exception escapes the whole turn. The agent then answers HTTP 500 instead of
    declining — unusable in production, and a war-game scores each successful block as an *error*
    rather than as blocked. Returning an outcome keeps the refusal an ordinary value.

    Conditional-execution guardrails are also unreachable from here: they run at stage 1 of the
    pipeline and execution intercepts at stage 4, so an intercept cannot catch and reshape one.

    Registered in the 0.9 shape, which takes the call as a context rather than as loose arguments.
    That is what carries ``tool_call_id``: a refusal completes the call without running the tool, so
    it is the intercept, not Relay, that builds the result the transcript will carry — and only a
    result stamped with the *managed* id correlates with the assistant message that asked for it.
    """

    async def intercept(context: nemo_relay.ToolExecutionContext, next_call: Any) -> Any:
        decision = decide(rail, context.tool_name, context.args, judge)
        if decision.refuse:
            _latch(blocked_message)
            return nemo_relay.ToolExecutionInterceptOutcome(
                _refusal_payload(decision.reason, context.tool_name, context.tool_call_id)
            )
        return _forward(await next_call(context.args))

    return intercept


def _forward(downstream: Any) -> Any:
    """Hand a downstream result back as the outcome Relay requires.

    Rebuilt rather than passed straight back: Relay rejects the downstream object and wants the
    payload it carries. What that object *is* varies by harness — a Relay result under LangChain, a
    bare dict under Hermes — so the payload is taken by attribute with the value itself as the
    fallback. Reading ``.result`` unconditionally raised AttributeError on Hermes, inside the tool
    path, on every allowed call: the tool ran and its output never came back.
    """
    payload = getattr(downstream, "result", downstream)
    if _OUTCOME_TAKES_ANNOTATION:
        return nemo_relay.ToolExecutionInterceptOutcome(payload, annotation=getattr(downstream, "annotation", None))
    return nemo_relay.ToolExecutionInterceptOutcome(payload)


#: Requests a rail has refused, keyed by propagation root uuid.
_BLOCKED = BlockedRequests()


def _request_key() -> str:
    """Identify the request this call belongs to.

    Relay exposes no request or conversation id, and no state that survives between callbacks — a
    ``ContextVar`` set in a tool intercept reads back empty in the model intercept that follows,
    because each callback runs in its own context. The propagation root uuid is what does carry:
    measured against the real victim, a tool intercept and every model call after it in the same
    turn report the same value.

    Returns ``""`` when there is nothing to key on, which reads everywhere as "not blocked" — the
    guardrail then behaves as it did before the latch existed rather than misfiring.
    """
    try:
        return str(nemo_relay.capture_propagation_context().root_uuid or "")
    except Exception:
        logger.debug("agent-hardener could not read the Relay propagation context", exc_info=True)
        return ""


def _latch(blocked_message: str) -> None:
    """Mark this request refused, so the intercepts installed at startup take over.

    Never raises. This runs inside the tool path on a call that is already being refused, so an
    exception here would replace a clean block with a crashed turn. The worst case on failure is an
    unmarked request, which behaves exactly as the guardrail did before.
    """
    del blocked_message  # the intercepts already hold it; this only flips the request's state
    key = _request_key()
    if not key:
        logger.warning("agent-hardener could not identify this request; not latching it")
        return
    _BLOCKED.mark(key)
    _record_latch()


def _record_latch() -> None:
    """Mark the latch on the current scope so a run report can show where the turn was cut short."""
    try:
        nemo_relay.scope.event("agent_hardener.latched")
    except Exception:
        logger.debug("agent-hardener could not record the latch event", exc_info=True)


def _latched_tool(blocked_message: str) -> Callable[..., Any]:
    """Refuse every remaining tool in a blocked request, without calling the judge."""
    del blocked_message  # the tool result is telemetry; only the model answer is user-facing

    async def intercept(context: nemo_relay.ToolExecutionContext, next_call: Any) -> Any:
        if not _BLOCKED.is_blocked(_request_key()):
            return _forward(await next_call(context.args))
        tool_name = context.tool_name
        reason = f"agent-hardener: refused {tool_name} (an earlier guardrail already blocked this request)"
        return nemo_relay.ToolExecutionInterceptOutcome(_refusal_payload(reason, tool_name, context.tool_call_id))

    return intercept


def _latched_llm(blocked_message: str) -> Callable[..., Any]:
    """Answer the model call from the plugin, so the agent stops instead of planning its next move."""

    async def intercept(name: str, request: Any, next_call: Any) -> Any:
        del name
        key = _request_key()
        if not _BLOCKED.is_blocked(key):
            return await next_call(request)
        payload = blocked_response(request, blocked_message)
        if payload is None:
            # Shape not recognised, so there is no answer we could return that the harness would
            # accept. Letting the model run loses the block, which the round reports honestly as an
            # attack that landed; returning a guess would raise through the turn as an HTTP 500.
            logger.warning("agent-hardener did not recognise the harness response shape; allowing the model call")
            return await next_call(request)
        _BLOCKED.discard(key)
        return payload

    return intercept


def _latched_llm_stream(blocked_message: str) -> Callable[..., Any]:
    """The streaming twin of :func:`_latched_llm`. Note there is no ``name`` argument here."""

    def intercept(request: Any, next_call: Any) -> Any:
        key = _request_key()
        if not _BLOCKED.is_blocked(key):
            return _passthrough_stream(request, next_call)
        chunks = blocked_stream_chunks(request, blocked_message)
        if chunks is None:
            logger.warning("agent-hardener did not recognise the harness stream shape; allowing the model call")
            return _passthrough_stream(request, next_call)
        _BLOCKED.discard(key)
        return _replay_stream(chunks)

    return intercept


async def _passthrough_stream(request: Any, next_call: Any) -> Any:
    """Hand the provider's own chunks straight through. ``next_call`` resolves to the iterator."""
    async for chunk in await next_call(request):
        yield chunk


async def _replay_stream(chunks: list[dict[str, Any]]) -> Any:
    for chunk in chunks:
        yield chunk


#: ``annotation`` is keyword-only and 0.8-and-later. NeMo Platform images pin nemo-relay 0.7.3,
#: where passing it is a ``TypeError`` — raised from inside the tool path, on every *allowed* call,
#: so the guardrail would break the tools it is supposed to let through. Probed rather than pinned:
#: the plugin runs inside the user's image and does not get to choose the Relay version there.
_OUTCOME_TAKES_ANNOTATION = "annotation" in inspect.signature(nemo_relay.ToolExecutionInterceptOutcome).parameters


def _refusal_payload(reason: str, tool_name: str, tool_call_id: str | None) -> Any:
    """The refusal, in the shape every harness accepts, carrying the id of the call it answers.

    Encoded with the same codec Relay's own LangChain integration uses for tool results, because
    the two ends want different things: Relay requires JSON on the wire, while DeepAgents asserts a
    ``ToolMessage | Command`` on the way out (``deepagents/middleware/filesystem.py``, in
    ``_aintercept_large_tool_result``). ``to_json`` tags the value so ``from_json`` rebuilds a real
    ``ToolMessage``, satisfying both. A bare string fails the assertion; a bare ``ToolMessage``
    fails serialisation.

    ``tool_call_id`` has to be the one Relay recorded for this call. An allowed call keeps its id for
    free — the tool's own ``ToolMessage`` carries it through the codec untouched — but a refusal
    never reaches the tool, so this is the only place the id can be set. Setting it to anything else
    (``tool_name``, as this did) leaves the transcript holding a tool result that answers no
    ``tool_calls`` entry, and the *next* provider request fails as an HTTP 500 that names neither the
    guardrail nor the tool.

    ``tool_call_id`` is ``None`` when the harness recorded none — Hermes, which has no assistant
    ``tool_calls`` list to correlate against. The tool name stands in there, as before: nothing reads
    it, and ``ToolMessage`` requires the field to be non-empty.

    Falls back to the plain string when LangChain is absent — a non-LangChain harness has no
    ``ToolMessage`` to want.
    """
    try:
        from langchain_core.messages import ToolMessage  # noqa: PLC0415
        from nemo_relay.typed import BestEffortAnyCodec  # noqa: PLC0415
    except ImportError:
        return reason
    return BestEffortAnyCodec().to_json(
        ToolMessage(content=reason, tool_call_id=tool_call_id or tool_name, name=tool_name, status="error")
    )


def register() -> None:
    """Register the plugin kind with Relay.

    Called from the staged ``sitecustomize`` before ``nemo_relay.plugin.initialize()``: initialization matches each
    discovered ``[[components]]`` entry against the kinds registered so far, so registering
    afterwards leaves the guardrail inert with no error. A function rather than an import
    side-effect, so importing the package does not mutate Relay's global registry.
    """
    nemo_relay.plugin.register(PLUGIN_KIND, AgentHardenerGuardrailPlugin())
