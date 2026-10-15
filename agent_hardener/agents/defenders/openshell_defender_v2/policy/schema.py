# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed mirror of the OpenShell policy YAML, scoped to what this agent reads and writes.

Only ``network_policies`` is modeled precisely (the dynamic, hot-reloadable section this agent is
allowed to patch), and it is modeled *strictly*: OpenShell rejects unknown, misspelled and
obsolete fields, so ``extra="forbid"`` on every network model surfaces a policy the gateway would
refuse at load time rather than at ``sandbox create`` / ``policy set``. Every other top-level section
(``network_middlewares``, ``filesystem_policy``, ``landlock``, ``process``) is static/out of scope and
left unmodeled — ``extra="allow"`` on :class:`Policy` alone preserves those keys unchanged on every
load/dump round trip rather than silently dropping them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_PassthroughModel = ConfigDict(extra="allow")
_StrictModel = ConfigDict(extra="forbid")

#: L7 protocols OpenShell accepts on an endpoint.
Protocol = Literal["rest", "websocket", "graphql", "mcp", "json-rpc", "tcp"]
#: Request-level protocols — the ones ``access`` / ``rules`` / ``deny_rules`` apply to.
REQUEST_PROTOCOLS = frozenset({"rest", "websocket", "graphql", "mcp", "json-rpc"})

#: A string matcher, or ``{any: [...]}`` (REST/WebSocket query values, MCP tool names).
Matcher = str | dict[str, Any]


class Binary(BaseModel):
    model_config = _StrictModel

    path: str


class AllowRuleAction(BaseModel):
    """The matcher inside one ``rules[].allow``. Shape varies by protocol — REST/WebSocket use

    method/path/query, GraphQL uses operation_type/operation_name/fields, MCP uses method + tool
    (or ``params.name``), JSON-RPC uses method.
    """

    model_config = _StrictModel

    method: str | None = None
    path: str | None = None
    query: dict[str, Matcher] | None = None
    operation_type: str | None = None
    operation_name: str | None = None
    fields: list[str] | None = None
    tool: Matcher | None = None
    params: dict[str, Matcher] | None = None


class AllowRule(BaseModel):
    model_config = _StrictModel

    allow: AllowRuleAction


class DenyRule(BaseModel):
    """One ``deny_rules`` entry. Shape varies by protocol — REST uses method/path(/query), GraphQL

    uses operation_type/fields, MCP uses method="tools/call"+tool, JSON-RPC uses method (exact, no glob).
    """

    model_config = _StrictModel

    method: str | None = None
    path: str | None = None
    query: dict[str, Matcher] | None = None
    operation_type: str | None = None
    operation_name: str | None = None
    fields: list[str] | None = None
    tool: Matcher | None = None
    params: dict[str, Matcher] | None = None


class Endpoint(BaseModel):
    """One ``network_policies.<key>.endpoints[]`` entry.

    Fields this agent never edits (credential binding/signing, protocol options) are typed loosely —
    they only need to round-trip — but are still declared so an unknown key is rejected.
    """

    model_config = _StrictModel

    host: str
    port: int | None = None
    ports: list[int] | None = None
    path: str | None = None
    allowed_ips: list[str] | None = None
    protocol: Protocol | None = None
    tls: Literal["skip"] | None = None  # omitted = automatic inspection; "skip" = L4 passthrough
    enforcement: Literal["enforce", "audit"] | None = None  # OpenShell default: audit (log only)
    access: Literal["read-only", "read-write", "full"] | None = None
    rules: list[AllowRule] | None = None
    deny_rules: list[DenyRule] | None = None
    allow_encoded_slash: bool | None = None
    # Credential handling — round-tripped, never edited.
    credential_binding: dict[str, Any] | None = None
    request_body_credential_rewrite: bool | None = None
    websocket_credential_rewrite: bool | None = None
    allow_uninspected_credentials: bool | None = None
    credential_signing: str | None = None
    signing_service: str | None = None
    signing_region: str | None = None
    # Protocol options — round-tripped, never edited.
    persisted_queries: str | None = None
    graphql_persisted_queries: dict[str, Any] | None = None
    graphql_max_body_bytes: int | None = None
    mcp: dict[str, Any] | None = None
    json_rpc: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _openshell_endpoint_constraints(self) -> Endpoint:
        """The per-endpoint rules OpenShell enforces when it loads a policy."""
        if self.port is not None and self.ports is not None:
            raise ValueError("endpoint sets both 'port' and 'ports'; OpenShell accepts one or the other")
        if self.access is not None and self.rules is not None:
            raise ValueError("endpoint sets both 'access' and 'rules'; OpenShell accepts one or the other")
        if self.deny_rules and self.protocol is None:
            raise ValueError("endpoint has 'deny_rules' but no 'protocol'")
        if self.tls == "skip" and self.protocol in REQUEST_PROTOCOLS:
            raise ValueError(f"'tls: skip' is not valid with request protocol {self.protocol!r}")
        if self.protocol == "tcp" and (self.access or self.rules or self.deny_rules):
            raise ValueError("'protocol: tcp' accepts no request-level fields (access/rules/deny_rules)")
        return self

    def port_list(self) -> list[int | None]:
        """Every port this endpoint covers; ``[None]`` when it names none (any port)."""
        if self.ports:
            return list(self.ports)
        return [self.port]

    def matches_port(self, port: int | None) -> bool:
        """True if ``port`` is ``None`` (unknown) or this endpoint covers it."""
        ports = self.port_list()
        return port is None or None in ports or port in ports

    def port_key(self) -> int | tuple[int, ...] | None:
        """A hashable identity for this endpoint's port(s), for keying endpoints by host + port."""
        return tuple(self.ports) if self.ports else self.port

    def enforces(self) -> bool:
        """True if a matching deny rule actually blocks (``enforce``) rather than only logs (``audit``)."""
        return self.enforcement == "enforce"


class NetworkPolicyEntry(BaseModel):
    model_config = _StrictModel

    name: str | None = None
    endpoints: list[Endpoint] = Field(default_factory=list)
    binaries: list[Binary] | None = None

    @field_validator("binaries", mode="before")
    @classmethod
    def _normalize_binaries(cls, value: Any) -> Any:
        """Historically the LLM writer path sometimes emitted bare strings instead of

        ``{path: ...}`` objects; normalize both shapes so downstream code has one representation
        (and the dump always emits the object form OpenShell requires in authored policies).
        """
        if value is None:
            return value
        return [{"path": item} if isinstance(item, str) else item for item in value]


class Policy(BaseModel):
    model_config = _PassthroughModel

    version: int = 1
    network_policies: dict[str, NetworkPolicyEntry] = Field(default_factory=dict)
    # Every other top-level section (network_middlewares, filesystem_policy, landlock, process)
    # is out of scope and preserved opaquely via extra="allow" rather than modeled here.

    @model_validator(mode="after")
    def _shared_host_port_settings(self) -> Policy:
        """OpenShell requires endpoints on the same exact host:port to agree on ``tls`` and

        ``allowed_ips`` (across entries too). Wildcard overlap is left to the gateway to judge.
        """
        seen: dict[tuple[str, int | None], tuple[str, str | None, list[str] | None]] = {}
        for key, entry in self.network_policies.items():
            for ep in entry.endpoints:
                settings = (ep.tls, sorted(ep.allowed_ips) if ep.allowed_ips else None)
                for port in ep.port_list():
                    first = seen.setdefault((ep.host.lower(), port), (key, *settings))
                    if first[1:] != settings:
                        raise ValueError(
                            f"{ep.host}:{port} in '{key}' disagrees with '{first[0]}' on tls/allowed_ips; "
                            "OpenShell requires endpoints on the same host:port to share them"
                        )
        return self


# Per OpenShell docs: the method set an ``access`` preset expands to when an endpoint has no
# explicit ``rules``.
ACCESS_PRESET_METHODS: dict[str, set[str]] = {
    "read-only": {"GET", "HEAD", "OPTIONS"},
    "read-write": {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH"},
    "full": {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"},
}


def effective_methods(endpoint: Endpoint) -> set[str]:
    """The set of HTTP methods an endpoint actually allows, from whichever of ``rules``/``access``

    it uses (the two are mutually exclusive at the policy-loader level).
    """
    if endpoint.rules:
        return {rule.allow.method.upper() for rule in endpoint.rules if rule.allow.method}
    if endpoint.access:
        return set(ACCESS_PRESET_METHODS.get(endpoint.access, set()))
    return set()


def effective_paths(endpoint: Endpoint) -> list[str]:
    """The set of path globs an endpoint actually allows. ``access``-only endpoints allow every

    path (represented as the wildcard ``"**"``).
    """
    if endpoint.rules:
        return [rule.allow.path or "**" for rule in endpoint.rules]
    if endpoint.access:
        return ["**"]
    return []


def parse_policy(raw: dict[str, Any]) -> Policy:
    """Validate a raw (already-YAML-parsed) policy document into :class:`Policy`."""
    return Policy.model_validate(raw)


def dump_policy(policy: Policy) -> dict[str, Any]:
    """Serialize a :class:`Policy` back to a plain dict suitable for YAML dumping."""
    return policy.model_dump(mode="json", exclude_none=True)
