# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed mirror of the OpenShell policy YAML, scoped to what this agent reads and writes.

Only ``network_policies`` is modeled precisely (the dynamic, hot-reloadable section this agent is
allowed to patch). Every other top-level section (``network_middlewares``, ``filesystem_policy``,
``landlock``, ``process``, ...) is static/out of scope and left unmodeled entirely — ``extra="allow"``
on every model here (deliberately not :class:`AgentHardenerModel`, which forbids extras) means those
keys are preserved and passed through unchanged on every load/dump round trip rather than silently
dropped, without needing an explicit field for each one.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_PassthroughModel = ConfigDict(extra="allow")


class Binary(BaseModel):
    model_config = _PassthroughModel

    path: str


class AllowRuleAction(BaseModel):
    model_config = _PassthroughModel

    method: str
    path: str


class AllowRule(BaseModel):
    model_config = _PassthroughModel

    allow: AllowRuleAction


class DenyRule(BaseModel):
    """One ``deny_rules`` entry. Shape varies by protocol — REST uses method/path, GraphQL uses

    operation_type/fields, MCP uses method="tools/call"+tool, JSON-RPC uses method (exact, no glob).
    """

    model_config = _PassthroughModel

    method: str | None = None
    path: str | None = None
    operation_type: str | None = None
    fields: list[str] | None = None
    tool: str | None = None


class Endpoint(BaseModel):
    model_config = _PassthroughModel

    host: str
    port: int | None = None
    access: Literal["read-only", "read-write", "full"] | None = None
    protocol: str | None = None  # http, graphql, mcp, json-rpc, ws, tcp, ...
    tls: str | None = None  # "skip" marks the endpoint uninspectable at L7
    options: dict[str, Any] | None = None
    rules: list[AllowRule] | None = None
    deny_rules: list[DenyRule] | None = None


class NetworkPolicyEntry(BaseModel):
    model_config = _PassthroughModel

    name: str | None = None
    endpoints: list[Endpoint] = Field(default_factory=list)
    binaries: list[Binary] | None = None

    @field_validator("binaries", mode="before")
    @classmethod
    def _normalize_binaries(cls, value: Any) -> Any:
        """Historically the LLM writer path sometimes emitted bare strings instead of

        ``{path: ...}`` objects; normalize both shapes so downstream code has one representation.
        """
        if value is None:
            return value
        return [{"path": item} if isinstance(item, str) else item for item in value]


class Policy(BaseModel):
    model_config = _PassthroughModel

    version: int = 1
    network_policies: dict[str, NetworkPolicyEntry] = Field(default_factory=dict)
    # Every other top-level section (network_middlewares, filesystem_policy, landlock, process,
    # ...) is out of scope and preserved opaquely via extra="allow" rather than modeled here.


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
        return {rule.allow.method.upper() for rule in endpoint.rules}
    if endpoint.access:
        return set(ACCESS_PRESET_METHODS.get(endpoint.access, set()))
    return set()


def effective_paths(endpoint: Endpoint) -> list[str]:
    """The set of path globs an endpoint actually allows. ``access``-only endpoints allow every

    path (represented as the wildcard ``"**"``).
    """
    if endpoint.rules:
        return [rule.allow.path for rule in endpoint.rules]
    if endpoint.access:
        return ["**"]
    return []


def parse_policy(raw: dict[str, Any]) -> Policy:
    """Validate a raw (already-YAML-parsed) policy document into :class:`Policy`."""
    return Policy.model_validate(raw)


def dump_policy(policy: Policy) -> dict[str, Any]:
    """Serialize a :class:`Policy` back to a plain dict suitable for YAML dumping."""
    return policy.model_dump(mode="json", exclude_none=True)
