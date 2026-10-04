# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Egress discovery primitives: the endpoint value, the source contract, and URL parsing.

OpenShell sandboxes are default-deny egress, so the policy must allowlist every host the agent calls.
Rather than make the user enumerate them, :class:`EgressSource` implementations surface the hosts an
agent reaches (from its workflow, code, manual hints, or — later — runtime deny-traces); the
discoverer turns them into valid OpenShell host patterns.
"""

from __future__ import annotations

import ipaddress
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Host (+ optional explicit port) inside a URL: http(s)://HOST[:port][/...].
_URL_RE = re.compile(r"(?P<scheme>https?)://(?P<host>[A-Za-z0-9._-]+)(?::(?P<port>\d+))?", re.IGNORECASE)

# Loopback / host-gateway aliases are reached via the backend egress block, not general egress.
_SKIP_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"})  # noqa: S104 - skip-list, not a bind


@dataclass(frozen=True)
class DiscoveredEndpoint:
    """A host:port an agent appears to reach, as surfaced by a source (concrete host, not a pattern)."""

    host: str
    port: int


@dataclass(frozen=True)
class PolicyEndpoint:
    """A policy-ready egress endpoint: a valid host pattern, port, and any SSRF-exemption IPs.

    ``allowed_ips`` lists internal/private addresses the host resolves to; OpenShell's anti-SSRF guard
    blocks internal IPs unless they are explicitly allowed here.
    """

    host: str
    port: int
    allowed_ips: tuple[str, ...] = ()


def _is_ip_address(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def parse_urls(text: str) -> set[DiscoveredEndpoint]:
    """Extract ``(host, port)`` endpoints from any ``http(s)`` URLs in ``text``.

    Port is the explicit ``:port`` when present, else 443 for https and 80 for http. Loopback /
    host-gateway aliases and bare IP addresses are dropped (IPs can't be a policy host pattern).
    """
    endpoints: set[DiscoveredEndpoint] = set()
    for match in _URL_RE.finditer(text):
        host = match.group("host").rstrip(".").lower()
        if host in _SKIP_HOSTS or _is_ip_address(host):
            continue
        explicit = match.group("port")
        port = int(explicit) if explicit else (443 if match.group("scheme").lower() == "https" else 80)
        endpoints.add(DiscoveredEndpoint(host=host, port=port))
    return endpoints


class EgressSource(ABC):
    """A source of egress endpoints the agent needs (workflow, code, manual hints, runtime logs…)."""

    @abstractmethod
    def discover(self) -> set[DiscoveredEndpoint]:
        """Return the endpoints this source can surface (empty set if none / not applicable)."""
