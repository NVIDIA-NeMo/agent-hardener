# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate egress sources and map discovered hosts to policy-ready endpoints.

Beyond mapping hostnames to valid OpenShell host patterns, this resolves each host: when it resolves
to a private/internal IP, that IP is added to the endpoint's ``allowed_ips`` so OpenShell's anti-SSRF
guard (which blocks internal addresses by default) permits the agent's own declared dependencies.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import TYPE_CHECKING

from agent_hardener.openshell.egress.base import PolicyEndpoint

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from agent_hardener.openshell.egress.base import DiscoveredEndpoint, EgressSource


def host_pattern(host: str) -> str:
    """Convert a hostname to a host pattern OpenShell accepts.

    OpenShell rejects a bare ``*`` and TLD wildcards (``*.com``). For 3+ labels we use a subdomain
    wildcard over the parent (``integrate.api.nvidia.com`` → ``*.api.nvidia.com``); for a 2-label
    registrable domain we keep it exact (``tavily.com``) so we never emit an illegal TLD wildcard.
    """
    labels = host.split(".")
    if len(labels) >= 3:
        return "*." + ".".join(labels[1:])
    return host


def resolve_private_ips(host: str) -> list[str]:
    """Resolve ``host`` and return any private/internal IPs it maps to (empty on failure).

    These become SSRF-exemption ``allowed_ips`` — without them OpenShell blocks internal addresses.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return []
    ips = {info[4][0] for info in infos}
    private: list[str] = []
    for ip in ips:
        try:
            if ipaddress.ip_address(ip).is_private:
                private.append(ip)
        except ValueError:
            continue
    return sorted(private)


class EgressDiscoverer:
    """Combine :class:`EgressSource` results into policy-ready, de-duplicated endpoints."""

    def __init__(
        self,
        sources: Iterable[EgressSource],
        resolver: Callable[[str], list[str]] = resolve_private_ips,
    ) -> None:
        self.sources = list(sources)
        self._resolver = resolver

    def endpoints(self) -> set[DiscoveredEndpoint]:
        """Union the raw (concrete host, port) endpoints surfaced by every source."""
        found: set[DiscoveredEndpoint] = set()
        for source in self.sources:
            found |= source.discover()
        return found

    def policy_endpoints(self) -> list[PolicyEndpoint]:
        """Map discovered hosts to sorted ``PolicyEndpoint``s (pattern, port, SSRF allowed_ips).

        Hosts that collapse to the same ``(pattern, port)`` are merged, unioning the private IPs
        each concrete host resolves to.
        """
        allowed: dict[tuple[str, int], set[str]] = {}
        for endpoint in self.endpoints():
            key = (host_pattern(endpoint.host), endpoint.port)
            allowed.setdefault(key, set()).update(self._resolver(endpoint.host))
        return [
            PolicyEndpoint(host=pattern, port=port, allowed_ips=tuple(sorted(ips)))
            for (pattern, port), ips in sorted(allowed.items())
        ]
