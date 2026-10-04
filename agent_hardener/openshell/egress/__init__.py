# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic egress discovery for the NAT victim sandbox policy.

Sources surface the hosts an agent reaches (workflow YAML, project code, manual hints, or — later —
runtime deny-traces); :class:`EgressDiscoverer` turns them into valid OpenShell host patterns that
:mod:`agent_hardener.openshell.policy_egress` injects into the sandbox policy.
"""

from __future__ import annotations

from agent_hardener.openshell.egress.base import DiscoveredEndpoint, EgressSource, PolicyEndpoint, parse_urls
from agent_hardener.openshell.egress.discoverer import EgressDiscoverer, host_pattern, resolve_private_ips
from agent_hardener.openshell.egress.sources import (
    CodeEgressSource,
    DockerfileEnvEgressSource,
    ManualEgressSource,
    OpenShellLogEgressSource,
)

__all__ = [
    "CodeEgressSource",
    "DiscoveredEndpoint",
    "DockerfileEnvEgressSource",
    "EgressDiscoverer",
    "EgressSource",
    "ManualEgressSource",
    "OpenShellLogEgressSource",
    "PolicyEndpoint",
    "host_pattern",
    "parse_urls",
    "resolve_private_ips",
]
