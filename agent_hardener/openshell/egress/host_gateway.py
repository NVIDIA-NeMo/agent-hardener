# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the sandbox->host bridge IP that ``host.docker.internal`` maps to.

Host-backend and telemetry endpoints reach services on the host through Docker's
``host.docker.internal`` alias. OpenShell's anti-SSRF guard blocks internal addresses unless they
are named in an endpoint's ``allowed_ips`` — so the policy must declare the concrete IP the alias
resolves to. That IP is **driver-specific**:

- Docker Desktop / native Linux (``--add-host=host-gateway``) → the bridge gateway ``172.17.0.1``.
- Colima (macOS Virtualization.Framework userspace network) → ``192.168.5.2``.

The alias only resolves *inside* a container, not on the host — and depending on the driver it lives
in the container's ``/etc/hosts`` (Docker Desktop) or only in the daemon's embedded DNS (Colima
serves it from ``192.168.5.1:53``, not ``/etc/hosts``). So we probe it with ``getent ahosts`` in a
throwaway container, which consults both, rather than hard-coding a value correct for only one driver.

``ahosts`` rather than ``hosts``: Docker Desktop publishes the alias dual-stack, and ``getent hosts``
returns only the preferred family — IPv6 — so the IPv4 address never appears in its output at all.
``ahosts`` lists every family, and *both* are allowlisted: the victim's HTTP client may connect over
either, and OpenShell denies whichever address it actually resolved if that one is missing.

Every Docker network's gateway is unioned in as well. The probe container and the sandbox can sit on
different networks, and on host-gateway drivers the alias maps to the gateway of whichever network
the caller is attached to — so probing alone can return an address the sandbox never sees.
"""

from __future__ import annotations

import ipaddress
import os
import subprocess
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

HOST_GATEWAY_ALIAS = "host.docker.internal"

# Docker Desktop / native-Linux default. Used as a fallback when in-container probing is unavailable
# (no Docker, offline, or the alias is absent) so the policy still has a sensible value.
FALLBACK_GATEWAY_IP = "172.17.0.1"

# Override the probe image for environments without registry access to the default. Needs `getent`
# (musl/glibc) — busybox lacks it — and a resolver that reaches the daemon's embedded DNS.
_PROBE_IMAGE_ENV = "AGENT_HARDENER_HOST_GATEWAY_PROBE_IMAGE"
_DEFAULT_PROBE_IMAGE = "alpine:latest"


@lru_cache(maxsize=1)
def resolve_host_gateway_ips(probe_image: str | None = None) -> tuple[str, ...]:
    """Every address ``host.docker.internal`` resolves to inside a container on this daemon.

    Runs ``getent ahosts host.docker.internal`` in a throwaway container (resolves via ``/etc/hosts``
    and the daemon's embedded DNS). All families are returned, IPv4 first: the alias is dual-stack on
    Docker Desktop and the victim's HTTP client may pick either, so the policy has to allow both.
    Falls back to :data:`FALLBACK_GATEWAY_IP` when Docker is unavailable, the probe fails, or the
    alias is absent. Cached: the daemon's bridge IPs are stable per process.
    """
    image = probe_image or os.environ.get(_PROBE_IMAGE_ENV) or _DEFAULT_PROBE_IMAGE
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", image, "getent", "ahosts", HOST_GATEWAY_ALIAS],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        resolved: tuple[str, ...] = ()
    else:
        resolved = _parse_getent_ips(result.stdout)
    # The probe container and the sandbox may sit on different networks, and on host-gateway drivers
    # the alias maps to whichever network's gateway the caller is attached to. Union in every
    # network's gateway so the allowlist covers the sandbox's own regardless of where it lands.
    combined = _dedupe_addresses([*resolved, *_docker_network_gateways()])
    return combined or (FALLBACK_GATEWAY_IP,)


def _docker_network_gateways() -> tuple[str, ...]:
    """Gateway address of every Docker network on this daemon (empty when Docker is unavailable)."""
    try:
        listed = subprocess.run(
            ["docker", "network", "ls", "-q"], capture_output=True, text=True, timeout=30, check=True
        )
        network_ids = listed.stdout.split()
        if not network_ids:
            return ()
        inspected = subprocess.run(
            ["docker", "network", "inspect", *network_ids, "--format", "{{range .IPAM.Config}}{{.Gateway}}\n{{end}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    return _dedupe_addresses(inspected.stdout.split())


def resolve_host_gateway_ip(probe_image: str | None = None) -> str:
    """The IPv4 address ``host.docker.internal`` resolves to, or :data:`FALLBACK_GATEWAY_IP`."""
    for ip in resolve_host_gateway_ips(probe_image):
        if ipaddress.ip_address(ip).version == 4:
            return ip
    return FALLBACK_GATEWAY_IP


def _parse_getent_ips(getent_output: str) -> tuple[str, ...]:
    """Addresses in ``getent ahosts`` output (``<ip>  <socktype>  <name>``), IPv4 first.

    ``ahosts`` repeats each address per socket type, so duplicates are collapsed. Non-address
    columns (``STREAM``/``DGRAM``, stray runtime output) fail to parse and are skipped.
    """
    return _dedupe_addresses(line.split()[0] for line in getent_output.splitlines() if line.split())


def _dedupe_addresses(candidates: Iterable[str]) -> tuple[str, ...]:
    """Parseable IP addresses from ``candidates``, de-duplicated and IPv4 first."""
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue  # not an address (socktype column, hostname, stray output)
        if address not in addresses:
            addresses.append(address)
    # IPv4 first so the single-address accessor and human-facing output favour it.
    return tuple(str(a) for a in sorted(addresses, key=lambda a: a.version))


def host_gateway_allowed_ips(probe_image: str | None = None) -> list[str]:
    """SSRF-exemption ``allowed_ips`` for ``host.docker.internal``, one CIDR per resolved address.

    Both families are listed. Allowing only IPv4 is not enough: on Docker Desktop the alias is
    dual-stack and the victim's client may connect over IPv6, which OpenShell then denies because the
    address it resolved is absent from the allowlist.
    """
    return [
        f"{ip}/{32 if ipaddress.ip_address(ip).version == 4 else 128}" for ip in resolve_host_gateway_ips(probe_image)
    ]
