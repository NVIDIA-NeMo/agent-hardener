# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only lookups over a parsed :class:`Policy`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Endpoint, NetworkPolicyEntry, Policy


def find_entry_for(
    policy: Policy, host: str, port: int | None, binary: str | None
) -> tuple[str, NetworkPolicyEntry] | None:
    """Find the ``network_policies`` entry (key, entry) that serves ``host``/``port``.

    If ``binary`` is given, only matches an entry that also lists it among its binaries.
    """
    for key, entry in policy.network_policies.items():
        # Only filter by binary when it's an absolute path — bare names like "curl" or
        # "api_caller" (as extracted from prompt text) can't be matched against policy
        # paths and would incorrectly eliminate all entries.
        if (
            binary is not None
            and binary.startswith("/")
            and entry.binaries is not None
            and not any(b.path == binary for b in entry.binaries)
        ):
            continue
        for endpoint in entry.endpoints:
            if endpoint.host == host and (port is None or endpoint.port is None or endpoint.port == port):
                return key, entry
    return None


def find_endpoint(entry: NetworkPolicyEntry, host: str, port: int | None, path: str | None = None) -> Endpoint | None:
    """Find the specific endpoint within ``entry`` matching ``host``/``port``."""
    for endpoint in entry.endpoints:
        if endpoint.host == host and (port is None or endpoint.port is None or endpoint.port == port):
            return endpoint
    return None


def endpoint_is_inspectable(endpoint: Endpoint) -> bool:
    """``False`` when the endpoint is L4 passthrough (``tls: skip``) or has no L7 protocol.

    Nothing at the request-tuple level to match on beyond host/port in that case.
    """
    if endpoint.tls == "skip":
        return False
    return bool(endpoint.protocol)
