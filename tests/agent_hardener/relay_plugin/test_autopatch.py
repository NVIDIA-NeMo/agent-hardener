# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for registering the guardrail kind without touching the victim's code.

Registration is the only thing that cannot be configuration: ``initialize()`` matches a discovered
component against the kinds registered so far, so a late registration is silently inert.
"""

from __future__ import annotations

import pytest

from agent_hardener.relay_plugin import autopatch
from agent_hardener.relay_plugin.config import PLUGIN_KIND

nemo_relay = pytest.importorskip("nemo_relay")


@pytest.fixture
def clean_registry() -> None:
    """Leave Relay's process-global registry as we found it.

    The kind registry is the only process-global state these tests touch: they register a kind but
    never activate components, and Relay 0.9 moved active components onto an owned activation
    handle. A leaked kind is not contained — Relay raises on re-registration, so it fails whatever
    test runs next rather than this one.
    """
    yield
    if PLUGIN_KIND in nemo_relay.plugin.list_kinds():
        nemo_relay.plugin.deregister(PLUGIN_KIND)


def test_install_registers_the_guardrail_kind(clean_registry: None) -> None:
    """A component naming an unregistered kind is inert, and Relay reports no error for it."""
    del clean_registry

    assert autopatch.install()
    assert PLUGIN_KIND in nemo_relay.plugin.list_kinds()


def test_install_is_idempotent(clean_registry: None) -> None:
    """Relay raises on re-registration, and a shim may be imported more than once."""
    del clean_registry
    autopatch.install()

    assert autopatch.install()


def test_install_reports_failure_rather_than_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """A victim that fails to start is worse than one that starts uninstrumented.

    The relay preflight reports the second clearly; the first just looks like a broken image.
    """
    monkeypatch.setattr(nemo_relay.plugin, "list_kinds", lambda: (_ for _ in ()).throw(RuntimeError("relay is angry")))

    assert autopatch.install() is False
