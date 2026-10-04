# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load/dump the policy YAML that ``current_policy`` / ``new_policy_yaml`` carry as text.

Uses ``ruamel.yaml`` rather than PyYAML (v1's choice) because ``new_policy_yaml`` goes to a
human reviewer: ruamel's round-trip mode preserves block style and doesn't silently re-sort
mapping keys the way ``yaml.dump(..., default_flow_style=False)`` can, so successive rounds'
diffs stay readable.
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING, Any

from ruamel.yaml import YAML

from agent_hardener.models.contracts import CURRENT_POLICY_KEY, DefenderInput

from ..errors import PolicyParseError
from .schema import Policy, dump_policy, parse_policy

if TYPE_CHECKING:
    from pathlib import Path


def _make_yaml() -> YAML:
    """Build a fresh ``YAML`` instance.

    ``YAML`` is stateful (parser/composer/scanner) and not thread-safe: defenders run
    concurrently via ``asyncio.to_thread``, so a shared module-level instance corrupts
    itself under concurrent ``.load()``/``.dump()`` calls. Construct one per call instead.
    """
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    return yaml


def load_policy_from_yaml(text: str) -> Policy:
    """Parse policy YAML text into a typed :class:`Policy`."""
    if not text or not text.strip():
        return Policy()
    try:
        raw = _make_yaml().load(StringIO(text)) or {}
    except Exception as exc:
        raise PolicyParseError(f"could not parse policy YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PolicyParseError("policy YAML did not parse to a mapping")
    try:
        return parse_policy(raw)
    except Exception as exc:
        raise PolicyParseError(f"policy YAML failed schema validation: {exc}") from exc


def load_policy(path: Path) -> Policy:
    """Parse a policy YAML file into a typed :class:`Policy`."""
    return load_policy_from_yaml(path.read_text())


def dump_policy_yaml(policy: Policy) -> str:
    """Serialize a :class:`Policy` back to YAML text."""
    stream = StringIO()
    _make_yaml().dump(dump_policy(policy), stream)
    return stream.getvalue()


def resolve_policy_path(defender_input: DefenderInput) -> Path | None:
    """Best-effort resolution of where the *current* policy came from, for diagnostics only.

    The actual policy text always comes from ``DefenderInput.context[CURRENT_POLICY_KEY]`` (see
    ``_build_context``); ``relay_plugins_path`` is informational and may point at a directory
    rather than the policy file itself, so callers must not depend on this for correctness.
    """
    if defender_input.relay_plugins_path is not None:
        return defender_input.relay_plugins_path
    return None


def current_policy_text(defender_input: DefenderInput) -> str:
    """The ``current_policy`` YAML text carried in ``DefenderInput.context``, or ``""``."""
    value: Any = defender_input.context.get(CURRENT_POLICY_KEY, "")
    return value if isinstance(value, str) else ""
