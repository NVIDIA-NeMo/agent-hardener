# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt loader: parse YAML frontmatter and render Jinja2 markdown body.

Every agent prompt under ``<agent>/prompts/*.md`` follows the format::

    ---
    name: <agent>.<kind>
    version: <int>
    author: <email>
    last_updated: <YYYY-MM-DD>
    inputs:
      - <variable_name>
    ---

    Markdown body with {{ variable_name }} placeholders and optional
    {% if %} / {% for %} control flow.

The loader validates that every variable the body references is declared under
``inputs``, and that every declared input is provided at render time.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import yaml
from jinja2 import StrictUndefined, Template

if TYPE_CHECKING:
    from pathlib import Path


class PromptFormatError(ValueError):
    """Raised when a prompt file is malformed (bad frontmatter, missing inputs)."""


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    try:
        _, frontmatter, body = text.split("---", 2)
    except ValueError as exc:
        msg = "prompt file has opening '---' but no closing '---' for frontmatter"
        raise PromptFormatError(msg) from exc
    meta = yaml.safe_load(frontmatter) or {}
    if not isinstance(meta, dict):
        msg = f"prompt frontmatter must be a mapping, got {type(meta).__name__}"
        raise PromptFormatError(msg)
    return meta, body.lstrip("\n")


@lru_cache(maxsize=256)
def _read(path: Path) -> tuple[dict[str, Any], str]:
    """Read and parse a prompt file, caching the result by path."""
    text = path.read_text(encoding="utf-8")
    return _split_frontmatter(text)


def load_prompt(path: Path) -> str:
    """Return the un-rendered body of a prompt file.

    Use for system prompts which by convention have ``inputs: []`` and no
    Jinja2 placeholders.
    """
    meta, body = _read(path)
    declared = meta.get("inputs") or []
    if declared:
        msg = (
            f"prompt {path.name} declares inputs {declared} but was loaded "
            "with load_prompt() (no rendering). Use render_prompt() instead."
        )
        raise PromptFormatError(msg)
    return body


def render_prompt(path: Path, variables: dict[str, Any]) -> str:
    """Render a Jinja2-templated prompt body.

    Args:
        path: Absolute path to the prompt file.
        variables: Mapping of variable name to value.

    Returns:
        Rendered string with frontmatter stripped.

    Raises:
        PromptFormatError: If declared ``inputs`` are missing from ``variables``
            or if the caller provides keys that are not declared in ``inputs``.
    """
    meta, body = _read(path)
    declared = set(meta.get("inputs") or [])
    provided = set(variables.keys())
    missing = declared - provided
    if missing:
        msg = f"prompt {path.name} declares inputs {sorted(declared)} but missing {sorted(missing)} at render time"
        raise PromptFormatError(msg)
    extras = provided - declared
    if extras:
        msg = (
            f"prompt {path.name} declares inputs {sorted(declared)} but caller "
            f"provided undeclared keys {sorted(extras)}"
        )
        raise PromptFormatError(msg)
    template = Template(body, undefined=StrictUndefined)
    return template.render(**variables)
