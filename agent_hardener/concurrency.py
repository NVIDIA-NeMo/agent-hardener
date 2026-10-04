# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded concurrency helpers for deterministic agent fan-out."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_CONCURRENCY = 2


async def gather_limited_ordered(
    items: Sequence[T],
    limit: int,
    worker: Callable[[T], Awaitable[R]],
    *,
    return_exceptions: bool = False,
) -> list[R | BaseException]:
    """Run async work with a concurrency cap and return results in input order."""
    if not items:
        return []
    effective_limit = max(1, int(limit))
    semaphore = asyncio.Semaphore(effective_limit)

    async def run_one(item: T) -> R | BaseException:
        async with semaphore:
            try:
                return await worker(item)
            except BaseException as exc:
                if return_exceptions:
                    return exc
                raise

    return list(await asyncio.gather(*(run_one(item) for item in items)))


def concurrency_limit(
    configured: int | None,
    env_name: str | None = None,
    *,
    default: int = DEFAULT_CONCURRENCY,
) -> int:
    """Return a positive concurrency limit, with an optional environment override."""
    if env_name:
        env_value = os.getenv(env_name)
        if env_value:
            parsed_env = _maybe_int(env_value)
            if parsed_env is not None and parsed_env >= 1:
                return parsed_env
    if configured is not None:
        return _positive_int(configured, default)
    return max(1, default)


def agent_concurrency(
    agent_config: dict[str, object],
    context: dict[str, object],
    key: str,
    env_name: str,
    *,
    default: int = DEFAULT_CONCURRENCY,
) -> int:
    """Resolve an agent item-concurrency limit from env, agent config, then context."""
    configured = None
    nested = agent_config.get("concurrency")
    if isinstance(nested, dict):
        raw = nested.get(key)
        if raw is not None:
            configured = _maybe_int(raw)
    if configured is None:
        raw = agent_config.get(f"{key}_concurrency")
        if raw is not None:
            configured = _maybe_int(raw)
    if configured is None:
        value = context.get("run_concurrency")
        if isinstance(value, dict):
            configured = _maybe_int(value.get(key))
    return concurrency_limit(configured, env_name, default=default)


def _positive_int(value: object, default: int) -> int:
    parsed = _maybe_int(value)
    if parsed is None or parsed < 1:
        return max(1, default)
    return parsed


def _maybe_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
