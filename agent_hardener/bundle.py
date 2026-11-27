# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble a run's hardened image bundle — the artifact a user can actually deploy.

A run's guardrails are a ``plugins.toml``, and that file alone is not adoptable: applying it to an
agent that lacks the plugin implementing its kind does not warn, it raises
(``plugin component 'agent_hardener.pre_tool_verifier' is not registered``), so the agent will not start.

Adoption needs three things — the plugin code, a registration that runs before Relay's
``initialize()``, and the config — and a run already produces all three. They are simply in two
places: the staged build context (your Dockerfile plus the appended ``COPY``/``ENV`` and the shim)
and the run's final ``plugins.toml``. This module puts them together, so what leaves a war-game is a
buildable image rather than evidence.
"""

from __future__ import annotations

import shutil
import tomllib
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from agent_hardener.openshell.relay_victim import IMAGE_SHIM_PATH, SHIM_DIR
from agent_hardener.relay_plugin.config import (
    DEFAULT_JUDGE_API_KEY_ENV,
    DEFAULT_JUDGE_BASE_URL,
    JUDGE_BASE_URL_ENVS,
    PLUGIN_KIND,
    SAFETY_API_KEY_ENV,
    JudgeModel,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: Where the bundle's Dockerfile bakes the guardrails. Matches the path Relay discovers as system
#: policy, so the image needs no runtime upload — unlike the war-game, where the file is delivered
#: per round.
BUNDLE_PLUGINS_DEST = "/etc/nemo-relay/plugins.toml"

#: The sandbox half of the staged shim. Excluded from the bundle: it defaults ``aiohttp`` sessions to
#: ``trust_env=True`` so OpenShell's egress proxy is honoured, which is a silent behaviour change in
#: an image that is no longer running inside OpenShell.
_SANDBOX_SHIM_MARKER = "# --- OpenShell:"

_DOCKERFILE_SUFFIX = """
# --- Agent Hardener: the guardrails this war-game produced -----------------------------------
# Baked rather than uploaded: the war-game delivers this file per round, but a shipped image
# should carry its own policy. Relay discovers this path as system-layer config.
COPY plugins.toml {destination}
"""

_STAGED_SOURCE = """\
Built from your own image plus the guardrails the war-game produced. Nothing here was invented: the
Dockerfile is yours with two lines appended, and `plugins.toml` is the config the run validated.
"""

_PREBUILT_SOURCE = """\
Built from `{image}`, the pre-built image this run validated, plus the guardrails the war-game
produced. The Dockerfile layers `plugins.toml` onto that image and replaces its `sitecustomize.py`
with one that drops the OpenShell proxy shim, so the agent code is exactly what was tested.
"""

_PREBUILT_SHIM_OVERRIDE = f"COPY sitecustomize.py {IMAGE_SHIM_PATH}/sitecustomize.py\n"

_JUDGE_RUNTIME = """\
Each guarded tool call is one LLM round trip to the safety judge, so the container needs:

{needs}

**Without them the judge errors and every call is allowed**: the guardrail fails open, by design,
and says so only in the log.
"""

_SAFETY_KEY_NOTE = f"""
`{SAFETY_API_KEY_ENV}` is the name the run gave the judge's key inside the sandbox. Its value is the
key the run was given as `{SAFETY_API_KEY_ENV}`, or as `{DEFAULT_JUDGE_API_KEY_ENV}` when that was unset.
Setting only `{DEFAULT_JUDGE_API_KEY_ENV}` in the container is not enough.
"""

#: No base_url in plugins.toml: the judge resolves its endpoint inside the container with ``judge_endpoint()``.
_DEFAULT_ENDPOINT = (
    "the endpoint named by "
    + ", else ".join(f"`{env}`" for env in JUDGE_BASE_URL_ENVS)
    + f", else `{DEFAULT_JUDGE_BASE_URL}` (`plugins.toml` sets no `base_url`)"
)

_UNREAD_JUDGE_RUNTIME = """\
Each guarded tool call is one LLM round trip to the safety judge, so the container needs the key
named by `api_key_env` in `plugins.toml` and network reach to its `base_url`. **Without them the judge
errors and every call is allowed**: the guardrail fails open, by design, and says so only in the log.
"""

_NO_JUDGE_RUNTIME = """\
`plugins.toml` enables no guardrail component, so the image makes no safety-judge calls and needs no
judge key.
"""

_README = """\
# Hardened image — {agent}

{source}

```bash
docker build -t {agent}-hardened:{run_id} .
```

## Deploying it

**NeMo Platform** — same agent config, hardened image:

```bash
nemo agents deploy --agent {agent} --mode docker --image {agent}-hardened:{run_id}
```

`--mode docker` is required: `--image` is rejected in the default `subprocess` mode.

**Anywhere else** — run the image. The guardrails are baked in at `{destination}`.

## What it needs at runtime

{runtime}
## The two halves of the hardening

`plugins.toml` is the guardrails: which tools are guarded, and what counts as an attack. It travels
in the image and takes effect wherever the image runs.

`openshell-policy.yaml`, if present, is the other half — the sandbox policy the run hardened:
filesystem scope, and which processes may reach which hosts. It is **not** applied by `docker run`;
it is an OpenShell artifact, and running the image without it gives you a guarded agent in an
unconstrained sandbox. Apply it with `openshell sandbox create --policy openshell-policy.yaml`, or
port its allow-lists to whatever confines your production runtime.

## Editing the policy

`plugins.toml` is data. Change a threshold or an instruction, rebuild, redeploy — no code change.
Removing the `agent_hardener.pre_tool_verifier` component removes the guardrails; removing the plugin
package from the image while leaving the component in place will stop the agent from starting.

## One assumption this image makes

The guardrail registers through `PYTHONPATH`, declared as a Dockerfile `ENV`. Any runtime that
starts the container *without* honouring image `ENV` — as OpenShell's `sandbox exec` does — will
start this agent unguarded, and nothing will say so. If yours does that, export
`PYTHONPATH=/app/openshell-shims` explicitly.
"""


def write_hardened_bundle(
    *,
    build_root: Path | None,
    plugins_toml: Path,
    destination: Path,
    agent: str,
    run_id: str,
    policy: Path | None = None,
    image: str | None = None,
) -> Path:
    """Assemble the deployable bundle and return its directory.

    Args:
        build_root: The run's staged build context — the user's project plus the appended Dockerfile
            and ``openshell-shims/``. When ``image`` is given only its ``sitecustomize.py`` is used: that
            tree was re-staged but never built, so shipping it would package something the run did not test.
        plugins_toml: The run's final guardrail config.
        destination: Directory to write (replaced if it exists).
        agent: Agent name, for the image tag in the README.
        run_id: The run this came from, so a hardened image is traceable to the evidence for it.
        policy: The run's final OpenShell sandbox policy, when there is one. Shipped beside the
            image because a war-game hardens two things, not one — the guardrails *and* the policy —
            and handing back only the half that fits in the image would understate the result.
        image: The pre-built image the run validated (``--victim-image``), when there was one. The
            bundle's Dockerfile then starts ``FROM`` it instead of rebuilding the staged context.

    Returns:
        The bundle directory.
    """
    if destination.exists():
        shutil.rmtree(destination)
    if image is None:
        if build_root is None:
            raise ValueError("a hardened bundle needs a staged build context or a pre-built image")
        shutil.copytree(build_root, destination, ignore=_build_context_ignore(build_root), symlinks=True)
        _strip_sandbox_shim(destination / SHIM_DIR / "sitecustomize.py")
        source = _STAGED_SOURCE
    else:
        destination.mkdir(parents=True)
        dockerfile_text = f"FROM {image}\n"
        staged_shim = build_root / SHIM_DIR / "sitecustomize.py" if build_root else None
        if staged_shim is not None and staged_shim.is_file():
            shutil.copyfile(staged_shim, destination / "sitecustomize.py")
            _strip_sandbox_shim(destination / "sitecustomize.py")
            dockerfile_text += _PREBUILT_SHIM_OVERRIDE
        (destination / "Dockerfile").write_text(dockerfile_text, encoding="utf-8")
        source = _PREBUILT_SOURCE.format(image=image)
    shutil.copyfile(plugins_toml, destination / "plugins.toml")

    if policy is not None and policy.is_file():
        shutil.copyfile(policy, destination / "openshell-policy.yaml")

    with (destination / "Dockerfile").open("a", encoding="utf-8") as dockerfile:
        dockerfile.write(_DOCKERFILE_SUFFIX.format(destination=BUNDLE_PLUGINS_DEST))
    (destination / "README.md").write_text(
        _README.format(
            agent=agent,
            run_id=run_id,
            destination=BUNDLE_PLUGINS_DEST,
            source=source,
            runtime=_judge_runtime(destination / "plugins.toml"),
        ),
        encoding="utf-8",
    )
    return destination


def _judge_runtime(plugins_toml: Path) -> str:
    """The README's runtime requirements, read from the ``plugins.toml`` the image will carry.

    Read from the file rather than recomputed from this process's environment: the judge's ``model`` table
    is written once, when the component is created, and may have been edited since. What the container
    reads is what the file says. Never raises: a malformed file still gets a bundle, with generic wording.
    """
    try:
        document = tomllib.loads(plugins_toml.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return _UNREAD_JUDGE_RUNTIME
    components = document.get("components")
    entries = [
        entry
        for entry in (components if isinstance(components, list) else [])
        if isinstance(entry, dict) and entry.get("kind") == PLUGIN_KIND and entry.get("enabled", True) is not False
    ]
    if not entries:
        return _NO_JUDGE_RUNTIME
    models = [model for entry in entries if (model := _judge_model(entry)) is not None]
    if len(models) != len(entries):
        return _UNREAD_JUDGE_RUNTIME
    keys = list(dict.fromkeys(model.api_key_env for model in models))
    endpoints = list(dict.fromkeys(f"`{model.base_url}`" if model.base_url else _DEFAULT_ENDPOINT for model in models))
    needs = [f"- `{key}`, set to the judge's API key" for key in keys]
    needs += [f"- network reach to {endpoint}" for endpoint in endpoints]
    note = _SAFETY_KEY_NOTE if SAFETY_API_KEY_ENV in keys else ""
    return _JUDGE_RUNTIME.format(needs="\n".join(needs)) + note


def _judge_model(entry: dict[str, Any]) -> JudgeModel | None:
    """The entry's judge, parsed the way the plugin parses it, so defaults match; ``None`` when invalid."""
    config = entry.get("config")
    try:
        return JudgeModel.model_validate(config.get("model") if isinstance(config, dict) else None)
    except ValidationError:
        return None


def _strip_sandbox_shim(sitecustomize: Path) -> None:
    """Drop the OpenShell half of the shim, keeping only the guardrail registration.

    The war-game image needs both; a shipped image needs one. Leaving the proxy shim in would change
    how the user's ``aiohttp`` sessions resolve proxies, in production, for no reason they asked for.
    """
    if not sitecustomize.is_file():
        return
    text = sitecustomize.read_text(encoding="utf-8")
    marker = text.find(_SANDBOX_SHIM_MARKER)
    if marker != -1:
        sitecustomize.write_text(text[:marker].rstrip() + "\n", encoding="utf-8")


_BYTECODE = shutil.ignore_patterns("__pycache__", "*.pyc")


def _build_context_ignore(build_root: Path) -> Callable[[str, list[str]], set[str]]:
    """A ``copytree`` filter that keeps exactly what ``docker build`` would send from ``build_root``.

    The bundle is the image the run built, so it carries what the build context carried. Copying the whole
    staged tree instead ships every file the project's ``.dockerignore`` keeps out of the image: one project's
    125 MB results folder landed in every run directory and filled the disk over a single campaign.
    """
    # Imported here, not at module level: the runner imports this module, and only the bundle needs docker.
    from docker.utils.build import exclude_paths  # noqa: PLC0415

    dockerignore = build_root / ".dockerignore"
    lines = dockerignore.read_text(encoding="utf-8").splitlines() if dockerignore.is_file() else []
    patterns = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    # Docker's own matcher, which always keeps the Dockerfile, as docker build always sends it.
    kept = {PurePath(path) for path in exclude_paths(str(build_root), patterns)}
    # A file re-included under an excluded directory comes back without its directory; copytree needs both.
    kept |= {parent for path in kept for parent in path.parents}
    kept_paths = {path.as_posix() for path in kept}

    def ignore(directory: str, names: list[str]) -> set[str]:
        relative = PurePath(directory).relative_to(build_root)
        dropped = {name for name in names if (relative / name).as_posix() not in kept_paths}
        return dropped | set(_BYTECODE(directory, names))

    return ignore
