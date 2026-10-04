# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Location, version, provisioning, and resolution for the dedicated garak venv.

garak is intentionally NOT an agent-hardener dependency: it pins ``litellm`` (``httpx>=0.28``) and
``torch``, which conflict with ``nvidia-nat`` and would bloat the venv. agent-hardener never imports
garak — it spawns the garak CLI from a separate venv. This module is the single source of truth
for that venv's location and the garak version, shared by ``agent-hardener setup`` (which provisions
it) and the agent_breaker attacker (which resolves the interpreter to spawn).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from agent_hardener.env import GARAK_ENV_PASSTHROUGH, GARAK_REQUIRED_API_KEYS, NIM_API_KEY, inference_api_key

# Interpreter override: point agent-hardener at an existing garak venv (e.g. a NeMo Platform job's
# managed location). When unset, both provisioning and resolution use the default below.
GARAK_PYTHON_ENVVAR = "AGENT_HARDENER_GARAK_PYTHON"
# Exact, not a range. `>=0.15.1,<0.16` gained nothing (0.15.1 is the only release in it) while leaving
# every `setup` free to pull a future 0.15.x the moment it is published — the cheap half of the
# supply-chain risk, since it needs a maintainer-account takeover rather than a registry compromise.
# Bumping this is a deliberate commit: check the upstream changelog before moving it.
GARAK_VERSION = "0.15.1"
GARAK_SPEC = f"garak=={GARAK_VERSION}"

# Installed by immutable URL with its hash, so a republished artifact under the same version fails
# the install instead of silently executing. This closes the TAVA T5 concern, which is garak's own
# release; the transitive closure is still version-resolved rather than hash-pinned, and full-closure
# pinning remains a separate item. Update all three constants together when bumping the version —
# take the values from https://pypi.org/pypi/garak/<version>/json.
GARAK_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/33/55/"
    "ec8a0083bea2238768444f05d62ab274bda6c1aac3d6d99dc54882e01880/garak-0.15.1-py3-none-any.whl"
)
GARAK_WHEEL_SHA256 = "c420e2f339662ace10b05dcd113e66dce7eaf918910c6128a2f4618dcac92431"  # pragma: allowlist secret
# PEP 508 direct reference: uv verifies the fragment hash before installing.
GARAK_PINNED_SPEC = f"garak @ {GARAK_WHEEL_URL}#sha256={GARAK_WHEEL_SHA256}"
# Pinned deliberately: garak pulls torch>=2.6.0, whose wheels are per-Python-version. Letting the
# interpreter float risks landing on a Python with no torch wheel (and litellm caps at <3.14).
# 3.12 sits inside garak's >=3.10 window with mature torch wheels; uv fetches it if the host lacks it.
GARAK_PYTHON_VERSION = "3.12"
_DEFAULT_GARAK_VENV = Path("~/.agent-hardener/garak-venv")


def default_garak_python() -> Path:
    """Default interpreter path inside the dedicated garak venv (``~/.agent-hardener/garak-venv``)."""
    return (_DEFAULT_GARAK_VENV / "bin" / "python").expanduser()


def resolve_garak_python() -> str:
    """Interpreter agent-hardener spawns garak with; ``AGENT_HARDENER_GARAK_PYTHON`` overrides the default."""
    override = os.environ.get(GARAK_PYTHON_ENVVAR)
    return override if override else str(default_garak_python())


def resolve_garak_venv_dir() -> Path:
    """The garak venv directory to provision — the parent of ``resolve_garak_python()``'s ``bin/``."""
    return Path(resolve_garak_python()).parent.parent


#: Non-credential variables forwarded verbatim into the garak subprocess. Everything outside this
#: list (and :data:`GARAK_REQUIRED_API_KEYS`) is dropped, so a compromised garak cannot read the
#: operator's unrelated secrets — cloud keys, GH_TOKEN, or INFERENCE_API_KEY itself, which reaches
#: garak only through the NIM_API_KEY alias below.
#: Kept deliberately short: a name here only has an effect when the operator actually has that
#: variable set, but every entry is one more thing a compromised garak can read. Anything with a
#: working fallback (USER, TZ, TMPDIR, HF_HOME) is intentionally absent.
_ENV_ALLOWLIST = (
    # Without these garak cannot find binaries or resolve its own config directory.
    "PATH",
    "HOME",
    # Encoding. With no locale set, Python falls back to ASCII and garak dies part-way through a run
    # writing an attack prompt containing non-ASCII — which jailbreak corpora routinely do.
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    # Corporate egress. Dropping these breaks garak on any proxied or TLS-intercepted network, which
    # is most NVIDIA hosts. Note a proxy URL may embed basic-auth credentials, so this group is the
    # one place the allow-list can still carry a secret.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    # garak depends on xdg-base-dirs; these only matter when the operator has customised them.
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_STATE_HOME",
)


def garak_subprocess_env() -> dict[str, str]:
    """Environment for any garak subprocess (the CLI attacker or the detector worker).

    Built as an **allow-list** rather than a copy of ``os.environ``: garak is third-party code that
    runs unsandboxed on the developer host, so forwarding the whole environment would hand every
    credential the operator happens to have exported to a dependency we do not control.

    - Garak's ``nim`` generator/detector authenticates to the NVIDIA inference gateway with
      ``NIM_API_KEY``; we mirror ``INFERENCE_API_KEY`` (the same credential the rest of agent-hardener
      uses) into it when it is not already set. ``INFERENCE_API_KEY`` itself is not forwarded.
    - Garak refuses to start unless every key in :data:`GARAK_REQUIRED_API_KEYS` is present, even for
      generators that never use them; we default the missing ones to ``"NOT_SET"`` so startup never
      depends on the ambient environment carrying placeholders.
    - :data:`_ENV_ALLOWLIST` carries the non-credential variables garak genuinely needs, notably proxy
      and CA-bundle settings. ``GARAK_*`` variables pass through so garak's own config still works,
      and ``AGENT_HARDENER_GARAK_ENV_PASSTHROUGH`` names anything else a given host requires.
    """
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    env.update({name: value for name, value in os.environ.items() if name.startswith("GARAK_")})

    extra = os.environ.get(GARAK_ENV_PASSTHROUGH, "")
    for name in (part.strip() for part in extra.split(",")):
        if name and name in os.environ:
            env[name] = os.environ[name]

    key = inference_api_key()
    nim_key = os.environ.get(NIM_API_KEY)
    if nim_key:
        env[NIM_API_KEY] = nim_key
    elif key:
        env[NIM_API_KEY] = key
    for var in GARAK_REQUIRED_API_KEYS:
        env.setdefault(var, "NOT_SET")
    return env


def _run(cmd: list[str]) -> None:
    """Run a provisioning subprocess, raising ``RuntimeError`` with the stderr tail on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({' '.join(cmd)}):\n{proc.stderr.strip()[-2000:]}")


def provision_garak_venv(*, force: bool = False) -> Path:
    """Create the dedicated garak venv and install garak into it via uv; return the interpreter path.

    Idempotent: when the interpreter already exists, returns it unchanged unless ``force`` is set.
    Honors ``AGENT_HARDENER_GARAK_PYTHON`` for the location (its ``bin/python`` parent is the venv dir).
    """
    if shutil.which("uv") is None:
        raise RuntimeError("uv not found — install it (https://docs.astral.sh/uv/) then retry.")

    venv_dir = resolve_garak_venv_dir()
    python = venv_dir / "bin" / "python"
    if python.exists() and not force:
        return python

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["uv", "venv", "--python", GARAK_PYTHON_VERSION, str(venv_dir)])
    _run(["uv", "pip", "install", "--python", str(python), GARAK_PINNED_SPEC])
    if not python.exists():
        raise RuntimeError(f"garak install finished but {python} is missing.")
    return python
