# Third-Party Licenses

Agent Hardener is licensed under Apache-2.0 (see `LICENSE`). It does not vendor or
redistribute third-party source code; the packages below are its **runtime dependency
closure**, resolved at install time from PyPI, and each remains under its own licence.

Generated for **nvidia-agent-hardener 0.0.14** on 2026-10-05 from the
resolved runtime closure (`uv export --no-dev`). Development-only dependencies are excluded from
the distributed package and are not listed.

Every dependency is under a permissive or weak-copyleft licence. **There are no strong copyleft
(GPL / LGPL / AGPL) dependencies.**

| Package | Version | License |
|---|---|---|
| `annotated-doc` | 0.0.4 | MIT |
| `annotated-types` | 0.7.0 | MIT |
| `anyio` | 4.15.1 | MIT |
| `backoff` | 2.2.1 | MIT |
| `certifi` | 2026.4.22 | MPL-2.0 |
| `charset-normalizer` | 3.4.7 | MIT |
| `click` | 8.4.1 | BSD-3-Clause |
| `colorama` | 0.4.6 | BSD-3-Clause |
| `distro` | 1.9.0 | Apache-2.0 |
| `fastapi` | 0.136.1 | MIT |
| `googleapis-common-protos` | 1.74.0 | Apache-2.0 |
| `h11` | 0.16.0 | MIT |
| `httpcore` | 1.0.9 | BSD-3-Clause |
| `httpx` | 0.28.1 | BSD-3-Clause |
| `idna` | 3.13 | BSD-3-Clause |
| `importlib-metadata` | 8.5.0 | Apache-2.0 |
| `jinja2` | 3.1.6 | BSD-3-Clause |
| `jiter` | 0.14.0 | MIT |
| `jsonpatch` | 1.33 | BSD-3-Clause |
| `jsonpointer` | 3.1.1 | BSD-3-Clause |
| `langchain-core` | 1.4.7 | MIT |
| `langchain-openai` | 1.1.9 | MIT |
| `langchain-protocol` | 0.0.15 | MIT |
| `langfuse` | 4.7.1 | MIT |
| `langgraph` | 1.2.5 | MIT |
| `langgraph-checkpoint` | 4.1.1 | MIT |
| `langgraph-prebuilt` | 1.1.0 | MIT |
| `langgraph-sdk` | 0.4.2 | MIT |
| `langsmith` | 0.8.0 | MIT |
| `linkify-it-py` | 2.1.0 | MIT |
| `markdown-it-py` | 4.0.0 | MIT |
| `markupsafe` | 3.0.3 | BSD-3-Clause |
| `mdit-py-plugins` | 0.6.1 | MIT |
| `mdurl` | 0.1.2 | MIT |
| `nemo-relay` | 0.9.0rc1 | Apache-2.0 |
| `openai` | 2.41.1 | Apache-2.0 |
| `opentelemetry-api` | 1.41.1 | Apache-2.0 |
| `opentelemetry-exporter-otlp-proto-common` | 1.41.1 | Apache-2.0 |
| `opentelemetry-exporter-otlp-proto-http` | 1.41.1 | Apache-2.0 |
| `opentelemetry-proto` | 1.41.1 | Apache-2.0 |
| `opentelemetry-sdk` | 1.41.1 | Apache-2.0 |
| `opentelemetry-semantic-conventions` | 0.62b1 | Apache-2.0 |
| `orjson` | 3.11.8 | MPL-2.0 AND (Apache-2.0 OR MIT) |
| `ormsgpack` | 1.12.2 | Apache-2.0 OR MIT |
| `packaging` | 26.2 | Apache-2.0 OR BSD-2-Clause |
| `platformdirs` | 4.9.6 | MIT |
| `prompt-toolkit` | 3.0.52 | BSD-3-Clause |
| `protobuf` | 6.33.6 | BSD-3-Clause |
| `pydantic` | 2.12.5 | MIT |
| `pydantic-core` | 2.41.5 | MIT |
| `pygments` | 2.20.0 | BSD-2-Clause |
| `pyyaml` | 6.0.3 | MIT |
| `questionary` | 2.1.1 | MIT |
| `regex` | 2026.4.4 | Apache-2.0 AND CNRI-Python |
| `requests` | 2.33.1 | Apache-2.0 |
| `requests-toolbelt` | 1.0.0 | Apache-2.0 |
| `rich` | 14.3.4 | MIT |
| `ruamel-yaml` | 0.19.1 | MIT |
| `shellingham` | 1.5.4 | ISC |
| `sniffio` | 1.3.1 | MIT OR Apache-2.0 |
| `starlette` | 1.6.0 | BSD-3-Clause |
| `tenacity` | 9.1.4 | Apache-2.0 |
| `textual` | 8.2.7 | MIT |
| `tiktoken` | 0.12.0 | MIT |
| `tomli-w` | 1.2.0 | MIT |
| `tqdm` | 4.67.3 | MPL-2.0 AND MIT |
| `typer` | 0.23.1 | MIT |
| `typing-extensions` | 4.16.0 | PSF-2.0 |
| `typing-inspection` | 0.4.2 | MIT |
| `uc-micro-py` | 2.0.0 | MIT |
| `urllib3` | 2.8.0 | MIT |
| `uuid-utils` | 0.14.1 | BSD-3-Clause |
| `uvicorn` | 0.47.0 | BSD-3-Clause |
| `wcwidth` | 0.7.0 | MIT |
| `websockets` | 15.0.1 | BSD-3-Clause |
| `wrapt` | 1.17.3 | BSD-3-Clause |
| `xxhash` | 3.7.0 | BSD-3-Clause |
| `zipp` | 3.23.1 | MIT |
| `zstandard` | 0.25.0 | BSD-3-Clause |

## Notes

- **MPL-2.0 components** (`certifi`, `orjson`, `tqdm`) are file-level weak copyleft. They are
  consumed as unmodified PyPI packages; no MPL-licensed file is modified or redistributed in source
  form by this project, so no source-disclosure obligation is triggered.
- `regex` is Apache-2.0 AND CNRI-Python; `typing-extensions` and `pywin32` are PSF-2.0. All are
  permissive and compatible with Apache-2.0 redistribution.
- `tiktoken` publishes no licence metadata on PyPI; its licence was read from the upstream
  repository (MIT).
- `pywin32` and `colorama` are Windows-only conditional dependencies, not installed on Linux or macOS.
- **garak is deliberately not a dependency.** Agent Hardener provisions it into a separate virtual
  environment at `agent-hardener setup` time and invokes its CLI as a subprocess, so it is not part
  of this closure. See `agent_hardener/garak_venv.py`.
