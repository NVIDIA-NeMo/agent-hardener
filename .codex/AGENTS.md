# Agent Hardener

## Project Context
- Task runner: `just` — run `just --list` for all commands
- Python environment: `uv`
- Package: `agent_hardener` (Python 3.11+)
- Do not create documentation files unless explicitly requested
- Planning artifacts under `.plans/` are exempt and may be created for implementation plans

## Code Style
- Ruff handles linting and formatting (line length: 120)
- Type hints required for function parameters and return values
- PEP 8 naming conventions; Google-style docstrings
- Lint: `just lint` (auto-fix) / `just lint-check` (check-only)
- Format: `just format` / `just format-check`

## Testing
- pytest with markers: `slow`, `integration`, `unit`
- Run: `just test` / `just test-coverage`
- 100% coverage target on new code
- Keep tests minimal — do not bloat

## Git Workflow
- Branch naming: `<descriptive-name>/<username>` in kebab case; no issue-tracker IDs
- Pre-commit hooks enforce quality on every commit (`just hooks`)

## Dependencies
- Add: `uv add <package>` (always pin versions)

## Plan Mode
- Conduct a thorough interview before finalizing the plan — ask about everything, leave nothing to assumption
- Ask a bunch of focused questions per round across multiple rounds until all ambiguity is resolved
- Mark one option as "(Recommended)" only when one genuinely stands out
- When appropriate, include a "Let Codex decide" option so the user can delegate
- Prefer asking over assuming — only write the plan once the user confirms nothing else to clarify
