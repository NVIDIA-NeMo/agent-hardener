# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import pytest
from pydantic import ValidationError

from agent_hardener.config import load_config, parse_config_data
from agent_hardener.ids import format_round_id, generate_round_id, normalize_agent_name, stable_config_json
from agent_hardener.models import AgentConfig, RunSettings, SessionConfig, TargetInput

if TYPE_CHECKING:
    from pathlib import Path


def test_agent_ids_are_deterministic_and_include_normalized_name() -> None:
    agent = AgentConfig(
        name="Primary Attacker!",
        role="attacker",
        service_url="http://agent.local",
        timeout_seconds=5,
        implementation="package.agent",
        config={"level": "smoke"},
    )
    same_agent = AgentConfig.model_validate(json.loads(agent.model_dump_json()))
    changed_agent = agent.model_copy(update={"config": {"level": "full"}})

    assert agent.agent_id == same_agent.agent_id
    assert agent.agent_id != changed_agent.agent_id
    assert stable_config_json(agent) == (
        '{"config":{"level":"smoke"},"implementation":"package.agent","name":"Primary Attacker!",'
        '"role":"attacker","service_url":"http://agent.local","timeout_seconds":5.0}'
    )


def test_model_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        TargetInput(name="")
    with pytest.raises(ValidationError, match="agent name must not be blank"):
        AgentConfig(name=" ", role="attacker")
    with pytest.raises(ValidationError):
        AgentConfig(name="agent", role="attacker", timeout_seconds=0)
    with pytest.raises(ValidationError):
        AgentConfig(name="validator", role="validator", config={"kind": "other"})
    with pytest.raises(ValidationError):
        RunSettings(rounds=0)
    with pytest.raises(ValidationError):
        RunSettings(concurrency={"defenders": 0})


def test_yaml_config_loading_adds_section_roles_and_resolves_storage(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
storage:
  root_dir: missions
run:
  retry_limit: 1
  rounds: 1
target:
  name: target
attackers:
  - name: attacker
    external_implementation: tests.agent
defenders:
  - name: defender
    capabilities: test defender capabilities
victim:
  name: victim
attack_validators:
  - name: attack-validator
benign_validators:
  - name: benign-validator
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.storage.root_dir == tmp_path / "missions"
    assert config.attackers[0].role == "attacker"
    assert config.attackers[0].implementation == "tests.agent"
    assert config.defenders[0].role == "defender"
    assert config.victim.role == "victim"
    assert config.attack_validators[0].config["kind"] == "attack"
    assert config.benign_validators[0].config["kind"] == "benign"
    assert config.run.concurrency.defenders == 2


def test_run_concurrency_config_is_loaded() -> None:
    config = parse_config_data(
        {
            "storage": "missions",
            "run": {
                "concurrency": {
                    "defenders": 3,
                    "validators": 4,
                    "guardrails_findings": 5,
                    "openshell_policy_findings": 6,
                    "benign_validator_rows": 7,
                    "attack_validator_hits": 8,
                }
            },
            "target": {"name": "target"},
            "victim": {"name": "victim"},
        }
    )

    assert config.run.concurrency.defenders == 3
    assert config.run.concurrency.validators == 4
    assert config.run.concurrency.guardrails_findings == 5
    assert config.run.concurrency.openshell_policy_findings == 6
    assert config.run.concurrency.benign_validator_rows == 7
    assert config.run.concurrency.attack_validator_hits == 8


def test_parse_config_accepts_absolute_storage_without_rewriting(tmp_path: Path) -> None:
    config = parse_config_data(
        {
            "storage": {"root_dir": str(tmp_path)},
            "target": {"name": "target"},
            "victim": {"name": "victim"},
        },
    )

    assert config.storage.root_dir == tmp_path


def test_json_config_loading_and_validation_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "storage_dir": "missions",
                "target": {"name": "target"},
                "attackers": [],
                "defenders": [],
                "victim": {"name": "victim"},
                "attack_validators": [],
                "benign_validators": [],
            },
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, SessionConfig)
    assert config.storage.root_dir == tmp_path / "missions"

    unsupported_path = tmp_path / "config.toml"
    unsupported_path.write_text("storage = 'missions'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported config file type"):
        load_config(unsupported_path)
    non_mapping_path = tmp_path / "config.yaml"
    non_mapping_path.write_text("- bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config root must be a mapping"):
        load_config(non_mapping_path)
    with pytest.raises(ValueError, match="config root must be a mapping"):
        parse_config_data([], base_dir=tmp_path)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="agents must be a list"):
        parse_config_data(
            {
                "storage": "missions",
                "target": {"name": "target"},
                "attackers": {"name": "bad"},
                "victim": {"name": "victim"},
            },
            base_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="agent config entries must be mappings"):
        parse_config_data(
            {
                "storage": "missions",
                "target": {"name": "target"},
                "attackers": ["bad"],
                "victim": {"name": "victim"},
            },
            base_dir=tmp_path,
        )


def test_session_config_rejects_wrong_roles_and_validator_kinds(tmp_path: Path) -> None:
    base = {
        "storage": str(tmp_path),
        "target": {"name": "target"},
        "victim": {"name": "victim"},
    }

    with pytest.raises(ValidationError, match="victim must have role 'victim'"):
        parse_config_data({**base, "victim": {"name": "victim", "role": "attacker"}})
    with pytest.raises(ValidationError, match="attack_validators must have config kind 'attack'"):
        parse_config_data({**base, "attack_validators": [{"name": "validator", "config": {"kind": "benign"}}]})
    with pytest.raises(ValidationError, match="benign_validators must have config kind 'benign'"):
        parse_config_data({**base, "benign_validators": [{"name": "validator", "config": {"kind": "attack"}}]})
    with pytest.raises(ValidationError, match="attackers entries must have role 'attacker'"):
        parse_config_data({**base, "attackers": [{"name": "bad", "role": "defender", "capabilities": "x"}]})


def test_session_id_generation_uses_utc_timestamp_and_short_uuid() -> None:
    round_id = generate_round_id(
        datetime(2026, 5, 2, 12, 34, 56, tzinfo=UTC),
        UUID("12345678-1234-5678-1234-567812345678"),
    )

    assert round_id == "20260502T123456Z-12345678"
    assert (
        generate_round_id(
            datetime(2026, 5, 2, 12, 34, 56, tzinfo=UTC).replace(tzinfo=None),
            UUID("12345678-1234-5678-1234-567812345678"),
        )
        == "20260502T123456Z-12345678"
    )
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", generate_round_id())


def test_mission_session_id_generation_is_sequential_and_sortable() -> None:
    assert format_round_id(1) == "round-0001"
    assert format_round_id(12) == "round-0012"
    assert format_round_id(12_345) == "round-12345"
    with pytest.raises(ValueError, match="round_number must be at least 1"):
        format_round_id(0)


def test_normalize_agent_name_has_safe_fallback() -> None:
    assert normalize_agent_name("!!!") == "agent"
