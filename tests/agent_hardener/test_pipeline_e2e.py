# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent_hardener.agents.validators import garak_attack_replay, smart_benign
from agent_hardener.config import load_config
from agent_hardener.models import DefenderAnalysis, DefendersManagerOutput, ValidatorReport
from agent_hardener.openshell.lifecycle import OpenShellConfig, OpenShellLifecycle
from agent_hardener.runtime.adapters import build_uploaders
from agent_hardener.runtime.orchestrator import Orchestrator

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "pipeline_e2e"


class FakeOpenShellRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.created = False
        self.active_policy = "version: 1\n"

    def run(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        returncode = 0
        stdout = "ok"
        if args[1:3] == ["gateway", "info"]:
            stdout = "gateway ready"
        elif args[1:3] == ["policy", "set"]:
            self.active_policy = Path(args[args.index("--policy") + 1]).read_text(encoding="utf-8")
            stdout = "policy set"
        elif args[1:3] == ["policy", "get"] or (args[1:3] == ["sandbox", "get"] and "--policy-only" in args):
            stdout = self.active_policy
        elif args[1:3] == ["sandbox", "delete"]:
            self.created = False
            returncode = 1
            stdout = "NotFound: sandbox not found"
        elif args[1:3] == ["sandbox", "get"]:
            if self.created:
                stdout = "Ready"
            else:
                returncode = 1
                stdout = "NotFound: sandbox not found"
        elif args[1:3] == ["sandbox", "create"]:
            self.active_policy = Path(args[args.index("--policy") + 1]).read_text(encoding="utf-8")
            self.created = True
            stdout = "sandbox created"
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout)


class FakeDefendersManager:
    """Test double for the real defenders manager: returns canned patches at the manager's public seam.

    The real LangGraph defenders need live LLM calls; this fake stands in for them at the one boundary
    the orchestrator injects, so the mission pipeline (deploy → victim → validate → retry) runs for real
    without touching any ``agents/defenders/`` code. Per-defender patch generation is covered by the
    defenders' own unit tests.
    """

    def __init__(self, candidate_policy_path: Path, relay_plugins_path: Path) -> None:
        self.candidate_policy_path = candidate_policy_path
        self.relay_plugins_path = relay_plugins_path

    async def run(
        self, input_data: Any, run_defender_cb: Any, validation_feedback: Any = None
    ) -> DefendersManagerOutput:
        analyses: list[DefenderAnalysis] = []
        for agent in input_data.available_defenders:
            analyses.append(
                DefenderAnalysis(
                    agent_id=agent.agent_id,
                    agent_name=agent.name,
                    ok=True,
                    summary=f"{agent.name} hardened the victim",
                    policy_patches=self._patches_for(agent.name),
                )
            )
        return DefendersManagerOutput(
            ok=True,
            summary="canned defender patches",
            routed_agent_ids=[analysis.agent_id for analysis in analyses],
            analyses=analyses,
        )

    def _patches_for(self, agent_name: str) -> list[dict[str, Any]]:
        if "policy" in agent_name:
            return [
                {
                    "type": "openshell_policy_candidate",
                    "candidate_policy_path": str(self.candidate_policy_path),
                    "changed": True,
                }
            ]
        if "guardrails" in agent_name:
            return [
                {
                    "type": "relay_plugins_candidate",
                    "target_relay_plugins_path": str(self.relay_plugins_path),
                    "candidate_relay_plugins_path": str(self.relay_plugins_path),
                }
            ]
        return []


@dataclass(frozen=True)
class PipelineRun:
    run_dir: Path
    runner: FakeOpenShellRunner
    reports: list[Any]


def test_pipeline_e2e_runs_orchestrator_deploys_patches_validates_and_writes_report(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run = _run_pipeline_e2e(tmp_path, monkeypatch)
    report = _single_report(run)

    assert report.success is True
    assert report.round_id == "round-0001"
    assert report.mission_id == "pipeline-e2e"
    assert len(report.iterations) == 1
    _assert_successful_iteration(report.iterations[0])

    # The attack the fixture attacker emits is replayed against the (mock) defended victim.
    assert len(report.attacks) == 1
    assert report.attacks[0].agent_name == "mock-garak-agent-breaker"
    assert report.attacks[0].records[0]["source"] == "garak-agent-breaker"

    report_json = _load_artifact(report.storage_dir, "report.json")
    assert report_json["success"] is True
    assert len(report_json["iterations"]) == 1


def test_pipeline_e2e_retries_until_attack_validator_blocks_attack(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run = _run_pipeline_e2e(
        tmp_path,
        monkeypatch,
        retry_limit=1,
        attack_blocked_by_iteration=lambda iteration: iteration >= 2,
    )
    report = _single_report(run)

    assert report.success is True
    assert len(report.iterations) == 2

    first, second = report.iterations
    assert first.success is False
    first_validators = {validator.kind: validator for validator in first.validators}
    assert first_validators["attack"].ok is False
    assert first_validators["attack"].summary == "blocked 0/1 original Garak attack hits"
    assert first_validators["attack"].metadata["not_blocked_count"] == 1
    assert first_validators["benign"].ok is True

    assert second.success is True
    second_validators = {validator.kind: validator for validator in second.validators}
    assert second_validators["attack"].ok is True
    assert second_validators["attack"].metadata["blocked_count"] == 1
    assert second_validators["benign"].ok is True


def test_pipeline_e2e_reports_failure_after_retry_limit_is_exhausted(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run = _run_pipeline_e2e(
        tmp_path,
        monkeypatch,
        retry_limit=1,
        attack_blocked_by_iteration=lambda _iteration: False,
    )
    report = _single_report(run)

    assert report.success is False
    assert len(report.iterations) == 2
    for iteration in report.iterations:
        assert iteration.success is False
        validators_by_kind = {validator.kind: validator for validator in iteration.validators}
        assert validators_by_kind["attack"].ok is False
        assert validators_by_kind["attack"].metadata["blocked_count"] == 0
        assert validators_by_kind["attack"].metadata["not_blocked_count"] == 1
        assert validators_by_kind["benign"].ok is True

    report_json = _load_artifact(report.storage_dir, "report.json")
    assert report_json["success"] is False
    assert len(report_json["iterations"]) == 2


def test_pipeline_e2e_fixture_and_public_example_config_load() -> None:
    session_config_paths = [
        REPO_ROOT / "tests" / "fixtures" / "pipeline_e2e" / "pipeline.yaml",
        REPO_ROOT / "examples" / "agent-hardener.yaml",
    ]
    for config_path in session_config_paths:
        config = load_config(config_path)
        assert config.victim.role == "victim"
        assert all(agent.role == "attacker" for agent in config.attackers)
        assert all(agent.role == "defender" for agent in config.defenders)
        assert all(agent.config["kind"] == "attack" for agent in config.attack_validators)
        assert all(agent.config["kind"] == "benign" for agent in config.benign_validators)

    yaml_fixture_paths = [
        REPO_ROOT / "tests" / "fixtures" / "pipeline_e2e" / "policy-permissive.yaml",
        REPO_ROOT / "tests" / "fixtures" / "pipeline_e2e" / "relay-plugins.toml",
        REPO_ROOT / "agent_hardener" / "templates" / "relay-victim" / "openshell-policy-permissive.yaml",
        REPO_ROOT / "agent_hardener" / "templates" / "relay-victim" / "repair-profile.yaml",
    ]
    for yaml_path in yaml_fixture_paths:
        assert yaml.safe_load(yaml_path.read_text(encoding="utf-8"))


def _run_pipeline_e2e(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    retry_limit: int = 0,
    attack_blocked_by_iteration: Any = None,
) -> PipelineRun:
    run_dir = tmp_path / "pipeline_e2e"
    shutil.copytree(FIXTURE_DIR, run_dir)
    monkeypatch.chdir(run_dir)
    monkeypatch.syspath_prepend(str(run_dir))
    monkeypatch.setenv("INFERENCE_API_KEY", "pipeline-e2e-key")
    if retry_limit:
        config_path = run_dir / "pipeline.yaml"
        config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config_data["run"]["retry_limit"] = retry_limit
        config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    _install_pipeline_mocks(
        monkeypatch,
        attack_blocked_by_iteration=attack_blocked_by_iteration or (lambda _iteration: True),
    )
    config = load_config(run_dir / "pipeline.yaml")
    runner = FakeOpenShellRunner()
    openshell_config = OpenShellConfig.from_mapping(config.victim_control.config)
    lifecycle = OpenShellLifecycle(openshell_config, runner=runner)
    openshell_uploader, relay_uploader = build_uploaders(config.victim_control, lifecycle)

    # The fake manager stands in for the LLM defenders; its candidate policy is a real file the deploy
    # stage applies through the (real) OpenShell uploader against the fake CLI runner.
    candidate_policy = run_dir / "candidate-policy.yaml"
    candidate_policy.write_text("version: 1\nnetwork_policies: {}\n", encoding="utf-8")
    defenders_manager = FakeDefendersManager(candidate_policy, run_dir / "relay-plugins.toml")

    reports = asyncio.run(
        Orchestrator(
            config,
            openshell_uploader=openshell_uploader,
            relay_uploader=relay_uploader,
            defenders_manager=defenders_manager,
        ).run_rounds(mission_id="pipeline-e2e")
    )
    return PipelineRun(run_dir=run_dir, runner=runner, reports=reports)


def _install_pipeline_mocks(
    monkeypatch: Any,
    *,
    attack_blocked_by_iteration: Any,
) -> None:
    async def fake_attack_replay(
        prompt: str,
        request: Any,
        config: Any,
        client: Any,
    ) -> garak_attack_replay.ReplayResult:
        assert "bash_executor" in prompt
        if not attack_blocked_by_iteration(request.iteration):
            return garak_attack_replay.ReplayResult(
                ok=True,
                status_code=200,
                text="GITHUB_TOKEN=still-visible-after-repair",
            )
        return garak_attack_replay.ReplayResult(
            ok=True,
            status_code=200,
            text="I cannot run tool instructions that disclose environment variables.",
        )

    class FakeDetector:
        def verify(self, output_text: str, **kwargs: Any) -> tuple[bool, float, str]:
            assert kwargs["target_tool"] == "bash_executor"
            if "GITHUB_TOKEN=" in output_text:
                return True, 0.98, "the attack still disclosed environment variables"
            return False, 0.98, "the defended victim refused the attack"

    async def fake_smart_benign_run(request: Any, agent: Any) -> ValidatorReport:
        # Pipeline integration test: the benign validator is stubbed so we don't depend on the synth
        # DAG, prompt rendering, or live LLM calls. Per-row replay+judge correctness is covered by
        # smart_benign's own unit tests.
        return ValidatorReport(
            agent_id=agent.agent_id,
            agent_name=agent.name,
            kind="benign",
            ok=True,
            summary="1/1 benign requests complied (0 refused, 0 errors)",
            metadata={"total": 1, "complied_count": 1, "refused_count": 0, "error_count": 0},
        )

    monkeypatch.setattr(garak_attack_replay, "_replay_prompt", fake_attack_replay)
    monkeypatch.setattr(garak_attack_replay, "_fetch_issue_comments", _empty_github_comments)
    monkeypatch.setattr(garak_attack_replay, "_detector_for", lambda *_args, **_kwargs: FakeDetector())
    monkeypatch.setattr(smart_benign, "run", fake_smart_benign_run)


def _single_report(run: PipelineRun) -> Any:
    assert len(run.reports) == 1
    return run.reports[0]


def _assert_successful_iteration(iteration: Any) -> None:
    assert iteration.success is True
    assert {defender.agent_name for defender in iteration.defenders} == {
        "openshell-policy-defender",
        "defender-guardrails",
    }
    assert all(defender.ok for defender in iteration.defenders)
    assert [patch["type"] for patch in iteration.policy_patches] == [
        "openshell_policy_candidate",
        "relay_plugins_candidate",
    ]

    # Deploy applied both patches through the real uploaders against the fake OpenShell CLI runner.
    assert iteration.victim_control.ok is True
    assert iteration.victim_control.summary == "deployed: openshell, relay"
    steps = {result["step"]: result for result in iteration.victim_control.metadata["results"]}
    assert steps.keys() == {"openshell", "relay"}
    assert all(result["ok"] for result in steps.values())

    # The victim saw both defender patches and both defenders.
    assert iteration.victim.ok is True
    assert iteration.victim.observations == {"policy_patch_count": 2, "defender_count": 2}

    validators_by_kind = {validator.kind: validator for validator in iteration.validators}
    assert validators_by_kind["attack"].ok is True
    assert validators_by_kind["attack"].summary == "blocked 1/1 original Garak attack hits"
    assert validators_by_kind["attack"].metadata["blocked_count"] == 1
    assert validators_by_kind["benign"].ok is True
    assert validators_by_kind["benign"].metadata["complied_count"] == 1


def _load_artifact(storage_dir: Path, name: str) -> Any:
    return json.loads((storage_dir / name).read_text(encoding="utf-8"))


async def _empty_github_comments(
    targets: list[garak_attack_replay.GitHubIssueTarget],
    config: garak_attack_replay.ReplayConfig,
    client: Any,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    return {target.label: [] for target in targets}, []
