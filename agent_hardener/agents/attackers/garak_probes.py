# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The garak probes Agent Hardener drives, and the one place their garak names live.

The attacker, its preflight and diagnosis, and the replay validator all read a probe's garak plugin
names, file names and attack type from here, so adding a probe is one entry rather than a hunt for
string literals. Imports nothing from Agent Hardener: the validators and the attacker both depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AttackType = Literal["direct_prompt_injection", "indirect_prompt_injection"]

#: garak's Python package for a plugin of each kind; a plugin name is ``<module>.<Class>`` beneath it.
GARAK_PACKAGES = {"probes": "garak.probes", "detectors": "garak.detectors"}


def split_plugin(name: str) -> tuple[str, str]:
    """``"<module>.<Class>"`` -> ``("<module>", "<Class>")``, the keys garak nests its plugin config under."""
    module, _, class_name = name.rpartition(".")
    if not (module and class_name):
        raise ValueError(f"garak plugin name must be '<module>.<Class>', got {name!r}")
    return module, class_name


@dataclass(frozen=True)
class GarakProbe:
    """One garak probe run as an Agent Hardener attacker, with the detector that judges its hits."""

    #: Registry key, and the word that marks this probe's hits (``identify_attack_type``).
    key: str
    #: garak probe plugin, ``<module>.<Class>`` under ``garak.probes``.
    probe: str
    #: garak detector plugin, ``<module>.<Class>`` under ``garak.detectors``.
    detector: str
    #: Prefix garak stamps on the report/hitlog, and the stem of this probe's own file names.
    report_prefix: str
    attack_type: AttackType

    @property
    def source(self) -> str:
        """The ``source`` stamped on this probe's attack records."""
        return f"garak-{self.report_prefix}"

    @property
    def probe_class(self) -> str:
        """The probe's class name, which garak prefixes its log lines with (``"<Class> # ..."``)."""
        return split_plugin(self.probe)[1]

    @property
    def stdout_log_name(self) -> str:
        """The file garak's stdout goes to, beside its report; the detailed log it names is read for diagnosis."""
        return f"{self.source}.log"

    @property
    def detail_log_name(self) -> str:
        """The detailed log garak writes, pinned per scan via ``GARAK_LOG_FILE``."""
        return f"{self.source}.detail.log"

    @property
    def resolved_config_name(self) -> str:
        """The rendered garak config passed to ``garak --config``."""
        return f"{self.source}.resolved.yaml"

    def plugin_config(self, plugins: dict[str, Any], kind: Literal["probes", "detectors"]) -> dict[str, Any]:
        """This probe's (or its detector's) config block in a garak ``plugins`` mapping; ``{}`` when absent."""
        module, class_name = split_plugin(self.probe if kind == "probes" else self.detector)
        block = ((plugins.get(kind) or {}).get(module) or {}).get(class_name)
        return block if isinstance(block, dict) else {}


AGENT_BREAKER = GarakProbe(
    key="agent_breaker",
    probe="agent_breaker.AgentBreaker",
    detector="agent_breaker.AgentBreakerResult",
    report_prefix="agent-breaker",
    attack_type="direct_prompt_injection",
)

PROBES: dict[str, GarakProbe] = {probe.key: probe for probe in (AGENT_BREAKER,)}

#: The detector that judges each attack type on replay. Indirect injection has no attacker here: its hits
#: arrive preloaded, so it appears only in this table.
DETECTORS: dict[AttackType, str] = {
    **{probe.attack_type: probe.detector for probe in PROBES.values()},
    "indirect_prompt_injection": "indirect_injection.IndirectInjectionResult",
}
