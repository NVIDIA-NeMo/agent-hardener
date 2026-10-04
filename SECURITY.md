# Security

## Reporting a vulnerability

**Do not report security vulnerabilities through public GitHub issues, discussions, or pull
requests.** Public disclosure before a fix is available puts users at risk.

Report vulnerabilities in Agent Hardener to the NVIDIA Product Security Incident Response Team
(PSIRT):

- **Web:** https://www.nvidia.com/en-us/security/report-vulnerability/
- **Email:** psirt@nvidia.com

Please include enough detail for us to reproduce the issue: the version of Agent Hardener, the
platform, the sequence of commands, and — where applicable — a minimal manifest or agent
configuration that triggers it. If you send email, encrypting with the NVIDIA PSIRT PGP key is
appreciated but not required.

NVIDIA's vulnerability disclosure policy, including our expected response and remediation
timelines, is published at https://www.nvidia.com/en-us/security/.

## Scope

Agent Hardener is a security-testing tool that runs on a developer's own machine. It has no
NVIDIA-hosted service component, so there is no production infrastructure in scope.

**In scope** — issues in this package, for example:

- Escape from the sandbox that isolates the agent under test, reaching the developer's host.
- Disclosure of credentials read from the environment (for example, an API key written into run
  artifacts or logs).
- A flaw that causes a generated guardrail or egress policy to be reported as effective when it
  is not — a false assurance, which is the most serious class of bug for this tool.
- Code execution triggered by parsing an untrusted run artifact or scan result.

**Out of scope:**

- Vulnerabilities in the agent *you* point Agent Hardener at. Finding those is what the tool is
  for; report them to whoever owns that agent.
- Attack techniques or jailbreak prompts that succeed against a target agent. These are findings
  produced by the tool working correctly, not vulnerabilities in the tool.
- Vulnerabilities in third-party dependencies, unless Agent Hardener's use of them is what makes
  the issue exploitable. Report those upstream.
- Issues in garak, OpenShell, NeMo Guardrails, or NeMo Platform. Those are separate projects with
  their own reporting channels.
- Consequences of documented behaviour described under "Operating safely" in the README — for
  example, running a victim manifest you did not author, or binding `agent-hardener serve` to a
  non-loopback interface.

## Operating this tool safely

Agent Hardener generates working attacks and executes a deliberately hostile workload. Before
running it, read the **Operating safely** section of the README. In particular: only test agents
you own or are authorised to test, point the agent under test at non-production backends, and
treat the contents of `.agent-hardener/` as sensitive — the hitlog is a record of which attacks
succeeded against your agent.
