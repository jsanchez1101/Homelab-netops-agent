# Homelab NetOps Agent

A local LLM agent that operates a virtualized network lab over SSH. A 3B model
(qwen2.5, served locally via Ollama) selects read-only `show` commands, executes
them against VyOS routers via netmiko, and reasons over the output. No API calls,
no cloud inference, no per-token cost.

Baseline: **8.0 ± 1.22 / 12** across 5 runs of a 12-task benchmark against the
running lab.

---

## How it works

```
  "How many OSPF neighbors does R1 have?"
                 │
                 ▼
        ┌──────────────────┐   model emits a tool call
        │  qwen2.5:3b      │   run_show_command(R1, "show ip ospf neighbor")
        │  (local, Ollama) │
        └────────┬─────────┘
                 │  JSON
                 ▼
        ┌──────────────────┐   allowlist check, then SSH
        │  agent.py        │   (read-only `show` commands only)
        │  (~200 lines)    │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  R1 (VyOS)       │──▶ live neighbor table
        └──────────────────┘
                 │
                 ▼
        "R1 has 2 OSPF neighbors (2.2.2.2, 3.3.3.3)."
```

No agent framework. System prompt plus tool schemas go to the model; the model
returns a tool call; the executor validates it against a read-only allowlist and
runs it; output is appended to context; repeat until the model answers in plain
text or hits an iteration cap. Every run is logged as a trajectory (JSONL).

## Architecture

```
Mac (M2)              node1 (Ubuntu)          node2 (EVE-NG)
Ollama serving   ◀──▶ agent.py           ──▶ R1 ── R2
qwen2.5:3b            run_evals.py            │  ╲  │   VyOS OSPF
(Metal, ~40 tok/s)    run_evals_multi.py      │   ╲ │   triangle
                      export_sft.py           R3 ───┘
     └────────── Tailscale mesh ──────────────┘
        management plane: 10.99.0.0/24 routed via node2
```

Three VyOS routers in an OSPF area-0 triangle inside EVE-NG. Router management
sits on an isolated `10.99.0.0/24` subnet advertised into a Tailscale mesh by
node2, so the lab is reachable across the tailnet without exposing it to the home
LAN. Addressing and build steps in [`TOPOLOGY.md`](TOPOLOGY.md).

## Baseline results

12 tasks, ground truth derived from the topology and checked programmatically
(exact or regex match against the model's answer).

A 3B model at temperature 0.2 varies meaningfully run to run, so the benchmark is
executed 5× and reported as a distribution rather than a single score.

**8.0 ± 1.22 / 12 (67%)** — range 6–9 across 5 runs.

| Task stability | Tasks | Count |
|---|---|---|
| Stable pass (5/5) | t01, t02, t03, t06, t12 | 5 |
| Flaky (1–4 of 5) | t04, t05, t07, t08, t09, t10 | 6 |
| Stable fail (0/5) | t11 | 1 |

A single run is not a measurement. Identical runs of this benchmark scored
between 6 and 9, so comparing one run against another cannot distinguish a real
change from noise. `run_evals_multi.py` reports mean, standard deviation, and
per-task pass rate, and flags when a session-over-session delta is smaller than
the observed run-to-run spread.

Each run takes about 3 minutes at ~40 tok/s on an M2 via Metal.

## Where it fails

The failures split into two kinds, and the distinction determines what to do
about them.

**Consistency, not capability (6 tasks).** t04, t05, t07, t08, t09, and t10 each
pass at least once — the model demonstrably can do them — but not reliably. t07
and t08 are the weakest at 1/5. These don't require new knowledge; they require
the model to do reliably what it already does occasionally.

**A genuine ceiling (1 task).** t11 fails 5/5. It's a counterfactual — "if the
R1–R2 link failed, would R1 still reach R2?" — requiring the model to simulate a
failure and trace an alternate path. The answer isn't present in any single
command's output. The model answers from priors without gathering evidence. This
is the one that needs a stronger teacher, not more consistency.

A prompt fix during this work moved t03 from failing to stable pass by stating
the VyOS interface syntax (`show interfaces ethernet <name>`) explicitly —
evidence that some failures are scaffold gaps rather than model limits.

## Design notes

**The allowlist is enforced in code, not the prompt.** The model is treated as
untrusted input. The system prompt says read-only, but prompts are suggestions to
a model; the `startswith("show ")` check in the executor is not. If the model
hallucinates a configuration command, or is prompt-injected by content in device
output, the executor refuses. Same trust boundary as parameterized SQL versus
trusting user input.

**Least-privilege account.** The agent authenticates as a dedicated `admin` user
rather than root, so a compromised credential is a restricted user on lab VMs.
Caveat: VyOS users created this way can still enter configuration mode, so the
read-only guarantee comes from the code-level allowlist, not a VyOS privilege
level.

**Output truncation is context management.** A 3B model's usable context is the
agent's entire working memory, so device output is capped to keep one verbose
command from crowding out the reasoning. The current implementation is a hard
character limit; smarter truncation is an open improvement.

## Roadmap

- [x] Agent loop, tool-calling, read-only allowlist
- [x] 3-router VyOS OSPF lab (EVE-NG, Tailscale-routed management plane)
- [x] 12-task benchmark with programmatic ground truth
- [x] Multi-run harness — baseline 8.0 ± 1.22 / 12 with per-task stability
- [ ] Self-distillation: fine-tune on the agent's own successful trajectories to
      convert the six flaky tasks into stable passes
- [ ] Teacher distillation for t11: collect a stronger model's verified
      trajectories on the counterfactual task and test whether the reasoning
      pattern transfers

The stability breakdown reframes the distillation target. Most lost points are
consistency, not missing capability — successful trajectories already exist for
every flaky task, so self-distillation has material to work with. t11 is the only
task with no successful example to learn from, and is the sole candidate for
teacher distillation. Success will be measured by re-running the same 5× harness:
a real improvement has to move the mean by more than the 1.22 run-to-run spread,
and ideally shrink that spread as well. A negative result is also a result.

## Setup

```bash
# 1. Serve the model (on a machine with a GPU)
ollama pull qwen2.5:3b
OLLAMA_HOST=0.0.0.0 ollama serve

# 2. On the agent host
python3 -m venv .venv && source .venv/bin/activate
pip install requests pyyaml netmiko

# 3. Configure
cp devices.yaml.example devices.yaml   # router IPs + credentials
# edit config.yaml -> base_url to point at the Ollama endpoint

# 4. Run
python agent.py "How many OSPF neighbors does R1 have?"
python run_evals.py                  # single run
python run_evals_multi.py 5 --quiet  # 5 runs, mean ± stdev, stability table
```

Any SSH-reachable device that speaks `show` commands works — point
`devices.yaml` at it. The endpoint is OpenAI-compatible, so swapping the local
model for a cloud API is a one-line config change.

## Repository

| File | Purpose |
|---|---|
| `agent.py` | Agent loop: model call, tool execution, allowlist, trajectory logging |
| `run_evals.py` | Single benchmark run; scores answers, tags trajectories pass/fail |
| `run_evals_multi.py` | Repeats the benchmark N times; reports mean ± stdev and per-task stability |
| `export_sft.py` | Filters successful trajectories into fine-tuning data |
| `evals/tasks.yaml` | Benchmark tasks with programmatic checks and ground truth |
| `configs/R*.txt` | VyOS bootstrap configs for the three routers |
| `devices.yaml.example` | Device inventory template |
| `config.yaml` | Model endpoint and agent settings |
| `TOPOLOGY.md` | Lab diagram, addressing plan, build steps |

## Related

[DNS Exfiltration Detector](https://github.com/jsanchez1101/DNS-Exfiltration-Detector)
— classical ML for detecting data exfiltration over DNS, running on the same
homelab.
