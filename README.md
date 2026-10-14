# Homelab NetOps Agent

A local LLM agent that operates a virtualized network lab over SSH. A 3B model
(qwen2.5, served locally via Ollama) selects read-only `show` commands, executes
them against VyOS routers via netmiko, and reasons over the output. No API calls,
no cloud inference, no per-token cost.

Baseline status: **corrected 10-run baseline complete: 6.0/12 mean (50%),
standard deviation 0.82, range 5–7.** Earlier scores were invalidated by
answer-bearing inventory descriptions, permissive substring checks, and a
temporary mismatch between YAML ground truth and the live R2 state.

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
        "R1 has 2 OSPF neighbors (10.50.0.2, 3.3.3.3)."
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

## Evaluation status

The benchmark has 12 training/development tasks plus five held-out question
variants. A pass now requires two independent conditions:

1. the first answer line satisfies a task-specific validator; and
2. the trajectory contains a successful, task-appropriate VyOS command whose
   output supports the conclusion.

This replaced permissive substring checks that could accept a wrong answer merely
because the explanation mentioned the expected device. It also prevents an
unsupported guess after an invalid command from becoming a positive SFT example.

The held-out variants use the same live topology, so they measure command and
evidence transfer to new questions. They do **not** prove generalization to unseen
network facts. That stronger experiment requires a second topology or a deliberate
live-state change with separately captured ground truth.

Earlier reported scores are retained only as historical artifacts and must not be
used as the fine-tuning baseline. The corrected baseline was captured over 10 runs
with the current task definitions and neutral inventory; scores were 7, 5, 6, 7,
6, 5, 6, 6, 7, and 5 (approximate 95% interval: 5.49–6.51 tasks).

`run_evals_multi.py` reports the score distribution, per-task pass rate, and an
approximate 95% confidence interval. Each run takes about 3 minutes at ~40 tok/s
on an M2 via Metal.

## Observed failure modes

- Wrong VyOS syntax followed by repeated retries or an unsupported guess.
- Correct facts paired with irrelevant evidence, such as deriving an OSPF router
  ID from `show version`.
- Incorrect semantic extraction from valid output, especially DR versus backup.
- Counterfactual reasoning without inspecting the OSPF link-state database.

The explicit `show interfaces ethernet <name>` prompt hint is still useful, but
syntax knowledge and evidence-grounded reasoning are measured separately now.

## Design notes

**The allowlist is enforced in code, not the prompt.** The model is treated as
untrusted input. The system prompt says read-only, but prompts are suggestions to
a model; the executor requires a single CLI line beginning with `show `. If the model
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
- [x] Evidence-aware evaluators and explicit train/holdout trajectory provenance
- [x] Multi-run harness with confidence intervals and same-suite comparisons
- [x] Fresh 10-run baseline after evaluator repair (6.0/12 mean)
- [ ] Self-distillation: fine-tune on the agent's own successful trajectories to
      convert the six flaky tasks into stable passes
- [ ] Teacher distillation for t11: collect a stronger model's verified
      trajectories on the counterfactual task and test whether the reasoning
      pattern transfers

Only trajectories accepted by the repaired answer and evidence validators are
eligible for export. Legacy records, holdout runs, invalid tool transcripts, and
the original answer-bearing inventory prompt are excluded. Fine-tuning should not
start until a new clean baseline and training export exist.

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
cp config.yaml config.local.yaml        # ignored private runtime settings
# edit config.local.yaml -> base_url to point at the Ollama endpoint

# 4. Run
python agent.py "How many OSPF neighbors does R1 have?"
python run_evals.py                            # training/dev suite, single run
python run_evals_multi.py 10 --quiet           # clean baseline distribution
python run_evals_multi.py 10 --tasks holdout.yaml --quiet
python -m unittest discover -s tests -v
python export_sft.py                          # verified training split only
```

### Smoke inventory versus evaluation inventory

`devices.yaml` is the neutral inventory used by normal agent runs and required by
both evaluation scripts. Its descriptions identify devices and intended links but
do not reveal router IDs, addresses, or expected answers.

For end-to-end integration debugging, create the ignored rich inventory once:

```bash
cp devices.smoke.yaml.example devices.smoke.yaml
# add the real local credentials, then run:
LAB_AGENT_INVENTORY=devices.smoke.yaml python agent.py "What devices are in the lab?"
```

The rich smoke descriptions intentionally expose topology facts. Evaluation
scripts refuse to start while `LAB_AGENT_INVENTORY` selects that file, preventing
the integration fixture from leaking into benchmark results.

Any SSH-reachable device that speaks `show` commands works — point
`devices.yaml` at it. The endpoint is OpenAI-compatible, so swapping the local
model for a cloud API is a one-line config change.

### Publishing safely

`devices.yaml`, `devices.smoke.yaml`, and `config.local.yaml` are private runtime
files and are ignored by Git. Keep real device passwords, model endpoints, API
keys, and Tailscale addresses out of tracked examples and screenshots. The
bootstrap configs contain an explicit password placeholder that must be replaced
locally before use. Crop browser chrome from public dashboard screenshots so a
tailnet address, account name, tabs, and bookmarks are not exposed.

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
| `devices.smoke.yaml.example` | Rich integration-test inventory; never used for evaluation |
| `config.yaml` | Model endpoint and agent settings |
| `TOPOLOGY.md` | Lab diagram, addressing plan, build steps |

## Related

[DNS Exfiltration Detector](https://github.com/jsanchez1101/DNS-Exfiltration-Detector)
— classical ML for detecting data exfiltration over DNS, running on the same
homelab.
