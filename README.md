# Homelab NetOps Agent

A local LLM agent that operates a virtualized network lab over SSH. A 3B model
(qwen2.5, served locally via Ollama) selects read-only `show` commands, executes
them against VyOS routers via netmiko, and reasons over the output. No API calls,
no cloud inference, no per-token cost.

Baseline: **8/12** on a 12-task benchmark against the running lab.

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
(Metal, ~40 tok/s)    export_sft.py           │   ╲ │   triangle
                                              R3 ───┘
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

| Difficulty | Passed |
|---|---|
| easy | 3/4 |
| medium | 4/4 |
| hard | 1/4 |
| **total** | **8/12** |

Full run takes about 2.5 minutes at ~40 tok/s on an M2 via Metal.

## Where it fails

The four failures fall into three categories, and only one is a model limitation.

| Task | Category | Detail |
|---|---|---|
| t03 | Scaffold gap | Model chose `show interfaces eth1`; VyOS expects `show interfaces ethernet eth1`. Fixable in the prompt or tool description. t08 hit the same error and self-corrected on the following turn, so the capability is present — the syntax just isn't stated explicitly. |
| t09 | Benchmark bug | Model answered correctly; the checker was wrong. |
| t10 | Benchmark bug / underspecified task | Asked whether OSPF timers match, the model read the live countdown timers (which always differ, since routers don't boot in sync) rather than the configured intervals. Hard to distinguish from raw output. |
| t11 | Capability limit | Counterfactual: "if the R1–R2 link failed, would R1 still reach R2?" Requires simulating a failure and tracing an alternate path — the answer isn't in any single command's output. The model answered from priors without gathering evidence, and got it backwards. |

Model-limited failures are closer to 1/12. Two of the four were the benchmark,
not the agent.

The pattern: a 3B model reliably selects the right command and reads structured
output, and degrades on multi-step reasoning where the answer has to be derived
rather than found. t11 is the ceiling, and it won't yield to prompt engineering.

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
- [x] 12-task benchmark with programmatic ground truth — baseline 8/12
- [ ] Scaffold iteration: t03-class syntax gaps, t09/t10 checker fixes
- [ ] Teacher distillation: collect verified trajectories, LoRA fine-tune
      qwen2.5:3b, reload into Ollama, report before/after on the same benchmark

t11-class reasoning failures won't respond to prompt changes, so the distillation
plan is to collect successful trajectories — the agent's own, plus a stronger
model's runs on the tasks the 3B can't do — fine-tune on the union, and measure
whether the capability transfers. A negative result is also a result.

## Setup

```bash
# 1. Serve the model (on the machine with a GPU)
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
python run_evals.py
```

Any SSH-reachable device that speaks `show` commands works — point
`devices.yaml` at it. The endpoint is OpenAI-compatible, so swapping the local
model for a cloud API is a one-line config change.

## Repository

| File | Purpose |
|---|---|
| `agent.py` | Agent loop: model call, tool execution, allowlist, trajectory logging |
| `run_evals.py` | Runs the benchmark, scores answers, tags trajectories pass/fail |
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


