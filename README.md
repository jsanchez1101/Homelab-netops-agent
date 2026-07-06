# Homelab NetOps Agent

A local LLM agent that operates a live network lab over SSH. It answers
operational questions about a running OSPF topology — "how many neighbors does
R1 have?", "is R1's route to R2 still valid?" — by choosing the right `show`
command, running it against real routers, and reasoning over the output.

The whole loop runs on hardware I own: a 3-billion-parameter model served
locally, an agent scaffold with no framework, and a virtual router lab. No
API calls, no cloud inference, no per-token cost.

This is the agentic/LLM half of a two-project cyber+ML portfolio; the other
half is a classical-ML [DNS exfiltration detector](https://github.com/jsanchez1101/DNS-Exfiltration-Detector).
Both run on the same self-built homelab.

---

## What it does

```
  "How many OSPF neighbors does R1 have?"
                 │
                 ▼
        ┌─────────────────┐   the model decides to call a tool
        │  qwen2.5:3b      │   run_show_command(R1, "show ip ospf neighbor")
        │  (local, Ollama) │
        └────────┬────────┘
                 │  tool call (JSON)
                 ▼
        ┌─────────────────┐   allowlist check + SSH via netmiko
        │  agent.py        │   (read-only `show` commands only)
        │  (~200 lines)    │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  R1 (VyOS)       │──▶ returns live neighbor table
        └─────────────────┘
                 │
                 ▼
        "R1 has 2 OSPF neighbors (2.2.2.2, 3.3.3.3)."
```

The agent loop is deliberately framework-free: system prompt + tool schemas →
model emits a tool call → the executor validates it against a read-only
allowlist and runs it → output is appended to context → repeat until the model
answers in plain text or hits an iteration cap. Every run is logged as a
trajectory (JSONL) for later fine-tuning.

## Architecture

```
Mac (M2)              node1 (Ubuntu)          node2 (EVE-NG)
Ollama serving   ◀──▶ agent.py           ──▶ R1 ── R2
qwen2.5:3b            run_evals.py            │  ╲  │   VyOS OSPF
(Metal GPU)          export_sft.py           │   ╲ │   triangle
                                              R3 ───┘
     └────────── Tailscale mesh ──────────────┘
        management plane: 10.99.0.0/24 routed via node2
```

- **Model host:** qwen2.5:3b on an M2 Mac via Ollama (Metal), ~40 tok/s.
- **Agent host:** node1 runs the loop and talks to the routers over SSH.
- **Lab:** three VyOS routers in an OSPF area-0 triangle inside EVE-NG on node2.
- **Connectivity:** a Tailscale mesh overlay; router management lives on an
  isolated `10.99.0.0/24` subnet advertised into the tailnet by node2, so the
  lab is reachable from anywhere without exposing it to the home LAN.

See [`TOPOLOGY.md`](TOPOLOGY.md) for the addressing plan and lab build steps.

## Design notes

- **The allowlist is the security boundary, and it lives in code, not the
  prompt.** The model is treated as untrusted input: even if it emits a
  configuration command or is prompt-injected by device output, the executor
  refuses anything that isn't a read-only `show`. Same trust model as
  parameterized SQL.
- **Least-privilege account.** The agent authenticates as a dedicated `admin`
  user, not root — so a compromised credential is a restricted user on lab
  VMs, not full control of anything.
- **Output truncation is context management.** A 3B model's usable context is
  the agent's entire working memory; device dumps are capped so one fat command
  can't crowd out the reasoning.

## Baseline results

12-task benchmark against the live lab, ground truth derived from the topology I
built. Scored programmatically (exact/regex match on the model's answer).

| Difficulty | Passed |
|------------|--------|
| easy       | 3/4    |
| medium     | 4/4    |
| hard       | 1/4    |
| **total**  | **8/12 (67%)** |

The number matters less than the failure breakdown, which splits into three
different categories:

- **t03 (scaffold gap):** the model chose `show interfaces eth1`; VyOS wants
  `show interfaces ethernet eth1`. A prompt/tool-description fix, not a model
  limitation. (Notably t08 hit the same error and self-corrected on the next
  turn — so the capability is there; the prompt just needs to make the syntax
  explicit.)
- **t09 & t10 (benchmark issues):** the model's answers were arguably correct;
  the checker or the ground truth was too strict. t10 in particular exposed a
  real subtlety — the model read OSPF *countdown* timers as if they were
  configured intervals, which is a genuinely hard distinction to make from raw
  output.
- **t11 (true capability gap):** a counterfactual — "if the R1–R2 link failed,
  would R1 still reach R2?" — that requires simulating a failure and tracing an
  alternate path. The answer isn't present in any single command's output, and
  the 3B model answered from priors without gathering evidence. This is the one
  that needs a stronger teacher.

So the *model-limited* failure rate is closer to 1/12; two of the four failures
are the benchmark being wrong, not the agent.

## Roadmap

1. **Scaffold iteration (free):** fix the t03-class syntax gaps and t09/t10
   ground-truth bugs by reading failed trajectories and tightening prompts and
   checks. Re-measure.
2. **Self-distillation (SFT):** filter the agent's own successful trajectories
   (`export_sft.py`) into training data, LoRA fine-tune qwen2.5:3b on rented
   GPU (~$1–5), reload into Ollama, re-benchmark.
3. **Teacher distillation:** for the hard reasoning tasks the small model can't
   do (t11), swap the endpoint to a stronger model, collect its *verified*
   trajectories, and fine-tune the 3B on the teacher's behavior — injecting
   capability the base model lacked.
4. **Capstone:** wire the [DNS detector](https://github.com/jsanchez1101/DNS-Exfiltration-Detector)
   to the agent — a detector alert triggers the agent to SSH in, gather
   evidence, and write a triage summary.

## Repository layout

| File | Purpose |
|------|---------|
| `agent.py` | The agent loop: model call, tool execution, allowlist, trajectory logging. |
| `run_evals.py` | Runs the 12-task benchmark, scores answers, tags trajectories pass/fail. |
| `export_sft.py` | Filters successful trajectories into SFT-ready training data. |
| `evals/tasks.yaml` | Benchmark tasks with programmatic checks and ground truth. |
| `configs/R*.txt` | VyOS bootstrap configs for the three routers. |
| `devices.yaml.example` | Device inventory template (copy to `devices.yaml`, fill in credentials). |
| `config.yaml` | Model endpoint + agent parameters (Ollama by default; swappable to any OpenAI-compatible API). |
| `TOPOLOGY.md` | Lab diagram, addressing plan, and build steps. |
| `PLAN.md` | Project plan and setup checklist. |

## Setup

```bash
# 1. Serve the model (on the machine with a GPU)
ollama pull qwen2.5:3b
OLLAMA_HOST=0.0.0.0 ollama serve

# 2. On the agent host
python3 -m venv .venv && source .venv/bin/activate
pip install requests pyyaml netmiko

# 3. Configure
cp devices.yaml.example devices.yaml   # fill in router IPs + credentials
# edit config.yaml -> base_url to point at your Ollama endpoint

# 4. Run
python agent.py "How many OSPF neighbors does R1 have?"
python run_evals.py
```

The lab itself (VyOS routers, EVE-NG, the management network) is documented in
`TOPOLOGY.md`. Any device reachable over SSH that speaks `show` commands works —
the agent is vendor-agnostic; point `devices.yaml` at it.
