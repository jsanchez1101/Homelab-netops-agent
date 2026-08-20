# Homelab NetOps Agent

A framework-free local LLM agent that answers network-operations questions from live VyOS evidence. Qwen2.5-3B selects code-allowlisted, read-only commands; the executor runs them over SSH against a three-router OSPF lab; and the model grounds its answer in the returned device output.

**Result:** a frozen 10-run benchmark improved from **7.1 ± 0.74 / 12** with base Qwen2.5-3B to **7.9 ± 0.32 / 12** after LoRA (Low Rank Adaptation) fine-tuning. Mean delta: **+0.8 tasks** (95% CI: +0.3 to +1.3).

## How it works

```text
"How many OSPF neighbors does R1 have?"
                 │
                 ▼
        ┌──────────────────┐   emits a structured tool call
        │  Qwen2.5-3B      │   run_show_command(R1,
        │  local / Ollama  │     "show ip ospf neighbor")
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐   validates device and command
        │  agent.py        │   against code-level allowlists
        └────────┬─────────┘
                 │ SSH
                 ▼
        ┌──────────────────┐
        │  VyOS router     │──▶ live command output
        └────────┬─────────┘
                 │
                 ▼
        Evidence-grounded answer
```

No agent framework is used. The system prompt and tool schemas go to the model; the model returns a tool call; the executor validates it, runs it, and appends the output to context. The loop ends when the model answers in plain text or reaches an iteration cap. Runs are logged as JSONL trajectories for evaluation and fine-tuning.

## Architecture

```text
Mac (M2)              node1 (Ubuntu)          node2 (EVE-NG)
Ollama serving   ◀──▶ agent.py           ──▶ R1 ── R2
Qwen2.5-3B            run_evals.py            │  ╲  │
or qwen2.5-vyos       run_evals_multi.py      │   ╲ │  OSPF area 0
                      export_sft.py           R3 ───┘
     └────────── private Tailscale management plane ──────────┘
```

Three VyOS routers form an OSPF area-0 triangle inside EVE-NG. The management subnet is routed through a private Tailscale mesh, keeping model inference and device access local. Addressing and lab build steps are documented in [TOPOLOGY.md](TOPOLOGY.md).

## Security boundaries

- **The model is treated as untrusted input.** The executor—not the prompt—enforces the device and command allowlists.
- **Read-only commands only.** Hallucinated configuration commands and unsupported command forms are rejected before SSH execution.
- **Local inference.** Ollama serves the model inside the private network; no device output is sent to a hosted model API.
- **Dedicated lab credentials.** The agent does not authenticate as root. VyOS privilege levels still allow configuration mode, so the code-level command boundary remains the primary control.
- **Bounded context.** Device output and loop iterations are capped to prevent one verbose command from consuming the model context.
- **Sanitized training data.** Published artifacts contain metrics and hashes, not topology-bearing trajectories or credentials.

## Evaluation

The benchmark contains 12 tasks with frozen prompts, graders, and ground truth. Every reported result is a distribution across 10 runs against the same live lab at temperature 0.2.

| Model | Mean score | Standard deviation | Range |
|---|---:|---:|---:|
| Qwen2.5-3B baseline | 7.1 / 12 | 0.74 | 6–8 |
| LoRA fine-tuned Q4_K_M | **7.9 / 12** | **0.32** | 7–8 |

The mean improved by **0.8 tasks** (11.3%). A normal-approximation 95% confidence interval for the delta is **+0.3 to +1.3 tasks**.

| Task | Baseline passes | Fine-tuned passes |
|---|---:|---:|
| t01 | 10/10 | 10/10 |
| t02 | 10/10 | 10/10 |
| t03 | 10/10 | 9/10* |
| t04 | 10/10 | 10/10 |
| t05 | 10/10 | 10/10 |
| t06 | 0/10 | 0/10 |
| t07 | 4/10 | **10/10** |
| t08 | 10/10 | 10/10 |
| t09 | 7/10 | **10/10** |
| t10 | 0/10 | 0/10 |
| t11 | 0/10 | 0/10 |
| t12 | 0/10 | 0/10 |

\* One t03 attempt abstained after a transient SSH protocol-banner error. The predetermined sample was retained and no replacement run was added.

The fine-tune converted the two flaky tasks represented by successful training trajectories (t07 and t09) into stable 10/10 passes and reduced run-to-run variance. Four tasks remained stable failures, indicating a remaining capability or evidence-gathering ceiling rather than a consistency problem.

Raw sanitized results are available in [artifacts/posttrain_evaluation.json](artifacts/posttrain_evaluation.json).

## Leakage audit

The original benchmark exposed topology metadata that could let the model answer from context instead of live evidence. That leakage was removed before the final baseline was frozen. The final harness:

- keeps prompts and graders fixed across model comparisons;
- reports repeated-run distributions instead of a favorable single run;
- records per-task pass rates to separate stable capability from sampling variance;
- tests safe abstention separately from factual correctness; and
- retains infrastructure failures without silently adding replacement runs.

## Fine-tuning

Qwen2.5-3B-Instruct was fine-tuned with LoRA on **59 sanitized successful tool-use trajectories**: 51 training records and 8 validation records. Only assistant tokens contributed to the loss.

| Setting | Value |
|---|---|
| LoRA rank / alpha | 8 / 16 |
| Target modules | q, k, v, o, gate, up, and down projections |
| Learning rate | 1e-4 |
| Effective batch size | 4 |
| Epochs | 3 |
| Seed | 3407 |
| Maximum sequence length | 3072 |
| Trainable parameters | 14,966,784 (0.48%) |

Validation loss fell from **0.0605** after epoch 1 to **0.0349** after epoch 3.

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 0.0333 | 0.0605 |
| 2 | 0.0200 | 0.0374 |
| 3 | 0.0270 | 0.0349 |

The adapter was merged and exported as a **Q4_K_M GGUF** for local Ollama deployment: 1,929,902,720 bytes, SHA-256 `cd219b4dbfe69848b592aa1c62ce07eee29e367cb522f45a4fea6dde182581c2`.

Reproducibility artifacts:

- [training_manifest.json](artifacts/training_manifest.json)
- [training_provenance.json](artifacts/training_provenance.json)
- [vyos_trainer_state.json](artifacts/vyos_trainer_state.json) — complete step/loss history
- [posttrain_evaluation.json](artifacts/posttrain_evaluation.json)

The dataset itself is not published because the trajectories retain detailed topology transcripts. Its SHA-256 hash is included so an authorized copy can be verified without exposing it.

## Setup

```bash
# 1. Baseline model
ollama pull qwen2.5:3b

# Fine-tuned model: place the exported GGUF beside Modelfile
ollama create qwen2.5-vyos -f Modelfile

# Serve Ollama on the private network interface used by the agent host
OLLAMA_HOST=0.0.0.0 ollama serve

# 2. Agent environment
python3 -m venv .venv
source .venv/bin/activate
pip install requests pyyaml netmiko

# 3. Configuration
cp devices.yaml.example devices.yaml
# Add router addresses and credentials to devices.yaml.
# Point config.yaml at Ollama and choose qwen2.5:3b or qwen2.5-vyos.

# 4. Run
python agent.py "How many OSPF neighbors does R1 have?"
python run_evals.py
python run_evals_multi.py 10 --quiet
```

Do not expose an unauthenticated Ollama listener to an untrusted network. The example binds beyond localhost only because the agent and model hosts communicate over a private mesh.

## Repository map

| Path | Purpose |
|---|---|
| `agent.py` | Model loop, tool execution, allowlists, and trajectory logging |
| `run_evals.py` | Single benchmark run and programmatic scoring |
| `run_evals_multi.py` | Repeated-run statistics and per-task stability |
| `export_sft.py` | Filters successful trajectories into training examples |
| `evals/tasks.yaml` | Frozen tasks, checks, and ground truth |
| `configs/R*.txt` | VyOS bootstrap configurations |
| `devices.yaml.example` | Device inventory template without credentials |
| `config.yaml` | Model endpoint and agent settings |
| `Modelfile` | Ollama definition for the fine-tuned GGUF |
| `TOPOLOGY.md` | Lab diagram, addressing plan, and build steps |
| `artifacts/` | Sanitized training and evaluation provenance |

## Limitations

- The benchmark has only 12 tasks and one three-router OSPF topology.
- Four tasks remain stable failures after fine-tuning.
- The training set contains 59 successful trajectories, so the experiment measures targeted consistency gains rather than broad network-operations competence.
- The read-only boundary protects the executor, but device output can still contain untrusted text and must remain data—not instructions.
- The current output truncation is character-based rather than command-aware.

## Roadmap

- [x] Framework-free tool-calling loop and code-enforced read-only allowlist
- [x] Three-router VyOS OSPF lab in EVE-NG
- [x] Leakage-audited frozen benchmark
- [x] Ten-run evaluation with per-task stability
- [x] LoRA fine-tuning on 59 sanitized trajectories
- [x] Q4_K_M export and local Ollama deployment
- [ ] Add stronger verified trajectories for the four stable-failure tasks
- [ ] Evaluate on larger and more varied topologies
- [ ] Replace hard output truncation with command-aware context reduction

## Related

[DNS Exfiltration Detector](https://github.com/jsanchez1101/DNS-Exfiltration-Detector) — classical ML detection using query-side window features, leakage audits, and live-versus-lab traffic validation.
