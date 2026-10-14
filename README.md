# Homelab NetOps Agent

A framework-free local LLM agent that answers network-operations questions from live VyOS evidence. Qwen2.5-3B selects code-allowlisted, read-only commands; the executor runs them over SSH against a three-router OSPF lab; and the model grounds its answer in the returned device output.

**Result:** a frozen 10-run benchmark improved from **7.1 ± 0.74 / 12** with base Qwen2.5-3B to **7.9 ± 0.32 / 12** after LoRA (Low Rank Adaptation) fine-tuning. Mean delta: **+0.8 tasks** (95% CI: +0.3 to +1.3).

The training data came from successful runs on the same 12 benchmark tasks, so this measures consistency on tasks the model had already seen, not generalization to new ones. See [What the fine-tune does and doesn't show](#what-the-fine-tune-does-and-doesnt-show).

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

- **The model is treated as untrusted input.** The executor, not the prompt, enforces the device and command allowlists.
- **Read-only commands only.** Hallucinated configuration commands and unsupported command forms are rejected before SSH execution.
- **Local inference.** Ollama serves the model inside the private network; no device output is sent to a hosted model API.
- **Dedicated lab credentials.** The agent does not authenticate as root. VyOS privilege levels still allow configuration mode, so the code-level command boundary remains the primary control.
- **Bounded context.** Device output and loop iterations are capped to prevent one verbose command from consuming the model context.
- **Sanitized training data.** Published artifacts contain metrics and hashes, not topology-bearing trajectories or credentials.
- **No hosted model egress.** The model endpoint must resolve to loopback, a private range, or the Tailscale range; public or cloud endpoints are rejected.

## Evaluation

The benchmark contains 12 tasks with frozen prompts, graders, and ground truth. Every reported result is a distribution across 10 runs against the same live lab at temperature 0.2.

A pass requires two independent conditions:

1. the first answer line satisfies a task-specific validator; and
2. the trajectory contains a successful, task-appropriate VyOS command whose output supports the conclusion.

This replaced permissive substring checks that could accept a wrong answer merely because the explanation mentioned the expected device. It also prevents an unsupported guess after an invalid command from becoming a positive training example.

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

### What the fine-tune does and doesn't show

The 59 training trajectories are successful runs of benchmark tasks t01 to t05 and t07 to t09, the same tasks the model is then scored on. There was no valid training example for t06 or t10 to t12, and those four tasks stayed at 0/10.

- **It does show** that a small LoRA fine-tune can turn flaky behavior into reliable behavior. t07 went from 4/10 to 10/10 and t09 from 7/10 to 10/10, and run-to-run spread dropped from 0.74 to 0.32.
- **It doesn't show** that the model learned anything it couldn't already do some of the time, or that it would do better on new questions or a different topology. That needs a held-out task set and a changed lab (renamed devices, new addressing), which is the next experiment.

Raw sanitized results are available in [artifacts/posttrain_evaluation.json](artifacts/posttrain_evaluation.json).

### Frozen benchmark files

The router configs, `TOPOLOGY.md`, and eval suites in this repo match the frozen `eval-v1` benchmark. R2's router ID is `10.50.0.2` rather than `2.2.2.2`, so the model can't guess it from the 1.1.1.1 / 2.2.2.2 / 3.3.3.3 pattern. `benchmark_freeze.json` records SHA-256 hashes of the agent, graders, and task files; the eval scripts refuse to run if any of them drift.

### Held-out and abstention suites

- `evals/holdout.yaml` holds five held-out question variants. They use the same live topology, so they measure command and evidence transfer to new questions, not generalization to unseen network facts.
- `evals/abstention.yaml` is a mixed calibration suite that checks whether the agent declines to answer when the evidence doesn't support a conclusion.

### Observed failure modes

- Wrong VyOS syntax followed by repeated retries or an unsupported guess.
- Correct facts paired with irrelevant evidence, such as deriving an OSPF router ID from `show version`.
- Incorrect semantic extraction from valid output, especially DR versus backup.
- Counterfactual reasoning without inspecting the OSPF link-state database.

`failure_reports.py` writes readable reports of failed trajectories to `reports/`.

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
- [vyos_trainer_state.json](artifacts/vyos_trainer_state.json): complete step/loss history
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
cp devices.yaml.example devices.yaml   # router addresses + credentials
cp config.yaml config.local.yaml        # ignored private runtime settings
# Point config.local.yaml at Ollama and choose qwen2.5:3b or qwen2.5-vyos.

# 4. Run
python agent.py "How many OSPF neighbors does R1 have?"
python run_evals.py                                  # single benchmark run
python run_evals_multi.py 10 --quiet                 # 10-run distribution
python run_evals_multi.py 10 --tasks holdout.yaml --quiet
python run_abstention_evals.py                       # abstention calibration
python -m unittest discover -s tests -v
python export_sft.py                                 # verified training split only
```

### Smoke inventory versus evaluation inventory

`devices.yaml` is the neutral inventory used by normal agent runs and required by the evaluation scripts. Its descriptions identify devices and intended links but do not reveal router IDs, addresses, or expected answers.

For end-to-end integration debugging, create the ignored rich inventory once:

```bash
cp devices.smoke.yaml.example devices.smoke.yaml
# add the real local credentials, then run:
LAB_AGENT_INVENTORY=devices.smoke.yaml python agent.py "What devices are in the lab?"
```

The rich smoke descriptions intentionally expose topology facts. Evaluation scripts refuse to start while `LAB_AGENT_INVENTORY` selects that file, so the integration fixture can't leak into benchmark results.

### Publishing safely

`devices.yaml`, `devices.smoke.yaml`, and `config.local.yaml` are private runtime files and are ignored by Git. Keep real device passwords, model endpoints, API keys, and Tailscale addresses out of tracked examples and screenshots. The bootstrap configs contain an explicit password placeholder that must be replaced locally before use.

Do not expose an unauthenticated Ollama listener to an untrusted network. The example binds beyond localhost only because the agent and model hosts communicate over a private mesh.

## Repository map

| Path | Purpose |
|---|---|
| `agent.py` | Model loop, tool execution, allowlists, and trajectory logging |
| `run_evals.py` | Single benchmark run and programmatic scoring |
| `run_evals_multi.py` | Repeated-run statistics and per-task stability |
| `run_abstention_evals.py` | Abstention calibration suite |
| `export_sft.py` | Filters successful trajectories into training examples |
| `benchmark_freeze.py` | Creates and enforces the `eval-v1` file-hash manifest |
| `failure_reports.py` | Readable reports of failed trajectories |
| `evals/tasks.yaml` | Frozen tasks, checks, and ground truth |
| `evals/holdout.yaml` | Held-out question variants |
| `evals/abstention.yaml` | Abstention calibration tasks |
| `tests/` | Unit tests for graders, export, and reports |
| `configs/R*.txt` | VyOS bootstrap configurations |
| `devices.yaml.example` | Device inventory template without credentials |
| `devices.smoke.yaml.example` | Rich integration-test inventory; never used for evaluation |
| `config.yaml` | Model endpoint and agent settings |
| `Modelfile` | Ollama definition for the fine-tuned GGUF |
| `TOPOLOGY.md` | Lab diagram, addressing plan, and build steps |
| `artifacts/` | Sanitized training and evaluation provenance |

## Limitations

- The benchmark has only 12 tasks and one three-router OSPF topology.
- Four tasks remain stable failures after fine-tuning.
- The training set contains 59 successful trajectories from the same tasks the model is evaluated on, so the experiment measures consistency on seen tasks rather than broad network-operations competence.
- The read-only boundary protects the executor, but device output can still contain untrusted text and must remain data, not instructions.
- The current output truncation is character-based rather than command-aware.

## Roadmap

- [x] Framework-free tool-calling loop and code-enforced read-only allowlist
- [x] Three-router VyOS OSPF lab in EVE-NG
- [x] Leakage-audited frozen benchmark
- [x] Ten-run evaluation with per-task stability
- [x] LoRA fine-tuning on 59 sanitized trajectories
- [x] Q4_K_M export and local Ollama deployment
- [ ] Add stronger verified trajectories for the four stable-failure tasks
- [x] Sync the frozen `eval-v1` configs and tasks into this repo
- [x] Held-out question variants and abstention suite
- [ ] Renamed/re-addressed lab to test generalization to unseen network facts
- [ ] Evaluate on larger and more varied topologies
- [ ] Replace hard output truncation with command-aware context reduction

## Related

[DNS Exfiltration Detector](https://github.com/jsanchez1101/DNS-Exfiltration-Detector): classical ML on DNS query windows, including an audit that found the benchmark dataset's attack captures contain no exfiltration queries.

## License

Original project code and documentation are available under the [MIT License](LICENSE). See [license scope and third-party materials](THIRD_PARTY_NOTICES.md) for exclusions and attribution boundaries.
