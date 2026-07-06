# Lab Agent — 2-Day Plan

Goal: local agent (qwen2.5:3b on M2 via Ollama) that answers questions about
your EVE-NG topology over SSH, with a scored eval benchmark and trajectory
collection for later fine-tuning. Everything below day 2 afternoon is stretch.

Architecture:

    Mac M2 (Ollama, qwen2.5:3b) <--Tailscale--> node1 (agent.py) --SSH--> node2 (EVE-NG devices)

---

## Day 1 morning — plumbing (60-90 min)

### 1. Mac: install + serve Ollama over Tailscale
```bash
brew install ollama          # or download from ollama.com
ollama pull qwen2.5:3b       # ~2GB
# Make Ollama listen on all interfaces (not just localhost):
launchctl setenv OLLAMA_HOST "0.0.0.0"
# Restart the Ollama app/service after setting this.
tailscale ip -4              # note the Mac's 100.x.x.x address
```
Sanity check on the Mac: `ollama run qwen2.5:3b "say hi"` — should stream fast.

### 2. node1: verify reachability + install deps
```bash
curl http://<MAC_TAILSCALE_IP>:11434/api/tags     # should list qwen2.5:3b
mkdir ~/lab-agent && cd ~/lab-agent               # copy project files here
pip install requests pyyaml netmiko
```
Edit `config.yaml` → put the Mac's Tailscale IP in `base_url`.

### 3. EVE-NG: build the topology
**Follow TOPOLOGY.md** — VyOS image install (one-time), diagram, addressing,
and paste-ready `set` configs in `configs/`. Ground truth in evals/tasks.yaml
matches that design. Then verify management access:
The agent SSHes from node1 into lab devices, so device mgmt interfaces must be
reachable from node1. Standard pattern: a "Cloud" network in EVE-NG bridged to
node2's NIC (pnet), devices' mgmt interfaces on it, and node1 routes to that
subnet (or just use a flat L2 since both nodes are on your LAN).
- Each device: configs already enable SSH + admin user (no extra steps).
- Test manually from node1 first: `ssh admin@<device-ip>` — if YOU can't get
  in, the agent can't either. Debug this before blaming the agent.
- Fill in `devices.yaml` with real IPs/creds/device_types.

### 4. Smoke test (no devices needed)
```bash
python agent.py "What devices are in the lab?"
```
This only exercises Mac↔node1↔model and the list_devices tool. If this works,
the model loop is healthy. Then:
```bash
python agent.py "How many OSPF neighbors does R1 have?"
```
First full end-to-end run: model → tool call → SSH → answer.

## Day 1 afternoon — benchmark

### 5. Verify ground truth
`evals/tasks.yaml` has 12 tasks pre-derived from TOPOLOGY.md. Spot-check 2-3
against the live lab (e.g. `show ip route ospf` on R2 = 3 routes) before
trusting the benchmark.

### 6. Baseline run
```bash
python run_evals.py
```
Record the score. **This number is your Phase-1 equivalent** — same move as
the DNS detector: an honest baseline that motivates everything after it.
Expect something like 50-80% with a 3B model. Failures are data, not
embarrassment.

### 7. Iterate the scaffold (free, do this a lot)
Look at the FAIL trajectories in `trajectories/trajectories.jsonl`. Typical
3B failure modes and the fix:
- wrong command chosen        → add hints to SYSTEM_PROMPT or tool description
- answer buried in rambling   → tighten "state answer in first sentence" rule
- loops without finishing     → lower max_iterations, add "be economical" pressure
- huge output confuses it     → lower max_tool_output_chars, suggest specific commands
Re-run evals after each change. Track the score. Stop when it plateaus.

## Day 2 — data + (stretch) training

### 8. Collect trajectories
Run the eval suite a few times (temperature 0.2 gives some variation), plus
ad-hoc tasks. Target: 50-100 successful trajectories minimum.
```bash
python export_sft.py        # → sft_data.jsonl, successes only
```

### 9. Stretch: LoRA fine-tune on Colab (free)
- Open Unsloth's Qwen2.5 conversational notebook on Colab (free T4 works for 3B).
- Upload sft_data.jsonl, point the dataset loader at it, train (~30-60 min).
- Export merged model as GGUF q4_k_m, download, then on the Mac:
  `ollama create qwen2.5-lab -f Modelfile` (FROM ./model.gguf)
- Change `model:` in config.yaml to `qwen2.5-lab`, re-run `run_evals.py`.
- **Before/after score on your own benchmark = the portfolio headline.**

If 50-100 trajectories isn't enough to move the number — that's a legitimate
finding too. Write it up honestly, same as Phase 1 of the DNS project.

## Realistic expectations for 2 days
- Day 1 fully done = already portfolio-worthy (local agent + benchmark + analysis).
- Day 2 SFT is genuinely stretch. If EVE-NG mgmt networking eats half of day 1
  (it might), push training to day 3 and don't sweat it.

## Guardrail notes (interview-ready talking points)
- Read-only allowlist enforced in code, not in the prompt — the model is
  untrusted input, the executor is the security boundary.
- Output truncation = context management for small models.
- Verifiable rewards: every eval check is programmatic because you control
  ground truth. This is the same property that makes the lab viable for RL later.
