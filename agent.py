"""
Minimal network lab agent. No frameworks.

Loop: prompt -> model returns tool call -> validate against allowlist ->
execute via netmiko against EVE-NG devices -> feed output back -> repeat
until final answer or max iterations.

Talks to any OpenAI-compatible endpoint (Ollama by default).
Swapping to a cloud API = change BASE_URL/API_KEY/MODEL in config.yaml.
"""

import json
import sys
import time
import uuid
from pathlib import Path

import requests
import yaml

CONFIG = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
DEVICES = yaml.safe_load(Path(__file__).with_name("devices.yaml").read_text())["devices"]

BASE_URL = CONFIG["base_url"].rstrip("/")
MODEL = CONFIG["model"]
API_KEY = CONFIG.get("api_key", "ollama")  # Ollama ignores this
MAX_ITERATIONS = CONFIG.get("max_iterations", 8)
MAX_TOOL_OUTPUT_CHARS = CONFIG.get("max_tool_output_chars", 2000)
TRAJ_DIR = Path(__file__).with_name("trajectories")

# ---------------------------------------------------------------- tools ---

ALLOWED_PREFIXES = ("show ",)  # read-only. Expand deliberately, never wildcard.


def list_devices() -> str:
    """Return the device inventory the agent is allowed to touch."""
    lines = [f"- {name}: {d.get('description', d['device_type'])}" for name, d in DEVICES.items()]
    return "Available devices:\n" + "\n".join(lines)


def run_show_command(device: str, command: str) -> str:
    """Run a read-only 'show' command on a lab device via SSH."""
    command = command.strip()
    if device not in DEVICES:
        return f"ERROR: unknown device '{device}'. Use list_devices to see valid names."
    if not command.lower().startswith(ALLOWED_PREFIXES):
        return f"ERROR: command rejected by allowlist. Only commands starting with {ALLOWED_PREFIXES} are permitted."

    d = DEVICES[device]
    try:
        from netmiko import ConnectHandler
        conn = ConnectHandler(
            device_type=d["device_type"],
            host=d["host"],
            username=d["username"],
            password=d["password"],
            timeout=15,
        )
        output = conn.send_command(command, read_timeout=20)
        conn.disconnect()
    except Exception as e:
        return f"ERROR: connection/command failed: {type(e).__name__}: {e}"

    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + f"\n...[truncated at {MAX_TOOL_OUTPUT_CHARS} chars]"
    return output or "(command returned no output)"


TOOL_IMPLS = {
    "list_devices": lambda **kw: list_devices(),
    "run_show_command": lambda **kw: run_show_command(kw.get("device", ""), kw.get("command", "")),
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_devices",
            "description": "List the lab devices you can query.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_show_command",
            "description": "Run a read-only VyOS 'show' command on a named lab device and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {"type": "string", "description": "Device name from list_devices, e.g. 'R1'"},
                    "command": {"type": "string", "description": "A VyOS operational 'show' command, e.g. 'show ip ospf neighbor', 'show interfaces', 'show ip route ospf'"},
                },
                "required": ["device", "command"],
            },
        },
    },
]

SYSTEM_PROMPT = f"""You are a network operations agent working inside an EVE-NG lab.
You can ONLY observe; you cannot change configuration. All commands must start with 'show'.

Rules:
- Use tools to gather evidence. Never guess device state.
- Be economical: each tool call is expensive. Plan which single command answers the question.
- Devices are VyOS routers. Use VyOS operational commands: 'show ip ospf neighbor', 'show ip route ospf', 'show interfaces', 'show ip ospf interface eth1', 'show version'.
- Keep commands specific to avoid huge output; never use configuration mode.
- When you have the answer, reply in plain text WITHOUT calling a tool. State the answer in the first sentence.

{list_devices()}
"""

# ----------------------------------------------------------------- loop ---


def chat(messages):
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": messages, "tools": TOOL_SCHEMAS, "temperature": 0.2},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def run_task(task: str, verbose: bool = True) -> dict:
    """Run one task. Returns {'answer': str|None, 'messages': [...], 'iterations': int}."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    for i in range(1, MAX_ITERATIONS + 1):
        msg = chat(messages)
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            answer = (msg.get("content") or "").strip()
            if verbose:
                print(f"\n[final answer after {i} turn(s)]\n{answer}")
            return {"answer": answer, "messages": messages + [msg], "iterations": i}

        messages.append(msg)
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args, result = {}, "ERROR: malformed JSON arguments. Retry with valid JSON."
            else:
                impl = TOOL_IMPLS.get(name)
                result = impl(**args) if impl else f"ERROR: unknown tool '{name}'."
            if verbose:
                print(f"[turn {i}] {name}({args}) -> {result[:120]!r}...")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", str(uuid.uuid4())),
                "content": result,
            })

    return {"answer": None, "messages": messages, "iterations": MAX_ITERATIONS}


def save_trajectory(task: str, result: dict, success: bool | None = None):
    TRAJ_DIR.mkdir(exist_ok=True)
    record = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "model": MODEL,
        "task": task,
        "success": success,
        "iterations": result["iterations"],
        "messages": result["messages"],
    }
    out = TRAJ_DIR / "trajectories.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record["id"]


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "How many devices are in the lab, and what are their names?"
    result = run_task(task)
    tid = save_trajectory(task, result)
    print(f"\n[trajectory saved: {tid}]")
