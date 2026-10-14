"""
Minimal network lab agent. No frameworks.


Loop: prompt -> model returns tool call -> validate against allowlist ->
execute via netmiko against EVE-NG devices -> feed output back -> repeat
until final answer or max iterations.

Talks to any OpenAI-compatible endpoint (Ollama by default).
Swapping to a cloud API = change BASE_URL/API_KEY/MODEL in config.yaml.
"""

import json
import hashlib
import os
import re
import sys
import time
import uuid
from pathlib import Path
import ipaddress
from urllib.parse import urlparse


import requests
import yaml

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_NAME = os.environ.get("LAB_AGENT_CONFIG")
if CONFIG_NAME is None:
    CONFIG_NAME = "config.local.yaml" if (PROJECT_DIR / "config.local.yaml").is_file() else "config.yaml"
if Path(CONFIG_NAME).name != CONFIG_NAME or not CONFIG_NAME.endswith(".yaml"):
    raise ValueError("LAB_AGENT_CONFIG must name a YAML file in the project directory")
CONFIG_PATH = PROJECT_DIR / CONFIG_NAME
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Agent configuration not found: {CONFIG_PATH}")
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())

INVENTORY_NAME = os.environ.get("LAB_AGENT_INVENTORY", "devices.yaml")
if Path(INVENTORY_NAME).name != INVENTORY_NAME or not INVENTORY_NAME.endswith(".yaml"):
    raise ValueError("LAB_AGENT_INVENTORY must name a YAML file in the project directory")
INVENTORY_PATH = PROJECT_DIR / INVENTORY_NAME
if not INVENTORY_PATH.is_file():
    raise FileNotFoundError(f"Device inventory not found: {INVENTORY_PATH}")
DEVICES = yaml.safe_load(INVENTORY_PATH.read_text())["devices"]

BASE_URL = CONFIG["base_url"].rstrip("/")
MODEL = CONFIG["model"]
API_KEY = CONFIG.get("api_key", "ollama")  # Ollama ignores this
MAX_ITERATIONS = CONFIG.get("max_iterations", 8)
MAX_TOOL_OUTPUT_CHARS = CONFIG.get("max_tool_output_chars", 2000)
MAX_EMPTY_RESPONSE_RETRIES = CONFIG.get("max_empty_response_retries", 1)
TRAJ_DIR = PROJECT_DIR / "trajectories"

def local_endpoint_scope(base_url: str) -> str:
    """Accept only loopback, private, or Tailscale model endpoints."""
    parsed = urlparse(base_url)

    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("Model endpoint must be an explicit local/private HTTP URL")

    if parsed.hostname == "localhost":
        return "loopback"

    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as exc:
        raise ValueError("Model endpoint must use a literal local/private IP address.") from exc

    if address.is_loopback:
        return "loopback"

    if address in ipaddress.ip_network("100.64.0.0/10"):
        return "tailscale"

    if address.is_private:
        return "private"

    raise ValueError("Public model endpoints are disabled for grounded evals.")

MODEL_ENDPOINT_SCOPE = local_endpoint_scope(BASE_URL)
MODEL_ENDPOINT_SHA256 = hashlib.sha256(BASE_URL.encode()).hexdigest()

MODEL_HTTP = requests.Session()
MODEL_HTTP.trust_env = False


# ---------------------------------------------------------------- tools ---

# Match the entire command. A prefix-only check is unsafe because VyOS executes
# operational commands through a shell, where suffixes such as ``; curl ...``
# or ``| sh`` would otherwise be interpreted.
ALLOWED_COMMAND_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"show version",
        r"show interfaces",
        r"show interfaces ethernet eth\d+",
        r"show ip ospf",
        r"show ip ospf neighbor(?: detail)?",
        r"show ip ospf interface(?: eth\d+)?",
        r"show ip ospf database(?: router)?",
        r"show ip route ospf",
        r"show ip route (?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?",
    )
)
SHELL_METACHARACTERS = frozenset("\n\r\x00;&|`$><(){}[]\\")


def list_devices() -> str:
    """Return the device inventory the agent is allowed to touch."""
    lines = [f"- {name}: {d.get('description', d['device_type'])}" for name, d in DEVICES.items()]
    return "Available devices:\n" + "\n".join(lines)


def run_show_command(device: str, command: str) -> str:
    """Run a read-only 'show' command on a lab device via SSH."""
    command = command.strip()
    if device not in DEVICES:
        return f"ERROR: unknown device '{device}'. Use list_devices to see valid names."
    if any(ch in command for ch in ("\n", "\r", "\x00")):
        return "ERROR: command rejected. Commands must contain exactly one CLI line."
    if any(ch in command for ch in SHELL_METACHARACTERS):
        return "ERROR: command rejected. Shell syntax and command chaining are not permitted."
    if not any(pattern.fullmatch(command) for pattern in ALLOWED_COMMAND_PATTERNS):
        return "ERROR: command rejected. Command is not in the approved read-only allowlist."

    d = DEVICES[device]
    conn = None
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
    except Exception as e:
        return f"ERROR: connection/command failed: {type(e).__name__}: {e}"
    finally:
        if conn is not None:
            try:
                conn.disconnect()
            except Exception:
                pass

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
            "description": "Run one approved read-only VyOS show command on a named lab device and return its output. Shell operators, pipes, redirects, substitutions, and command chaining are forbidden.",
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
- If a command fails or does not contain the requested fact, try at most one sensible alternative command.
- If the permitted commands still cannot establish the answer, say "I cannot determine that from the available evidence" and state what evidence is missing. Giving up is acceptable.
- Never invent missing device state or attempt to bypass the tool restrictions.
- Be economical: each tool call is expensive. Plan which single command answers the question.
- Devices are VyOS routers. Use VyOS operational commands: 'show ip ospf neighbor', 'show ip route ospf', 'show interfaces', and 'show ip ospf interface eth1'.
- For a router's own OSPF router ID, use 'show ip ospf'. 'Neighbor ID' identifies a remote router, not the local router.
- In 'show ip ospf interface', State describes the local router's OSPF role. 'State DR' means it is the Designated Router; 'State Backup' or 'State DROther' means it is not. Compare the local Internet Address with the Designated Router Interface Address to identify the DR.
- For a single interface's status, use 'show interfaces ethernet <name>' (for example, 'show interfaces ethernet eth1') — not 'show interfaces <name>'.
- Keep commands specific to avoid huge output; never use configuration mode.
- When you have the answer, reply in plain text without calling another tool. State the answer in the first sentence.

{list_devices()}
"""

# ----------------------------------------------------------------- loop ---


def chat(messages):
    resp = MODEL_HTTP.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": messages, "tools": TOOL_SCHEMAS, "temperature": 0.2},
        timeout=300,
        allow_redirects=False,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def run_task(task: str, verbose: bool = True) -> dict:
    """Run one task. Returns {'answer': str|None, 'messages': [...], 'iterations': int}."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    empty_response_retries = 0
    for i in range(1, MAX_ITERATIONS + 1):
        msg = chat(messages)
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            answer = (msg.get("content") or "").strip()
            if not answer and empty_response_retries < MAX_EMPTY_RESPONSE_RETRIES:
                empty_response_retries += 1
                if verbose:
                    print(
                        f"[empty model response after turn {i}; retry "
                        f"{empty_response_retries}/{MAX_EMPTY_RESPONSE_RETRIES}]"
                    )
                continue
            if verbose:
                print(f"\n[final answer after {i} turn(s)]\n{answer}")
            return {
                "answer": answer,
                "messages": messages + [msg],
                "iterations": i,
                "empty_response_retries": empty_response_retries,
            }

        for tc in tool_calls:
            if not tc.get("id"):
                tc["id"] = f"call_{uuid.uuid4().hex}"
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
                "tool_call_id": tc["id"],
                "content": result,
            })

    return {
        "answer": None,
        "messages": messages,
        "iterations": MAX_ITERATIONS,
        "empty_response_retries": empty_response_retries,
    }


def save_trajectory(
    task: str,
    result: dict,
    success: bool | None = None,
    *,
    task_id: str | None = None,
    split: str | None = None,
    failure_reason: str | None = None,
):
    TRAJ_DIR.mkdir(exist_ok=True)
    record = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "model": MODEL,
        "model_endpoint_scope": MODEL_ENDPOINT_SCOPE,
        "model_endpoint_sha256": MODEL_ENDPOINT_SHA256,
        "inventory": INVENTORY_NAME,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "task_id": task_id,
        "split": split,
        "task": task,
        "success": success,
        "failure_reason": failure_reason,
        "answer": result.get("answer"),
        "iterations": result["iterations"],
        "empty_response_retries": result.get("empty_response_retries", 0),
        "messages": result["messages"],
    }
    out = TRAJ_DIR / "trajectories.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record["id"]


def main():
    # If prompt was passed via CLI -> run it once and save trajectory
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        result = run_task(task)
        tid = save_trajectory(task, result)
        print(f"\n[trajectory saved: {tid}]")
        return

    # Otherwise -> start an interactive loop
    print("Agent started. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("\nAgent> ")
            if prompt.lower() in ["exit", "quit"]:
                break
            if not prompt.strip():
                continue

            # Run the task and save the trajectory for each chat turn
            result = run_task(prompt)
            tid = save_trajectory(prompt, result)
            print(f"\n[trajectory saved: {tid}]")

        except KeyboardInterrupt:
            # Catches Ctrl+C cleanly
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
