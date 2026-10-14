from fastapi import FastAPI as fastapi
from fastapi import HTTPException, Request
from fastapi.templating import Jinja2Templates
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

app = fastapi()
DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEMO_DIR.parent
templates = Jinja2Templates(directory=str(DEMO_DIR / "templates"))

 
def extract(task_id, pass_counts):
    return pass_counts.get(task_id, 0)


def div(value, divisor):
    return value / divisor if divisor else 0


def mul(value, multiplier):
    return value * multiplier


templates.env.filters["extract"] = extract
templates.env.filters["div"] = div
templates.env.filters["mul"] = mul

MAC_ADDRESS = re.compile(r"(?i)\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b")


def eastern_timestamp(value):
    if not value:
        return ""
    captured = datetime.fromisoformat(value)
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=timezone.utc)
    return captured.astimezone(ZoneInfo("America/New_York")).strftime(
        "%Y-%m-%d %I:%M:%S %p %Z"
    )


def load_trajectory_records():
    path = PROJECT_DIR / "trajectories" / "trajectories.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def legacy_baseline_records(result, records):
    """Recover an old unlinked session only when sequence and scores match exactly."""
    task_ids = [task["id"] for task in result.get("tasks", [])]
    n_runs = result.get("n_runs", 0)
    expected_sequence = task_ids * n_runs
    window_size = len(expected_sequence)
    if not task_ids or not window_size:
        return []

    candidates = []
    for start in range(len(records) - window_size + 1):
        window = records[start:start + window_size]
        if [record.get("task_id") for record in window] != expected_sequence:
            continue
        scores = [
            sum(
                bool(record.get("success"))
                for record in window[index * len(task_ids):(index + 1) * len(task_ids)]
            )
            for index in range(n_runs)
        ]
        if scores == result.get("scores"):
            candidates.append(window)
    if not candidates:
        return []
    window = candidates[-1]
    return [
        {
            "run": index + 1,
            "tasks": {
                record["task_id"]: record
                for record in window[index * len(task_ids):(index + 1) * len(task_ids)]
            },
        }
        for index in range(n_runs)
    ]


def linked_baseline_records(result, records):
    records_by_id = {record.get("id"): record for record in records if record.get("id")}
    references = result.get("trajectories")
    if isinstance(references, list) and references:
        linked = []
        for run_ref in references:
            task_records = {
                task_id: records_by_id[trajectory_id]
                for task_id, trajectory_id in run_ref.get("tasks", {}).items()
                if trajectory_id in records_by_id
            }
            linked.append({"run": run_ref.get("run"), "tasks": task_records})
        if any(item["tasks"] for item in linked):
            return linked, "recorded trajectory IDs"
    legacy = legacy_baseline_records(result, records)
    return legacy, "reconstructed legacy session" if legacy else "unavailable"


def sanitized_output(value, limit=1200):
    text = MAC_ADDRESS.sub("<mac-address>", value or "")
    if len(text) > limit:
        return text[:limit].rstrip() + f"\n…[truncated at {limit} characters]"
    return text


def tool_observations(messages):
    pending = {}
    observations = []
    for message in messages or []:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                pending[call.get("id")] = {
                    "name": function.get("name", "unknown tool"),
                    "device": arguments.get("device", ""),
                    "command": arguments.get("command", ""),
                }
        elif message.get("role") == "tool":
            call = pending.get(message.get("tool_call_id"))
            if call:
                observations.append({
                    **call,
                    "output": sanitized_output(message.get("content") or ""),
                })
    return observations


def display_attempt(record, run_number):
    reason = record.get("failure_reason")
    success = bool(record.get("success"))
    answer = (record.get("answer") or "").strip()
    if success:
        answer_status, evidence_status = "pass", "pass"
        summary = "The final answer and its required live evidence both passed."
    elif reason == "missing_or_invalid_evidence":
        answer_status, evidence_status = "pass", "fail"
        summary = "The answer matched, but the required command evidence was missing or invalid."
    else:
        answer_status, evidence_status = "fail", "not-evaluated"
        summary = (
            "The model returned no final answer."
            if not answer else
            "The final answer did not match the expected answer; evidence grading stopped there."
        )
    return {
        "run": run_number,
        "success": success,
        "failure_reason": reason,
        "answer": answer or "<NO ANSWER>",
        "answer_status": answer_status,
        "evidence_status": evidence_status,
        "summary": summary,
        "iterations": record.get("iterations", 0),
        "empty_response_retries": record.get("empty_response_retries", 0),
        "prompt_hash": (record.get("prompt_sha256") or "")[:10],
        "commands": tool_observations(record.get("messages")),
    }


def task_attempts(result, records):
    linked_runs, source = linked_baseline_records(result, records)
    attempts = {task["id"]: [] for task in result.get("tasks", [])}
    for linked_run in linked_runs:
        for task_id, record in linked_run["tasks"].items():
            attempts.setdefault(task_id, []).append(
                display_attempt(record, linked_run["run"])
            )
    return attempts, source


def load_results():
    with (PROJECT_DIR / "eval_runs.json").open() as f:
        data = json.load(f)
    tasks_path = PROJECT_DIR / "evals" / "tasks.yaml"
    suite_hash = hashlib.sha256(tasks_path.read_bytes()).hexdigest()
    valid = [
        session for session in data
        if session.get("valid")
        and session.get("tasks_file") == "tasks.yaml"
        and session.get("split") == "train"
        and session.get("n_runs", 0) >= 10
        and session.get("n_tasks") == 12
        and session.get("suite_sha256") == suite_hash
    ]
    if not valid:
        raise HTTPException(
            status_code=503,
            detail="No valid post-repair baseline yet. Run the corrected 10-run suite.",
        )
    result = dict(valid[-1])
    result["display_timestamp"] = eastern_timestamp(result.get("timestamp"))
    result["task_attempts"], result["trajectory_source"] = task_attempts(
        result, load_trajectory_records()
    )
    return result

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "r": load_results(),
    })
