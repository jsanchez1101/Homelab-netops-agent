"""Export verified baseline trajectories for supervised fine-tuning."""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

from agent import SYSTEM_PROMPT
from benchmark_freeze import verify_freeze
from run_evals import evaluate


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "trajectories" / "trajectories.jsonl"
OUT = ROOT / "sft_data.jsonl"
MANIFEST = ROOT / "sft_data.manifest.json"
TASKS_PATH = ROOT / "evals" / "tasks.yaml"
HISTORY_PATH = ROOT / "eval_runs.json"

MAX_PER_TASK = 8
TRAIN_SPLIT = "train"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def valid_transcript(record, expected_prompt_sha256=None):
    messages = record.get("messages") or []
    if not messages or messages[0].get("role") != "system":
        return False
    system_prompt = messages[0].get("content", "")
    if expected_prompt_sha256 and (
        record.get("prompt_sha256") != expected_prompt_sha256
        or sha256_text(system_prompt) != expected_prompt_sha256
    ):
        return False
    # Reject the original answer-bearing inventory prompt.
    if re.search(r"\bRID\s+(?:\d{1,3}\.){3}\d{1,3}\b", system_prompt):
        return False
    call_ids = {
        call.get("id")
        for message in messages
        for call in (message.get("tool_calls") or [])
        if call.get("id")
    }
    tool_ids = {
        message.get("tool_call_id")
        for message in messages
        if message.get("role") == "tool"
    }
    return bool(call_ids) and call_ids == tool_ids


def select_records(
    records,
    tasks=None,
    expected_prompt_sha256=None,
    eligible_ids=None,
):
    tasks_by_id = {task["id"]: task for task in tasks or []}
    eligible_ids = set(eligible_ids) if eligible_ids is not None else None
    groups = defaultdict(list)
    skipped = defaultdict(int)
    seen_transcripts = set()

    for record in records:
        if record.get("split") != TRAIN_SPLIT:
            skipped["non_training_or_legacy_split"] += 1
            continue
        if eligible_ids is not None and record.get("id") not in eligible_ids:
            skipped["outside_frozen_baseline"] += 1
            continue
        if not record.get("success"):
            skipped["failed_eval"] += 1
            continue
        task_id = record.get("task_id")
        if not task_id:
            skipped["missing_task_id"] += 1
            continue
        if tasks_by_id:
            task = tasks_by_id.get(task_id)
            if not task or record.get("task") != task["task"]:
                skipped["stale_or_unknown_task"] += 1
                continue
        if not valid_transcript(record, expected_prompt_sha256):
            skipped["invalid_leaky_or_stale_prompt"] += 1
            continue
        if tasks_by_id:
            result = {
                "answer": record.get("answer"),
                "messages": record.get("messages", []),
                "iterations": record.get("iterations", 0),
            }
            passed, _ = evaluate(task, result)
            if not passed:
                skipped["fails_current_grader"] += 1
                continue

        transcript_hash = hashlib.sha256(
            json.dumps(record["messages"], sort_keys=True).encode()
        ).hexdigest()
        if transcript_hash in seen_transcripts:
            skipped["duplicate_transcript"] += 1
            continue
        seen_transcripts.add(transcript_hash)
        groups[task_id].append(record)

    selected = []
    for task_id, candidates in sorted(groups.items()):
        candidates.sort(key=lambda r: (r.get("iterations", 999), -r.get("ts", 0)))
        kept = candidates[:MAX_PER_TASK]
        selected.extend(kept)
        print(f"  {len(kept)}/{len(candidates)} kept — {task_id}")
    return selected, skipped


def latest_baseline_session(suite_sha256):
    if not HISTORY_PATH.exists():
        return None
    history = json.loads(HISTORY_PATH.read_text())
    return next(
        (
            session
            for session in reversed(history)
            if session.get("valid") is True
            and session.get("tasks_file") == "tasks.yaml"
            and session.get("suite_sha256") == suite_sha256
            and session.get("trajectories")
        ),
        None,
    )


def main():
    verify_freeze()
    if not SRC.exists():
        print("No trajectories yet. Run the frozen baseline first.")
        return

    suite_sha256 = hashlib.sha256(TASKS_PATH.read_bytes()).hexdigest()
    prompt_sha256 = sha256_text(SYSTEM_PROMPT)
    session = latest_baseline_session(suite_sha256)
    if not session:
        print("No trajectory-linked session matches the current task suite.")
        print("Run the frozen baseline before exporting training data.")
        return

    eligible_ids = {
        trajectory_id
        for run in session["trajectories"]
        for trajectory_id in run["tasks"].values()
    }
    tasks = yaml.safe_load(TASKS_PATH.read_text())["tasks"]
    records = [json.loads(line) for line in SRC.read_text().splitlines() if line.strip()]
    selected, skipped = select_records(
        records,
        tasks=tasks,
        expected_prompt_sha256=prompt_sha256,
        eligible_ids=eligible_ids,
    )
    if not selected:
        print("No verified frozen-baseline trajectories found; output was not overwritten.")
        return

    temporary = OUT.with_suffix(".jsonl.tmp")
    with temporary.open("w") as output:
        for record in selected:
            output.write(json.dumps({"messages": record["messages"]}) + "\n")
    temporary.replace(OUT)

    manifest = {
        "source_session_timestamp": session["timestamp"],
        "source_session_runs": session["n_runs"],
        "suite_sha256": suite_sha256,
        "prompt_sha256": prompt_sha256,
        "selected_count": len(selected),
        "selected_ids": [record["id"] for record in selected],
        "per_task": dict(sorted(
            (task_id, sum(record["task_id"] == task_id for record in selected))
            for task_id in {record["task_id"] for record in selected}
        )),
        "skipped": dict(skipped),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nWrote {len(selected)} trajectories to {OUT.name}.")
    print(f"Wrote provenance to {MANIFEST.name}.")
    print("Skipped:", dict(skipped))


if __name__ == "__main__":
    main()
