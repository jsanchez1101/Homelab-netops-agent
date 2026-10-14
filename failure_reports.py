"""Create readable reports from failed evaluation trajectories."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent
TRAJECTORY_PATH = PROJECT_DIR / "trajectories" / "trajectories.jsonl"
REPORT_DIR = PROJECT_DIR / "reports"
MAX_OUTPUT_CHARS = 1200


def tool_steps(messages):
    calls = {}
    steps = []
    for message in messages or []:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                try:
                    args = json.loads(call.get("function", {}).get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"raw_arguments": call.get("function", {}).get("arguments")}
                calls[call.get("id")] = {
                    "name": call.get("function", {}).get("name", "unknown"),
                    "device": args.get("device", ""),
                    "command": args.get("command", ""),
                }
        elif message.get("role") == "tool":
            call = calls.get(message.get("tool_call_id"), {})
            output = message.get("content") or ""
            if len(output) > MAX_OUTPUT_CHARS:
                output = output[:MAX_OUTPUT_CHARS] + "\n...[report truncated]"
            steps.append({**call, "output": output})
    return steps


def failure_entry(task, result, reason, run_number):
    return {
        "run": run_number,
        "task_id": task["id"],
        "difficulty": task.get("difficulty", "?"),
        "question": task["task"],
        "reason": reason or "unknown",
        "iterations": result.get("iterations", 0),
        "answer": result.get("answer") or "<NO ANSWER>",
        "steps": tool_steps(result.get("messages", [])),
    }


def entry_from_trajectory(record, task, run_number):
    result = {
        "answer": record.get("answer"),
        "iterations": record.get("iterations", 0),
        "messages": record.get("messages", []),
    }
    return failure_entry(task, result, record.get("failure_reason"), run_number)


def write_failure_report(suite_name, failures, scores=None, timestamp=None, output=None):
    REPORT_DIR.mkdir(exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    output = Path(output) if output else REPORT_DIR / f"{Path(suite_name).stem}-{timestamp}-failures.txt"

    lines = [
        "EVALUATION FAILURE REPORT",
        "=" * 80,
        f"Suite: {suite_name}",
        f"Generated: {timestamp}",
        f"Scores: {scores if scores is not None else 'reconstructed from trajectories'}",
        f"Failed attempts: {len(failures)}",
        "",
    ]
    for failure in failures:
        lines.extend([
            "-" * 80,
            f"Run {failure['run']} | {failure['task_id']} | {failure['difficulty']}",
            f"Reason: {failure['reason']}",
            f"Question: {failure['question']}",
            "",
        ])
        if failure["steps"]:
            for number, step in enumerate(failure["steps"], 1):
                lines.extend([
                    f"Tool step {number}: {step.get('name', 'unknown')}",
                    f"Device: {step.get('device') or '<none>'}",
                    f"Command: {step.get('command') or '<none>'}",
                    "Router output:",
                    step.get("output") or "<NO OUTPUT>",
                    "",
                ])
        else:
            lines.extend(["Tool steps: <NONE>", ""])
        lines.extend(["Final answer:", failure["answer"], ""])

    output.write_text("\n".join(lines) + "\n")
    return output


def reconstruct_report(tasks_file, runs, output=None):
    suite_path = PROJECT_DIR / "evals" / tasks_file
    suite = yaml.safe_load(suite_path.read_text())
    tasks = {task["id"]: task for task in suite["tasks"]}
    split = suite.get("split", "train")
    records = [json.loads(line) for line in TRAJECTORY_PATH.read_text().splitlines() if line.strip()]
    matching = [
        record for record in records
        if record.get("split") == split and record.get("task_id") in tasks
    ]
    needed = runs * len(tasks)
    selected = matching[-needed:]
    if len(selected) != needed:
        raise SystemExit(f"Expected {needed} recent records for {tasks_file}, found {len(selected)}")

    failures = []
    for index, record in enumerate(selected):
        run_number = index // len(tasks) + 1
        if not record.get("success"):
            failures.append(entry_from_trajectory(record, tasks[record["task_id"]], run_number))
    return write_failure_report(tasks_file, failures, output=output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="tasks.yaml")
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    path = reconstruct_report(args.tasks, args.runs, args.output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
