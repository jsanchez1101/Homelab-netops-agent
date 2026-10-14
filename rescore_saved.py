"""Read-only rescoring of saved trajectories under the current graders."""

import argparse
import json
from pathlib import Path

import yaml

from run_abstention_evals import abstention_dimensions, grade_task
from run_evals import evaluate


ROOT = Path(__file__).resolve().parent
TRAJECTORIES = ROOT / "trajectories" / "trajectories.jsonl"
HISTORY = ROOT / "eval_runs.json"


def load_records() -> list[dict]:
    return [
        json.loads(line)
        for line in TRAJECTORIES.read_text().splitlines()
        if line.strip()
    ]


def as_result(record: dict) -> dict:
    return {
        "answer": record.get("answer"),
        "messages": record.get("messages", []),
        "iterations": record.get("iterations", 0),
        "run_error": record.get("run_error"),
    }


def rescore_development() -> int:
    history = json.loads(HISTORY.read_text())
    session = next(
        item
        for item in reversed(history)
        if item.get("valid")
        and item.get("tasks_file") == "tasks.yaml"
        and item.get("trajectories")
    )
    tasks = yaml.safe_load((ROOT / "evals" / "tasks.yaml").read_text())["tasks"]
    tasks_by_id = {task["id"]: task for task in tasks}
    records_by_id = {record.get("id"): record for record in load_records()}

    old_scores = session.get("scores", [])
    new_scores = []
    print(
        "DEVELOPMENT BENCHMARK — SAVED-TRAJECTORY RESCORE\n"
        f"Session: {session['timestamp']}\n"
        "No model calls; current graders applied to original answers and evidence.\n"
    )

    for run in session["trajectories"]:
        score = 0
        print(f"Run {run['run']}")
        for task_id, trajectory_id in run["tasks"].items():
            task = tasks_by_id[task_id]
            record = records_by_id[trajectory_id]
            passed, reason = evaluate(task, as_result(record))
            score += int(passed)
            old = "PASS" if record.get("success") else "FAIL"
            new = "PASS" if passed else f"FAIL ({reason})"
            changed = "  CHANGED" if bool(record.get("success")) != passed else ""
            print(f"  {task_id}: {old} -> {new}{changed}")
        new_scores.append(score)
        print(f"  score: {score}/{len(tasks)}\n")

    print(f"Recorded scores: {old_scores}")
    print(f"Rescored results: {new_scores}")
    return 0


def rescore_abstention() -> int:
    tasks = yaml.safe_load((ROOT / "evals" / "abstention.yaml").read_text())["tasks"]
    records = [
        record
        for record in load_records()
        if record.get("split") == "abstention"
    ]
    latest_by_task = {}
    for record in records:
        if record.get("task_id") in {task["id"] for task in tasks}:
            latest_by_task[record["task_id"]] = record

    missing = [task["id"] for task in tasks if task["id"] not in latest_by_task]
    if missing:
        raise SystemExit(f"Missing saved abstention trajectories: {', '.join(missing)}")

    safe = explained = controls = total = 0
    print(
        "ABSTENTION SUITE — SAVED-TRAJECTORY RESCORE\n"
        "No model calls; current graders applied to the latest saved attempt per task.\n"
    )
    for task in tasks:
        record = latest_by_task[task["id"]]
        result = as_result(record)
        passed, reason = grade_task(task, result)
        total += int(passed)
        if task["expected_behavior"] == "abstain":
            safe_refusal, useful_explanation = abstention_dimensions(task, result)
            safe += int(safe_refusal)
            explained += int(useful_explanation)
            detail = f"safe={safe_refusal}, explained={useful_explanation}"
        else:
            controls += int(passed)
            detail = "answerable control"
        old = "PASS" if record.get("success") else "FAIL"
        new = "PASS" if passed else f"FAIL ({reason})"
        changed = "  CHANGED" if bool(record.get("success")) != passed else ""
        print(f"  {task['id']}: {old} -> {new}; {detail}{changed}")

    print(f"\nSafe refusals: {safe}/3")
    print(f"Useful explanations: {explained}/3")
    print(f"Correct answerable controls: {controls}/3")
    print(f"Strict total: {total}/{len(tasks)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=("development", "abstention"))
    args = parser.parse_args()
    if args.suite == "development":
        return rescore_development()
    return rescore_abstention()


if __name__ == "__main__":
    raise SystemExit(main())
