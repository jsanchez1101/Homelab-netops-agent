"""
Run the benchmark N times and report stability, not a single lucky/unlucky score.

Why: a 3B model at temperature 0.2 gives meaningfully different results run to
run. One run is an anecdote. This reports mean, spread, and per-task pass rate
so you can tell a real change from noise.

Usage:
    python run_evals_multi.py            # 5 runs (default)
    python run_evals_multi.py 3          # 3 runs
    python run_evals_multi.py 5 --quiet  # suppress per-task chatter

Writes results to eval_runs.json (appends), so history accumulates across
sessions and you can compare before/after a change.
"""

import json
import statistics
import math
import hashlib
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from agent import BASE_URL, MODEL_HTTP, run_task, save_trajectory
from benchmark_freeze import verify_freeze
from failure_reports import failure_entry, write_failure_report
from run_evals import ensure_neutral_inventory, evaluate

HISTORY_PATH = Path(__file__).with_name("eval_runs.json")


def ensure_model_available():
    """Refuse to score a run when the local inference service is unavailable."""
    try:
        response = MODEL_HTTP.get(
            f"{BASE_URL}/models",
            timeout=5,
            allow_redirects=False,
        )
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "Local model preflight failed; no benchmark tasks were scored"
        ) from exc


def load_suite(path):
    suite = yaml.safe_load(path.read_text())
    return suite.get("split", "train"), suite["tasks"]

def one_run(tasks, verbose, split):
    """Run every task once and retain exact trajectory IDs for the report."""
    results = {}
    failures = []
    trajectory_ids = {}
    t0 = time.time()
    for t in tasks:
        try:
            r = run_task(t["task"], verbose=verbose)
        except Exception as e:
            raise RuntimeError(
                f"Benchmark aborted during {t['id']}; the partial run was not scored"
            ) from e
        ok, reason = evaluate(t, r)
        trajectory = save_trajectory(
            t["task"], r, success=ok, task_id=t["id"], split=split,
            failure_reason=reason,
        )
        trajectory_ids[t["id"]] = trajectory
        results[t["id"]] = ok
        if not ok:
            failures.append((t, r, reason))
        if not verbose:
            suffix = "" if ok else f" ({reason})"
            print(f"    {t['id']}: {'PASS' if ok else 'FAIL'}{suffix}", flush=True)
    return results, failures, time.time() - t0, trajectory_ids

def main():
    verify_freeze()
    ensure_neutral_inventory()
    ensure_model_available()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    quiet = "--quiet" in sys.argv or "-q" in sys.argv

    tasks_file = "tasks.yaml"
    for i, a in enumerate(sys.argv):
        if a == "--tasks" and i + 1 < len(sys.argv):
            tasks_file = sys.argv[i + 1]

    args = [a for a in args if a != tasks_file]
    n_runs = int(args[0]) if args else 5

    tasks_path = Path(__file__).with_name("evals") / tasks_file
    if tasks_path.name != tasks_file or not tasks_path.exists():
        raise SystemExit(f"Invalid task suite: {tasks_file}")
    split, tasks = load_suite(tasks_path)
    by_difficulty = {t["id"]: t.get("difficulty", "?") for t in tasks}
    total_tasks = len(tasks)

    print(f"Running the {total_tasks}-task benchmark {n_runs}x "
          f"({'quiet' if quiet else 'verbose'} mode)\n")

    all_runs = []
    all_failures = []
    all_trajectories = []
    for i in range(1, n_runs + 1):
        print(f"--- run {i}/{n_runs} ---", flush=True)
        results, failures, elapsed, trajectory_ids = one_run(
            tasks, verbose=not quiet, split=split
        )
        score = sum(results.values())
        all_runs.append(results)
        all_trajectories.append({"run": i, "tasks": trajectory_ids})
        all_failures.extend(
            failure_entry(task, result, reason, i)
            for task, result, reason in failures
        )
        print(f"    run {i}: {score}/{total_tasks} ({elapsed:.0f}s)\n", flush=True)

    # ---------------- aggregate ----------------
    scores = [sum(r.values()) for r in all_runs]
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    se = stdev / math.sqrt(len(scores)) if len(scores) > 1 else None
    ci95 = [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None

    per_task = defaultdict(int)
    for r in all_runs:
        for tid, ok in r.items():
            per_task[tid] += int(ok)

    print("=" * 62)
    print(f"RESULTS OVER {n_runs} RUNS")
    print("=" * 62)
    print(f"scores: {scores}")
    print(f"mean:   {mean:.1f}/{total_tasks}  ({100*mean/total_tasks:.0f}%)"
          f"   stdev: {stdev:.2f}")
    print(f"range:  {min(scores)}–{max(scores)}")
    if ci95 is None:
        print("95% CI: unavailable with fewer than 2 runs")
    else:
        print(f"95% CI: {ci95[0]:.2f}–{ci95[1]:.2f} tasks (normal approximation)")
    print()
    print(f"{'task':<6}{'diff':<9}{'pass rate':>12}   stability")
    print("-" * 62)
    for t in tasks:
        tid = t["id"]
        passes = per_task[tid]
        rate = passes / n_runs
        if rate == 1.0:
            tag = "stable pass"
        elif rate == 0.0:
            tag = "stable fail"
        else:
            tag = "FLAKY"
        bar = "#" * passes + "." * (n_runs - passes)
        print(f"{tid:<6}{by_difficulty[tid]:<9}{passes}/{n_runs} {bar:<8}   {tag}")
    print("=" * 62)

    flaky = [t["id"] for t in tasks if 0 < per_task[t["id"]] < n_runs]
    if flaky:
        print(f"\nFlaky tasks (inconsistent across runs): {', '.join(flaky)}")
        print("These are variance, not capability — a single run can't tell you")
        print("whether a change helped. Compare means across runs instead.")

    # ---------------- persist ----------------
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "valid": True,
        "suite_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "tasks_file": tasks_file,
        "split": split,
        "n_runs": n_runs,
        "n_tasks": total_tasks,
        "tasks": [
            {key: task[key] for key in ("id", "difficulty", "task", "topology") if key in task}
            for task in tasks
        ],
        "scores": scores,
        "mean": round(mean, 2),
        "stdev": round(stdev, 2),
        "ci95": [round(x, 2) for x in ci95] if ci95 is not None else None,
        "per_task_passes": dict(per_task),
        "trajectories": all_trajectories,
    }
    history = []
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text())
        except Exception:
            history = []
    history.append(record)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))
    print(f"\nAppended to {HISTORY_PATH.name} ({len(history)} sessions recorded).")

    report_path = write_failure_report(
        tasks_file,
        all_failures,
        scores=scores,
        timestamp=datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    print(f"Failure details written to {report_path.relative_to(Path(__file__).parent)}")

    comparable = [
        item for item in history[:-1]
        if item.get("valid") is True
        and item.get("tasks_file") == tasks_file
        and item.get("n_tasks") == total_tasks
    ]
    if comparable:
        prev = comparable[-1]
        delta = record["mean"] - prev["mean"]
        print(f"Previous session mean: {prev['mean']}/{prev['n_tasks']} "
              f"({prev['timestamp']})  ->  delta {delta:+.2f}")
        prev_se = prev.get("stdev", 0.0) / math.sqrt(max(prev.get("n_runs", 1), 1))
        if se is None or prev.get("n_runs", 0) < 2:
            print("Delta interval unavailable: both sessions need at least 2 runs.")
        else:
            delta_margin = 1.96 * math.sqrt(se**2 + prev_se**2)
            print(f"Approximate 95% CI for delta: {delta-delta_margin:+.2f} to "
                  f"{delta+delta_margin:+.2f} tasks")
            if delta - delta_margin <= 0 <= delta + delta_margin:
                print("The observed change is not resolved from zero by this approximation.")


if __name__ == "__main__":
    main()
