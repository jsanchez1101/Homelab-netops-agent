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
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from agent import run_task, save_trajectory
from run_evals import check

TASKS_PATH = Path(__file__).with_name("evals") / "tasks.yaml"
HISTORY_PATH = Path(__file__).with_name("eval_runs.json")


def load_tasks():
    return yaml.safe_load(TASKS_PATH.read_text())["tasks"]


def one_run(tasks, verbose):
    """Run every task once. Returns {task_id: bool} plus timing."""
    results = {}
    t0 = time.time()
    for t in tasks:
        try:
            r = run_task(t["task"], verbose=verbose)
        except Exception as e:
            print(f"    [{t['id']}] RUN ERROR: {type(e).__name__}: {e}")
            r = {"answer": None, "messages": [], "iterations": 0}
        ok = check(r["answer"], t["check"], t["expect"])
        save_trajectory(t["task"], r, success=ok)
        results[t["id"]] = ok
        if not verbose:
            print(f"    {t['id']}: {'PASS' if ok else 'FAIL'}", flush=True)
    return results, time.time() - t0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    quiet = "--quiet" in sys.argv or "-q" in sys.argv
    n_runs = int(args[0]) if args else 5

    tasks = load_tasks()
    by_difficulty = {t["id"]: t.get("difficulty", "?") for t in tasks}
    total_tasks = len(tasks)

    print(f"Running the {total_tasks}-task benchmark {n_runs}x "
          f"({'quiet' if quiet else 'verbose'} mode)\n")

    all_runs = []
    for i in range(1, n_runs + 1):
        print(f"--- run {i}/{n_runs} ---", flush=True)
        results, elapsed = one_run(tasks, verbose=not quiet)
        score = sum(results.values())
        all_runs.append(results)
        print(f"    run {i}: {score}/{total_tasks} ({elapsed:.0f}s)\n", flush=True)

    # ---------------- aggregate ----------------
    scores = [sum(r.values()) for r in all_runs]
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0

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
        "n_runs": n_runs,
        "n_tasks": total_tasks,
        "scores": scores,
        "mean": round(mean, 2),
        "stdev": round(stdev, 2),
        "per_task_passes": dict(per_task),
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

    if len(history) > 1:
        prev = history[-2]
        delta = record["mean"] - prev["mean"]
        print(f"Previous session mean: {prev['mean']}/{prev['n_tasks']} "
              f"({prev['timestamp']})  ->  delta {delta:+.2f}")
        if abs(delta) < record["stdev"]:
            print("Note: change is smaller than run-to-run stdev — not yet a real signal.")


if __name__ == "__main__":
    main()