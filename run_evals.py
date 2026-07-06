"""
Eval harness. Runs every task in evals/tasks.yaml through the agent,
checks the answer against ground truth, prints a scoreboard, and saves
every trajectory (tagged success/fail) for later SFT data filtering.

Usage:
    python run_evals.py             # run all tasks
    python run_evals.py t01 t05     # run specific task ids
"""

import re
import sys
import time
from pathlib import Path

import yaml

from agent import run_task, save_trajectory


def check(answer: str, ctype: str, expect) -> bool:
    if answer is None:
        return False
    a = answer.lower()
    if ctype == "contains":
        return str(expect).lower() in a
    if ctype == "contains_all":
        return all(str(e).lower() in a for e in expect)
    if ctype == "regex":
        return re.search(expect, answer, re.IGNORECASE) is not None
    raise ValueError(f"unknown check type: {ctype}")


def main():
    tasks = yaml.safe_load((Path(__file__).parent / "evals" / "tasks.yaml").read_text())["tasks"]
    only = set(sys.argv[1:])
    if only:
        tasks = [t for t in tasks if t["id"] in only]

    results = []
    t0 = time.time()
    for t in tasks:
        print(f"\n{'='*60}\n[{t['id']}] ({t['difficulty']}) {t['task']}")
        try:
            r = run_task(t["task"], verbose=True)
        except Exception as e:
            print(f"  RUN ERROR: {e}")
            r = {"answer": None, "messages": [], "iterations": 0}
        ok = check(r["answer"], t["check"], t["expect"])
        save_trajectory(t["task"], r, success=ok)
        results.append((t, ok, r["iterations"]))
        print(f"  -> {'PASS' if ok else 'FAIL'}")

    elapsed = time.time() - t0
    n_pass = sum(ok for _, ok, _ in results)
    print(f"\n{'='*60}\nSCOREBOARD  ({elapsed:.0f}s total)")
    for t, ok, iters in results:
        print(f"  {t['id']}  {'PASS' if ok else 'FAIL'}  ({t['difficulty']}, {iters} turns)")
    by_diff = {}
    for t, ok, _ in results:
        by_diff.setdefault(t["difficulty"], []).append(ok)
    for d, oks in sorted(by_diff.items()):
        print(f"  {d}: {sum(oks)}/{len(oks)}")
    print(f"\nTOTAL: {n_pass}/{len(results)} = {100*n_pass/len(results):.0f}%")
    print("Trajectories appended to trajectories/trajectories.jsonl (tagged success/fail).")


if __name__ == "__main__":
    main()
