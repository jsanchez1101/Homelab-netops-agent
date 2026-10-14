"""
Eval harness. Runs every task in evals/tasks.yaml through the agent,
checks the answer against ground truth, prints a scoreboard, and saves
every trajectory (tagged success/fail) for later SFT data filtering.

Usage:
    python run_evals.py             # run all tasks
    python run_evals.py t01 t05     # run specific task ids
"""

import re
import json
import sys
import time
from pathlib import Path

import yaml

from agent import INVENTORY_PATH, PROJECT_DIR, run_task, save_trajectory
from benchmark_freeze import verify_freeze


def ensure_neutral_inventory(inventory_path: Path = INVENTORY_PATH):
    expected = PROJECT_DIR / "devices.yaml"
    if inventory_path.resolve() != expected.resolve():
        raise SystemExit(
            "Evaluation refused: LAB_AGENT_INVENTORY must be unset so the neutral "
            "devices.yaml inventory is used. Rich smoke metadata would leak answers."
        )


def lead_line(answer: str) -> str:
    """Return the first non-empty line, where the prompt requires the answer."""
    return next((line.strip() for line in answer.splitlines() if line.strip()), "")


ABSTENTION_RE = re.compile(
    r"(?i)\b(?:"
    r"i (?:cannot|can't|am unable to) (?:determine|answer)|"
    r"(?:cannot|can't) be determined|"
    r"insufficient evidence|"
    r"(?:do not|don't) have enough evidence|"
    r"not enough evidence|"
    r"i (?:do not|don't) know"
    r")\b"
)


def is_abstention(answer: str | None) -> bool:
    if not answer:
        return False
    return ABSTENTION_RE.search(lead_line(answer)) is not None


def _device_role_answer(answer: str, expected_device: str) -> bool:
    """Require one unambiguous DR claim on the first meaningful line."""
    lead = lead_line(answer)
    claims = set()

    for pattern in (
        r"(?i)\b(?:router\s+)?(R\d+)\s+is\s+"
        r"(?:the\s+)?(?:OSPF\s+)?(?:designated router|DR)\b",
        r"(?i)(?:designated router\b(?:\s*\(DR\))?|\bDR\b).*?"
        r"\bis\s+(?:router\s+)?(R\d+)\b",
    ):
        claims.update(match.upper() for match in re.findall(pattern, lead))

    if not claims:
        short_answer = re.fullmatch(
            r"(?i)\s*(?:router\s+)?(R\d+)[.!]?\s*",
            lead,
        )
        if short_answer:
            claims.add(short_answer.group(1).upper())

    return claims == {expected_device.upper()}


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
    lead = lead_line(answer)
    if ctype == "lead_regex":
        return re.search(expect, lead) is not None
    if ctype == "lead_contains_all":
        lowered = lead.lower()
        return all(str(e).lower() in lowered for e in expect)
    if ctype == "device_role_answer":
        return _device_role_answer(answer, str(expect))
    if ctype == "integer_answer":
        # Device labels such as R1/R2 are identifiers, not numeric answers.
        # Keep the check strict for every other standalone number.
        without_device_labels = re.sub(r"(?i)\bR\d+\b", "", lead)
        numbers = [
            int(n)
            for n in re.findall(r"(?<![.\d])\d+(?![.\d])", without_device_labels)
        ]
        return numbers == [int(expect)]
    if ctype == "lead_ip_set":
        addresses = set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", lead))
        return addresses == set(expect)
    if ctype == "ip_set":
        addresses = set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", answer))
        return addresses == set(expect)
    if ctype == "affirmative_contains_all":
        if re.search(r"(?i)^\s*no\b", lead):
            return False
        affirmative = re.search(
            r"(?i)\b(?:yes|can reach|has (?:a )?route)\b", answer
        )
        lowered = answer.lower()
        return bool(affirmative) and all(
            str(value).lower() in lowered for value in expect
        )
    raise ValueError(f"unknown check type: {ctype}")


def tool_observations(messages: list[dict]) -> list[dict]:
    """Join assistant tool-call arguments to their tool responses."""
    pending = {}
    observations = []
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                try:
                    args = json.loads(call.get("function", {}).get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                pending[call.get("id")] = {
                    "name": call.get("function", {}).get("name"),
                    "device": args.get("device", ""),
                    "command": args.get("command", ""),
                }
        elif message.get("role") == "tool":
            call = pending.get(message.get("tool_call_id"))
            if call:
                observations.append({**call, "output": message.get("content") or ""})
    return observations

def _ospf_neighbor_ids(output: str) -> set[str]:
    """Extract the complete neighbor-ID set from a VyOS OSPF table."""
    ipv4 = r"(?:\d{1,3}\.){3}\d{1,3}"

    return set(
        re.findall(
            rf"(?m)^\s*({ipv4})\s+\d+\s+\S+",
            output,
        )
    )


def _typed_output_matches(output: str, rule: dict) -> bool:
    """Run parser-backed checks that require exact structured evidence."""
    validator = rule.get("validator")

    if validator == "ospf_neighbor_id_set":
        observed = _ospf_neighbor_ids(output)
        expected = set(rule["expect"])
        return observed == expected

    if validator == "ospf_via_route_prefix_set":
        ipv4 = r"(?:\d{1,3}\.){3}\d{1,3}"
        observed = set(
            re.findall(
                rf"(?m)^O[^\n]*?\s({ipv4}/\d+)\s+[^\n]*\bvia\b",
                output,
            )
        )
        expected = set(rule["expect"])
        return observed == expected

    if validator == "route_via":
        prefix = re.escape(str(rule["prefix"]))
        next_hop = re.escape(str(rule["via"]))
        return re.search(
            rf"(?m)^.*\b{prefix}\b[^\n]*\bvia\s+{next_hop}\b",
            output,
        ) is not None

    if validator == "ospf_database_router_id_set":
        ipv4 = r"(?:\d{1,3}\.){3}\d{1,3}"
        observed = set()
        in_router_section = False

        for line in output.splitlines():
            if re.search(r"(?i)Router Link States", line):
                in_router_section = True
                continue
            if in_router_section and re.search(r"(?i)\bLink States\b", line):
                break
            if not in_router_section:
                continue
            row = re.match(rf"^\s*({ipv4})\s+({ipv4})(?:\s+|$)", line)
            if row:
                observed.add(row.group(2))

        return observed == set(rule["expect"])

    if validator == "local_ospf_router_id":
        ipv4 = r"(?:\d{1,3}\.){3}\d{1,3}"
        match = re.search(
            rf"(?im)^\s*Router ID\s*:?\s*({ipv4})\b",
            output,
        )
        return bool(match) and match.group(1) == str(rule["expect"])

    raise ValueError(f"unknown evidence validator: {validator}")
def _rule_matches(observation: dict, rule: dict) -> bool:
    output = observation.get("output", "")
    if output.lstrip().startswith("ERROR:"):
        return False
    if rule.get("device") and observation.get("device") != rule["device"]:
        return False
    if rule.get("command") and not re.fullmatch(
        rule["command"], observation.get("command", ""), re.IGNORECASE
    ):
        return False
    if rule.get("output") and not re.search(rule["output"], output):
        return False
    if rule.get("validator") and not _typed_output_matches(output, rule):
        return False
    return True


def check_evidence(messages: list[dict], evidence: dict | None) -> bool:
    if not evidence:
        return True
    if evidence.get("alternatives"):
        return any(
            check_evidence(messages, alternative)
            for alternative in evidence["alternatives"]
        )
    observations = tool_observations(messages)
    all_rules = evidence.get("all", [])
    any_rules = evidence.get("any", [])
    if all_rules and not all(
        any(_rule_matches(observation, rule) for observation in observations)
        for rule in all_rules
    ):
        return False
    if any_rules and not any(
        _rule_matches(observation, rule)
        for rule in any_rules
        for observation in observations
    ):
        return False
    return True


def evaluate(task: dict, result: dict) -> tuple[bool, str | None]:
    answer = result.get("answer")
    if is_abstention(answer):
        return False, "abstained"
    if not check(answer, task["check"], task["expect"]):
        return False, "wrong_answer"
    if not check_evidence(result.get("messages", []), task.get("evidence")):
        return False, "missing_or_invalid_evidence"
    return True, None


def main():
    verify_freeze()
    ensure_neutral_inventory()
    tasks = yaml.safe_load((Path(__file__).parent / "evals" / "tasks.yaml").read_text())["tasks"]
    only = set(sys.argv[1:])
    if only:
        tasks = [t for t in tasks if t["id"] in only]

    results = []
    t0 = time.time()
    for t in tasks:
        print(f"\n{'-'*60}\n[{t['id']}] ({t['difficulty']}) {t['task']}")
        try:
            r = run_task(t["task"], verbose=True)
        except Exception as e:
            print(f"  RUN ERROR: {e}")
            r = {"answer": None, "messages": [], "iterations": 0}
        ok, reason = evaluate(t, r)
        save_trajectory(
            t["task"], r, success=ok, task_id=t["id"], split="train",
            failure_reason=reason,
        )
        results.append((t, ok, r["iterations"]))
        print(f"  -> {'PASS' if ok else f'FAIL ({reason})'}")

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
