"""Create and enforce the immutable benchmark-v1 file manifest."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "benchmark_freeze.json"
FROZEN_FILES = (
    "agent.py",
    "benchmark_freeze.py",
    "evals/tasks.yaml",
    "evals/abstention.yaml",
    "evals/holdout.yaml",
    "run_evals.py",
    "run_evals_multi.py",
    "run_abstention_evals.py",
    "export_sft.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_hashes() -> dict[str, str]:
    return {
        name: file_sha256(ROOT / name)
        for name in FROZEN_FILES
    }


def create_freeze() -> dict:
    if MANIFEST_PATH.exists():
        raise SystemExit(
            "benchmark_freeze.json already exists; do not replace a frozen benchmark "
            "without documenting a new benchmark version"
        )

    from agent import (
        MAX_EMPTY_RESPONSE_RETRIES,
        MAX_ITERATIONS,
        MAX_TOOL_OUTPUT_CHARS,
        MODEL,
        MODEL_ENDPOINT_SCOPE,
        SYSTEM_PROMPT,
        TOOL_SCHEMAS,
    )

    manifest = {
        "benchmark_version": "eval-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": current_hashes(),
        "runtime": {
            "model": MODEL,
            "model_endpoint_scope": MODEL_ENDPOINT_SCOPE,
            "max_iterations": MAX_ITERATIONS,
            "max_tool_output_chars": MAX_TOOL_OUTPUT_CHARS,
            "max_empty_response_retries": MAX_EMPTY_RESPONSE_RETRIES,
            "temperature": 0.2,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "tool_schemas_sha256": hashlib.sha256(
                json.dumps(TOOL_SCHEMAS, sort_keys=True).encode()
            ).hexdigest(),
        },
        "unfreeze_policy": (
            "Only a demonstrated grader or infrastructure bug reproducible on a "
            "saved trajectory may create a new benchmark version. Model capability "
            "failures do not justify changing this manifest."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_freeze() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            "Benchmark is not frozen. Run: python benchmark_freeze.py --create"
        )
    manifest = json.loads(MANIFEST_PATH.read_text())
    expected = manifest.get("files", {})
    observed = current_hashes()
    changed = [
        name
        for name in FROZEN_FILES
        if expected.get(name) != observed.get(name)
    ]
    if changed:
        raise SystemExit(
            "Frozen benchmark verification failed; changed files: "
            + ", ".join(changed)
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = create_freeze() if args.create else verify_freeze()
    print(
        f"Benchmark {manifest['benchmark_version']} "
        f"{'created' if args.create else 'verified'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
