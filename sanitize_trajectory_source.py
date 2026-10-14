#!/usr/bin/env python3
"""Create a least-data, locally sanitized trajectory JSONL for transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

TAILSCALE_CGNAT = re.compile(r"100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.(?:\d{1,3})\.(?:\d{1,3})")
MAC_ADDRESS = re.compile(r"(?i)\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b")
ALLOWED_RECORD_FIELDS = ("id", "ts", "model", "task", "success", "iterations")
OPTIONAL_PROVENANCE_FIELDS = ("task_id", "split", "failure_reason")
ALLOWED_MESSAGE_ROLES = {"assistant", "tool"}


def redact(value: Any) -> Any:
    if isinstance(value, str):
        value = TAILSCALE_CGNAT.sub("[REDACTED_TAILSCALE_IP]", value)
        return MAC_ADDRESS.sub("[REDACTED_MAC_ADDRESS]", value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    return value


def project(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    missing = [field for field in ALLOWED_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"line {line_number}: missing {', '.join(missing)}")
    if not isinstance(record["success"], bool):
        raise ValueError(f"line {line_number}: success must be a frozen boolean verdict")
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"line {line_number}: messages must be a list")
    kept_messages = [message for message in messages if message.get("role") in ALLOWED_MESSAGE_ROLES]
    projected = {field: record[field] for field in ALLOWED_RECORD_FIELDS}
    projected.update(
        {field: record[field] for field in OPTIONAL_PROVENANCE_FIELDS if field in record}
    )
    projected["messages"] = kept_messages
    return redact(projected)


def sanitize(
    source: Path,
    destination: Path,
    model: str | None = None,
    expected_records: int | None = None,
    expected_prompts: int | None = None,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    scanned = 0
    models: dict[str, int] = {}
    task_hashes: set[str] = set()
    with source.open("r", encoding="utf-8") as input_file, tempfile.NamedTemporaryFile(
        "wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as output_file:
        temporary = Path(output_file.name)
        try:
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"line {line_number}: invalid JSON: {error}") from error
                scanned += 1
                if model is not None and record.get("model") != model:
                    continue
                clean = project(record, line_number)
                encoded = (json.dumps(clean, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
                if TAILSCALE_CGNAT.search(encoded.decode()):
                    raise AssertionError(f"line {line_number}: CGNAT address survived redaction")
                if MAC_ADDRESS.search(encoded.decode()):
                    raise AssertionError(f"line {line_number}: MAC address survived redaction")
                output_file.write(encoded)
                digest.update(encoded)
                count += 1
                models[clean["model"]] = models.get(clean["model"], 0) + 1
                task_hashes.add(hashlib.sha256(" ".join(clean["task"].split()).encode()).hexdigest())
            output_file.flush()
            os.fsync(output_file.fileno())
            if count == 0:
                raise ValueError("source contained no trajectory records")
            if expected_records is not None and count != expected_records:
                raise ValueError(f"selected {count} records; expected {expected_records}")
            if expected_prompts is not None and len(task_hashes) != expected_prompts:
                raise ValueError(f"selected {len(task_hashes)} unique prompts; expected {expected_prompts}")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return {
        "source": str(source),
        "output": str(destination),
        "records": count,
        "records_scanned": scanned,
        "records_skipped": scanned - count,
        "model_filter": model,
        "unique_prompt_hashes": len(task_hashes),
        "models": models,
        "sha256": digest.hexdigest(),
        "tailscale_cgnat_matches": 0,
        "mac_address_matches": 0,
        "system_and_user_messages_removed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", help="keep only records from this exact model name")
    parser.add_argument("--expect-records", type=int)
    parser.add_argument("--expect-prompts", type=int)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("destination must not overwrite the raw source")
    manifest = sanitize(
        args.source,
        args.destination,
        model=args.model,
        expected_records=args.expect_records,
        expected_prompts=args.expect_prompts,
    )
    manifest_path = args.manifest or args.destination.with_suffix(args.destination.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
