"""
Filter successful trajectories into SFT-ready chat format (ShareGPT-style
JSONL that Unsloth/axolotl ingest directly).

Usage:
    python export_sft.py            # writes sft_data.jsonl
"""

import json
from pathlib import Path

SRC = Path(__file__).with_name("trajectories") / "trajectories.jsonl"
OUT = Path(__file__).with_name("sft_data.jsonl")


def main():
    if not SRC.exists():
        print("No trajectories yet. Run the agent / evals first.")
        return
    kept, skipped = 0, 0
    with OUT.open("w") as out:
        for line in SRC.read_text().splitlines():
            rec = json.loads(line)
            if not rec.get("success"):
                skipped += 1
                continue
            # Keep messages as-is: system, user, assistant(tool_calls), tool, assistant(final).
            # Unsloth's chat template handling for qwen2.5 accepts this structure.
            out.write(json.dumps({"messages": rec["messages"]}) + "\n")
            kept += 1
    print(f"Wrote {kept} successful trajectories to {OUT.name} (skipped {skipped} failures).")
    print("Next: upload sft_data.jsonl to Colab and run the Unsloth LoRA notebook.")


if __name__ == "__main__":
    main()
