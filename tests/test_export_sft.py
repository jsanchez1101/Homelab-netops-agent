import hashlib
import unittest

from export_sft import select_records


def record(
    split="train",
    success=True,
    prompt="Safe inventory",
    task_id="t01",
    record_id="record-1",
    answer="correct",
):
    return {
        "id": record_id,
        "split": split,
        "success": success,
        "task_id": task_id,
        "task": "question",
        "answer": answer,
        "iterations": 2,
        "ts": 1,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_show_command", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "evidence"},
            {"role": "assistant", "content": answer},
        ],
    }


class ExportTests(unittest.TestCase):
    def test_only_verified_training_split_is_selected(self):
        selected, skipped = select_records([
            record(),
            record(split="holdout", task_id="h01", record_id="record-2"),
            record(split=None, record_id="record-3"),
            record(success=False, record_id="record-4"),
            record(
                prompt="R2: VyOS router, RID 10.50.0.2",
                record_id="record-5",
            ),
        ])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["task_id"], "t01")
        self.assertEqual(skipped["non_training_or_legacy_split"], 2)
        self.assertEqual(skipped["failed_eval"], 1)
        self.assertEqual(skipped["invalid_leaky_or_stale_prompt"], 1)

    def test_current_prompt_and_frozen_session_are_required(self):
        prompt = "current prompt"
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        selected, skipped = select_records(
            [
                record(prompt=prompt, record_id="eligible"),
                record(prompt="old prompt", record_id="old-prompt"),
                record(prompt=prompt, record_id="probe"),
            ],
            expected_prompt_sha256=prompt_hash,
            eligible_ids={"eligible", "old-prompt"},
        )
        self.assertEqual([item["id"] for item in selected], ["eligible"])
        self.assertEqual(skipped["invalid_leaky_or_stale_prompt"], 1)
        self.assertEqual(skipped["outside_frozen_baseline"], 1)

    def test_saved_success_must_still_pass_the_current_grader(self):
        task = {
            "id": "t01",
            "task": "question",
            "check": "contains",
            "expect": "correct",
        }
        selected, skipped = select_records(
            [record(answer="wrong")],
            tasks=[task],
        )
        self.assertEqual(selected, [])
        self.assertEqual(skipped["fails_current_grader"], 1)


if __name__ == "__main__":
    unittest.main()
