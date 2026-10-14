import tempfile
import unittest
from pathlib import Path

from failure_reports import failure_entry, write_failure_report


class FailureReportTests(unittest.TestCase):
    def test_report_contains_command_output_answer_and_reason(self):
        task = {"id": "h01", "difficulty": "easy", "task": "What is the ID?"}
        result = {
            "answer": "The ID is wrong.",
            "iterations": 2,
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "run_show_command",
                            "arguments": '{"device":"R2","command":"show ip ospf neighbor"}',
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "1.1.1.1 Full"},
            ],
        }
        entry = failure_entry(task, result, "wrong_answer", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = write_failure_report(
                "holdout.yaml", [entry], scores=[2], output=Path(directory) / "report.txt"
            )
            report = path.read_text()
        self.assertIn("show ip ospf neighbor", report)
        self.assertIn("1.1.1.1 Full", report)
        self.assertIn("The ID is wrong.", report)
        self.assertIn("wrong_answer", report)


if __name__ == "__main__":
    unittest.main()
