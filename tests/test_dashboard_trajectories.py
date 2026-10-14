import unittest
from unittest.mock import patch

from demo.main import display_attempt, legacy_baseline_records
from run_evals_multi import one_run


class DashboardTrajectoryTests(unittest.TestCase):
    def test_multi_run_aborts_instead_of_scoring_model_outage(self):
        task = {
            "id": "t01",
            "difficulty": "easy",
            "task": "question",
        }
        with (
            patch(
                "run_evals_multi.run_task",
                side_effect=ConnectionError("model offline"),
            ),
            patch("run_evals_multi.save_trajectory") as save,
        ):
            with self.assertRaisesRegex(RuntimeError, "partial run was not scored"):
                one_run([task], verbose=False, split="train")
        save.assert_not_called()

    def test_legacy_baseline_requires_exact_task_sequence_and_scores(self):
        result = {
            "tasks": [{"id": "t01"}, {"id": "t02"}],
            "n_runs": 2,
            "scores": [1, 2],
        }
        records = [
            {"task_id": "noise", "success": True},
            {"task_id": "t01", "success": True, "id": "a"},
            {"task_id": "t02", "success": False, "id": "b"},
            {"task_id": "t01", "success": True, "id": "c"},
            {"task_id": "t02", "success": True, "id": "d"},
        ]

        linked = legacy_baseline_records(result, records)

        self.assertEqual([run["run"] for run in linked], [1, 2])
        self.assertEqual(linked[0]["tasks"]["t01"]["id"], "a")
        self.assertEqual(linked[1]["tasks"]["t02"]["id"], "d")

    def test_attempt_separates_answer_and_evidence_verdicts(self):
        record = {
            "success": False,
            "failure_reason": "missing_or_invalid_evidence",
            "answer": "R2",
            "iterations": 2,
            "messages": [],
        }

        attempt = display_attempt(record, 3)

        self.assertEqual(attempt["answer_status"], "pass")
        self.assertEqual(attempt["evidence_status"], "fail")

    def test_dashboard_redacts_mac_addresses_from_tool_output(self):
        record = {
            "success": True,
            "failure_reason": None,
            "answer": "up",
            "iterations": 2,
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "run_show_command",
                            "arguments": '{"device":"R1","command":"show interfaces"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "link/ether 50:01:00:01:00:01 state UP",
                },
            ],
        }

        attempt = display_attempt(record, 1)

        self.assertIn("<mac-address>", attempt["commands"][0]["output"])
        self.assertNotIn("50:01:00:01:00:01", attempt["commands"][0]["output"])

    @patch("run_evals_multi.save_trajectory", return_value="trajectory-123")
    @patch("run_evals_multi.evaluate", return_value=(True, None))
    @patch("run_evals_multi.run_task")
    def test_multi_run_retains_trajectory_id(self, mocked_run, _mocked_eval, _mocked_save):
        mocked_run.return_value = {
            "answer": "2",
            "messages": [],
            "iterations": 1,
        }
        tasks = [{"id": "t01", "task": "How many?", "difficulty": "easy"}]

        results, failures, _elapsed, trajectory_ids = one_run(
            tasks, verbose=False, split="train"
        )

        self.assertEqual(results, {"t01": True})
        self.assertEqual(failures, [])
        self.assertEqual(trajectory_ids, {"t01": "trajectory-123"})


if __name__ == "__main__":
    unittest.main()
