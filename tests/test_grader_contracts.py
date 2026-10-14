import json
import unittest
from pathlib import Path

import yaml

from run_evals import evaluate


ROOT = Path(__file__).resolve().parents[1]


def load_task(task_id):
    suite = yaml.safe_load((ROOT / "evals" / "tasks.yaml").read_text())
    return next(task for task in suite["tasks"] if task["id"] == task_id)


def make_result(answer, observations):
    messages = []

    for index, (device, command, output) in enumerate(observations):
        call_id = f"call_{index}"
        messages.extend([
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "run_show_command",
                        "arguments": json.dumps({
                            "device": device,
                            "command": command,
                        }),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": output,
            },
        ])

    messages.append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "messages": messages,
        "iterations": len(observations) + 1,
    }


class GraderContractTests(unittest.TestCase):
    def test_dr_contradiction_is_rejected(self):
        task = load_task("t06")
        run = make_result(
            "R2 is the DR? No, R1 is the DR.",
            [(
                "R2",
                "show ip ospf interface eth1",
                "State DR",
            )],
        )

        self.assertEqual(evaluate(task, run), (False, "wrong_answer"))

    def test_neighbor_evidence_rejects_an_extra_neighbor(self):
        task = load_task("t01")
        run = make_result(
            "R1 has 2 OSPF neighbors.",
            [(
                "R1",
                "show ip ospf neighbor",
                (
                    "Neighbor ID Pri State\n"
                    "10.50.0.2 1 Full/DR\n"
                    "3.3.3.3 1 Full/DR\n"
                    "4.4.4.4 1 Full/DR\n"
                ),
            )],
        )

        self.assertEqual(
            evaluate(task, run),
            (False, "missing_or_invalid_evidence"),
        )

    def test_exact_learned_route_set_is_accepted(self):
        task = load_task("t07")
        run = make_result(
            "3",
            [(
                "R2",
                "show ip route ospf",
                (
                    "O>* 1.1.1.1/32 via 10.0.12.1, eth1\n"
                    "O>* 3.3.3.3/32 via 10.0.23.2, eth2\n"
                    "O>* 10.0.13.0/30 via 10.0.12.1, eth1\n"
                    "O 10.0.12.0/30 is directly connected, eth1\n"
                ),
            )],
        )

        self.assertEqual(evaluate(task, run), (True, None))

    def test_route_count_evidence_rejects_an_extra_learned_route(self):
        task = load_task("t07")
        run = make_result(
            "3",
            [(
                "R2",
                "show ip route ospf",
                (
                    "O>* 1.1.1.1/32 via 10.0.12.1, eth1\n"
                    "O>* 3.3.3.3/32 via 10.0.23.2, eth2\n"
                    "O>* 10.0.13.0/30 via 10.0.12.1, eth1\n"
                    "O>* 192.0.2.0/24 via 10.0.12.1, eth1\n"
                ),
            )],
        )

        self.assertEqual(
            evaluate(task, run),
            (False, "missing_or_invalid_evidence"),
        )

    def test_backup_path_requires_evidence_for_both_legs(self):
        task = load_task("t11")
        run = make_result(
            "Yes, R1 would still reach R2 through R3.",
            [
                (
                    "R1",
                    "show ip route 3.3.3.3",
                    "3.3.3.3/32 via 10.0.13.2, eth2",
                ),
                (
                    "R3",
                    "show ip route 10.50.0.2",
                    "10.50.0.2/32 via 10.0.23.1, eth1",
                ),
            ],
        )

        self.assertEqual(evaluate(task, run), (True, None))

    def test_backup_path_rejects_only_the_current_direct_route(self):
        task = load_task("t11")
        run = make_result(
            "Yes, R1 would still reach R2 through R3.",
            [(
                "R1",
                "show ip route 10.50.0.2",
                "10.50.0.2/32 via 10.0.12.2, eth1",
            )],
        )

        self.assertEqual(
            evaluate(task, run),
            (False, "missing_or_invalid_evidence"),
        )

    def test_router_database_rejects_an_extra_router(self):
        task = load_task("t12")
        output = (
            "OSPF Router with ID (1.1.1.1)\n"
            "Router Link States (Area 0.0.0.0)\n"
            "1.1.1.1 1.1.1.1\n"
            "10.50.0.2 10.50.0.2\n"
            "3.3.3.3 3.3.3.3\n"
            "4.4.4.4 4.4.4.4\n"
        )
        run = make_result(
            "1.1.1.1, 10.50.0.2, 3.3.3.3",
            [("R1", "show ip ospf database", output)],
        )

        self.assertEqual(
            evaluate(task, run),
            (False, "missing_or_invalid_evidence"),
        )

    def test_exact_router_database_is_accepted(self):
        task = load_task("t12")
        output = (
            "OSPF Router with ID (1.1.1.1)\n"
            "Router Link States (Area 0.0.0.0)\n"
            "1.1.1.1 1.1.1.1\n"
            "10.50.0.2 10.50.0.2\n"
            "3.3.3.3 3.3.3.3\n"
        )
        run = make_result(
            "1.1.1.1, 10.50.0.2, 3.3.3.3",
            [("R1", "show ip ospf database", output)],
        )

        self.assertEqual(evaluate(task, run), (True, None))


if __name__ == "__main__":
    unittest.main()
