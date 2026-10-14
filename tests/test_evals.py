import json
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from agent import (
    MODEL_HTTP,
    SYSTEM_PROMPT,
    local_endpoint_scope,
    run_show_command,
    run_task,
)
from run_evals import check, ensure_neutral_inventory, evaluate, is_abstention


ROOT = Path(__file__).resolve().parents[1]


def load_task(filename, task_id):
    suite = yaml.safe_load((ROOT / "evals" / filename).read_text())
    return next(task for task in suite["tasks"] if task["id"] == task_id)


def result(answer, device, command, output):
    call_id = "call_test"
    return {
        "answer": answer,
        "iterations": 2,
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "run_show_command",
                        "arguments": json.dumps({"device": device, "command": command}),
                    },
                }],
            },
            {"role": "tool", "tool_call_id": call_id, "content": output},
            {"role": "assistant", "content": answer},
        ],
    }


class EvaluationTests(unittest.TestCase):
    def test_local_model_endpoints_are_allowed(self):
        self.assertEqual(
            local_endpoint_scope("http://127.0.0.1:11434/v1"),
            "loopback",
        )
        self.assertEqual(
            local_endpoint_scope("http://100.64.0.10:11434/v1"),
            "tailscale",
        )
        self.assertEqual(
            local_endpoint_scope("http://10.0.0.5:11434/v1"),
            "private",
        )

    def test_public_or_cloud_model_endpoints_are_rejected(self):
        rejected = [
            "https://api.openai.com/v1",
            "https://artifactory.example.com/v1",
            "http://8.8.8.8:11434/v1",
        ]

        for endpoint in rejected:
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ValueError):
                    local_endpoint_scope(endpoint)

    def test_model_http_session_ignores_proxy_environment(self):
        self.assertFalse(MODEL_HTTP.trust_env)

    def test_system_prompt_allows_evidence_based_abstention(self):
        self.assertIn("Giving up is acceptable", SYSTEM_PROMPT)
        self.assertIn("Never invent missing device state", SYSTEM_PROMPT)

    def test_abstention_is_detected_from_the_lead_answer(self):
        self.assertTrue(
            is_abstention(
                "I cannot determine that from the available evidence. "
                "R2 did not respond."
            )
        )
        self.assertTrue(is_abstention("There is not enough evidence to answer."))
        self.assertFalse(is_abstention("R2 is the OSPF designated router."))

    def test_answerable_task_abstention_is_still_a_failure(self):
        task = load_task("tasks.yaml", "t03")
        run = result(
            "I cannot determine that from the available evidence.",
            "R1",
            "show interfaces ethernet eth1",
            "ERROR: connection failed",
        )
        self.assertEqual(evaluate(task, run), (False, "abstained"))

    def test_empty_model_response_is_retried_once(self):
        responses = iter([
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "OSPF is running."},
        ])
        with patch("agent.chat", side_effect=lambda messages: next(responses)) as mocked_chat:
            run = run_task("What protocol is running?", verbose=False)

        self.assertEqual(run["answer"], "OSPF is running.")
        self.assertEqual(run["iterations"], 2)
        self.assertEqual(run["empty_response_retries"], 1)
        self.assertEqual(mocked_chat.call_count, 2)

    def test_second_empty_model_response_remains_a_failure(self):
        with patch(
            "agent.chat",
            side_effect=[
                {"role": "assistant", "content": ""},
                {"role": "assistant", "content": ""},
            ],
        ) as mocked_chat:
            run = run_task("What protocol is running?", verbose=False)

        self.assertEqual(run["answer"], "")
        self.assertEqual(run["empty_response_retries"], 1)
        self.assertEqual(mocked_chat.call_count, 2)

    def test_integer_answer_ignores_router_device_label(self):
        self.assertTrue(check("R2 learned 3 routes.", "integer_answer", 3))
        self.assertFalse(check("R2 learned 4 routes.", "integer_answer", 3))
        self.assertFalse(check("R2 learned 3 routes with 2 backups.", "integer_answer", 3))

    def test_evaluation_refuses_smoke_inventory(self):
        with self.assertRaises(SystemExit):
            ensure_neutral_inventory(ROOT / "devices.smoke.yaml")

    def test_evaluation_accepts_neutral_inventory(self):
        ensure_neutral_inventory(ROOT / "devices.yaml")

    def test_wrong_dr_answer_is_not_rescued_by_mentioning_r2(self):
        task = load_task("tasks.yaml", "t06")
        run = result(
            "The OSPF DR is R1; R2 is the backup.",
            "R1",
            "show ip ospf interface eth1",
            "Designated Router (ID) 10.50.0.2 Interface Address 10.0.12.2/30",
        )
        self.assertEqual(evaluate(task, run), (False, "wrong_answer"))

    def test_correct_dr_answer_with_direct_evidence_passes(self):
        task = load_task("tasks.yaml", "t06")
        run = result(
            "R2 is the OSPF designated router on the R1-R2 link.",
            "R1",
            "show ip ospf interface eth1",
            "Designated Router (ID) 10.50.0.2 Interface Address 10.0.12.2/30",
        )
        self.assertEqual(evaluate(task, run), (True, None))

    def test_guessed_interface_state_after_invalid_command_fails(self):
        task = load_task("tasks.yaml", "t03")
        run = result(
            "Interface eth1 on R1 is up.",
            "R1",
            "show interfaces eth1",
            "ERROR: invalid command",
        )
        self.assertEqual(evaluate(task, run), (False, "missing_or_invalid_evidence"))

    def test_interface_state_with_vyos_command_passes(self):
        task = load_task("tasks.yaml", "t03")
        run = result(
            "Interface eth1 on R1 is up.",
            "R1",
            "show interfaces ethernet eth1",
            "eth1: <BROADCAST,UP,LOWER_UP> mtu 1500 state UP group default",
        )
        self.assertEqual(evaluate(task, run), (True, None))

    def test_holdout_router_id_requires_live_ospf_evidence(self):
        task = load_task("holdout.yaml", "h01")
        unsupported = result(
            "R2's OSPF router ID is 10.50.0.2.", "R2", "show version", "VyOS 2026"
        )
        grounded = result(
            "R2's OSPF router ID is 10.50.0.2.",
            "R2",
            "show ip ospf",
            "OSPF Routing Process, Router ID: 10.50.0.2",
        )
        self.assertEqual(evaluate(task, unsupported)[1], "missing_or_invalid_evidence")
        self.assertEqual(evaluate(task, grounded), (True, None))

    def test_ospf_interface_output_is_valid_router_id_evidence(self):
        task = load_task("tasks.yaml", "t04")
        run = result(
            "R3's OSPF router ID is 3.3.3.3.",
            "R3",
            "show ip ospf interface eth1",
            "eth1 is up\n  Router ID 3.3.3.3, Network Type BROADCAST, Cost: 1",
        )
        self.assertEqual(evaluate(task, run), (True, None))

    def test_neighbor_count_requires_both_neighbor_rows(self):
        task = load_task("tasks.yaml", "t01")
        one_neighbor = result(
            "R1 has 2 OSPF neighbors.",
            "R1",
            "show ip ospf neighbor",
            "10.50.0.2 1 Full/DR",
        )
        two_neighbors = result(
            "R1 has 2 OSPF neighbors.",
            "R1",
            "show ip ospf neighbor",
            "10.50.0.2 1 Full/DR\n3.3.3.3 1 Full/DR",
        )
        self.assertEqual(evaluate(task, one_neighbor)[1], "missing_or_invalid_evidence")
        self.assertEqual(evaluate(task, two_neighbors), (True, None))

    def test_specific_route_output_supports_next_hop(self):
        task = load_task("tasks.yaml", "t09")
        run = result(
            "Yes, the next-hop IP is 10.0.12.2.",
            "R1",
            "show ip route 10.50.0.2",
            "Routing entry for 10.50.0.2/32\n  * 10.0.12.2, via eth1, weight 1",
        )
        self.assertEqual(evaluate(task, run), (True, None))

    def test_reachability_accepts_a_natural_affirmative_answer(self):
        task = load_task("tasks.yaml", "t09")
        run = result(
            "R1 can reach R2's loopback through next-hop 10.0.12.2.",
            "R1",
            "show ip route 10.50.0.2",
            "Routing entry for 10.50.0.2/32\n  * 10.0.12.2, via eth1",
        )
        self.assertEqual(evaluate(task, run), (True, None))

    def test_reachability_rejects_an_explicit_no(self):
        task = load_task("tasks.yaml", "t09")
        run = result(
            "No, R1 cannot reach it. The next-hop shown is 10.0.12.2.",
            "R1",
            "show ip route 10.50.0.2",
            "Routing entry for 10.50.0.2/32\n  * 10.0.12.2, via eth1",
        )
        self.assertEqual(evaluate(task, run), (False, "wrong_answer"))

    def test_router_id_list_accepts_bullets_with_three_direct_observations(self):
        task = load_task("tasks.yaml", "t12")
        calls = [
            ("R1", "show ip ospf interface eth1", "Router ID 1.1.1.1"),
            ("R2", "show ip ospf interface eth1", "Router ID 10.50.0.2"),
            ("R3", "show ip ospf interface eth1", "Router ID 3.3.3.3"),
        ]
        messages = []
        for index, (device, command, output) in enumerate(calls):
            call_id = f"call_{index}"
            messages.extend([
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "run_show_command",
                            "arguments": json.dumps({"device": device, "command": command}),
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": call_id, "content": output},
            ])
        answer = "The router IDs are:\n- 1.1.1.1\n- 10.50.0.2\n- 3.3.3.3"
        messages.append({"role": "assistant", "content": answer})
        run = {"answer": answer, "messages": messages, "iterations": 4}
        self.assertEqual(evaluate(task, run), (True, None))

    def test_router_id_set_rejects_extra_addresses(self):
        expected = ["1.1.1.1", "10.50.0.2", "3.3.3.3"]
        self.assertTrue(check("1.1.1.1, 10.50.0.2, 3.3.3.3", "lead_ip_set", expected))
        self.assertFalse(
            check("1.1.1.1, 10.50.0.2, 3.3.3.3 via 10.0.12.2", "lead_ip_set", expected)
        )

    def test_shell_syntax_is_rejected_before_ssh(self):
        dangerous_commands = [
            "show version; echo escaped",
            "show version && echo escaped",
            "show version || echo escaped",
            "show version | echo escaped",
            "show version & echo escaped",
            "show version $(echo escaped)",
            "show version `echo escaped`",
            "show version > /tmp/output",
            "show version < /etc/passwd",
            "show ip ospf neighbor | match R1",
        ]

        with patch("netmiko.ConnectHandler") as connect:
            connection = connect.return_value
            connection.send_command.return_value = "unexpected execution"

            for command in dangerous_commands:
                with self.subTest(command=command):
                    rejection = run_show_command("R1", command)
                    self.assertIn("rejected", rejection.lower())

            # Compare only the count so a failure cannot print inventory secrets.
            self.assertEqual(connect.call_count, 0)

    def test_approved_commands_reach_ssh(self):
        approved_commands = [
            "show version",
            "show interfaces",
            "show interfaces ethernet eth1",
            "show ip ospf",
            "show ip ospf neighbor",
            "show ip ospf neighbor detail",
            "show ip ospf interface eth1",
            "show ip ospf database",
            "show ip ospf database router",
            "show ip route ospf",
            "show ip route 10.50.0.2",
            "show ip route 10.50.0.2/32",
        ]

        with patch("netmiko.ConnectHandler") as connect:
            connection = connect.return_value
            connection.send_command.return_value = "safe output"

            for command in approved_commands:
                with self.subTest(command=command):
                    self.assertEqual(run_show_command("R1", command), "safe output")

            self.assertEqual(connect.call_count, len(approved_commands))

    def test_multiline_command_is_rejected_before_ssh(self):
        rejection = run_show_command("R1", "show version\nconfigure")
        self.assertIn("exactly one CLI line", rejection)


if __name__ == "__main__":
    unittest.main()
