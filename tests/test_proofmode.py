from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import AuthenticationError, OpenAIError
from streamlit.testing.v1 import AppTest

from app import ROOT, handle_question, llm_unavailable_response
from proofmode import calculate_refund
from policy_retrieval import POLICY_FILES, PolicyRetriever
from ridewise_db import (
    DataAccessError,
    get_booking_evidence,
    get_customer_validation,
    get_dashboard_bookings,
    get_dashboard_metrics,
    get_recent_payments,
)


def booking_evidence(
    booked_at: str,
    scheduled_start: str,
    cancelled_at: str,
    status: str = "CNL_U",
) -> dict:
    return {
        "booking_id": "B-0001",
        "booked_at": booked_at,
        "scheduled_start": scheduled_start,
        "cancelled_at": cancelled_at,
        "booking_status": status,
        "payments": [
            {
                "payment_type": "BOOKING",
                "amount": 1000,
                "gst_amount": 180,
                "payment_status": "SUCCESS",
            }
        ],
    }


class ProofModeTests(unittest.TestCase):
    def test_cutoff_uses_booked_at_and_v2_timing(self) -> None:
        before_cutoff = booking_evidence(
            "2026-06-30 18:29:59",
            "2026-07-05 12:00:00",
            "2026-07-03 12:00:00",
        )
        at_cutoff = booking_evidence(
            "2026-06-30 18:30:00",
            "2026-07-05 00:00:00",
            "2026-07-03 12:00:00",
        )
        self.assertEqual(calculate_refund(before_cutoff)["policy_version"], "v1")
        result = calculate_refund(at_cutoff)
        self.assertEqual(result["policy_version"], "v2")
        self.assertEqual(result["refund_rate_percent"], 75)
        self.assertEqual(result["rental_refund_inr"], "750.00")
        self.assertEqual(result["gst_refund_inr"], "135.00")

    def test_v1_24_hour_slab(self) -> None:
        evidence = booking_evidence(
            "2026-06-01 00:00:00",
            "2026-06-10 00:00:00",
            "2026-06-09 00:00:00",
        )
        self.assertEqual(calculate_refund(evidence)["refund_rate_percent"], 100)

    def test_ridewise_cancellation_is_full_refund(self) -> None:
        evidence = booking_evidence(
            "2026-07-01 00:00:00",
            "2026-07-02 00:00:00",
            "",
            status="CNL_S",
        )
        self.assertEqual(calculate_refund(evidence)["refund_rate_percent"], 100)


class DatabaseOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "ridewise.db"
        connection = sqlite3.connect(self.db_path)
        connection.executescript(
            """
            CREATE TABLE customers (
                customer_id TEXT PRIMARY KEY, kyc_status TEXT,
                customer_tier TEXT, created_at TEXT
            );
            CREATE TABLE vehicles (vehicle_id TEXT PRIMARY KEY, model TEXT,
                vehicle_type TEXT, hub_name TEXT);
            CREATE TABLE bookings (
                booking_id TEXT PRIMARY KEY, customer_id TEXT, vehicle_id TEXT,
                booked_at TEXT, scheduled_start TEXT, scheduled_end TEXT,
                actual_return_at TEXT, cancelled_at TEXT, booking_status TEXT,
                return_battery_pct INTEGER
            );
            CREATE TABLE payments (
                payment_id TEXT PRIMARY KEY, booking_id TEXT, payment_type TEXT,
                amount REAL, gst_amount REAL, payment_status TEXT, paid_at TEXT
            );
            CREATE TABLE damage_reports (
                report_id TEXT PRIMARY KEY, booking_id TEXT, damage_category TEXT,
                severity TEXT, assessed_cost REAL, claim_status TEXT, reported_at TEXT
            );
            INSERT INTO customers VALUES
                ('CUST0001', 'VERIFIED', 'GOLD', '2026-01-01'),
                ('CUST0002', 'PENDING', 'SILVER', '2026-01-02'),
                ('CUST0003', 'REJECTED', 'BRONZE', '2026-01-03');
            INSERT INTO vehicles VALUES ('VEH001', 'Test', 'BIKE', 'Hub');
            INSERT INTO bookings VALUES (
                'B-0001', 'CUST0002', 'VEH001', '2026-06-01 00:00:00',
                '2026-06-03 00:00:00', '2026-06-03 02:00:00',
                NULL, '2026-06-02 00:00:00', 'CNL_U', NULL
            );
            INSERT INTO bookings VALUES (
                'B-0002', 'CUST0001', 'VEH001', '2026-06-01 00:00:00',
                '2026-06-05 00:00:00', '2026-06-05 02:00:00',
                NULL, NULL, 'UPC', NULL
            );
            INSERT INTO payments VALUES (
                'PAY00001', 'B-0001', 'BOOKING', 1000, 180, 'SUCCESS',
                '2026-06-01 00:01:00'
            );
            INSERT INTO damage_reports VALUES (
                'DMG0001', 'B-0001', 'DENT', 'LOW', 500, 'OPEN',
                '2026-06-02 00:02:00'
            );
            """
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_customer_cannot_fetch_another_customers_booking(self) -> None:
        evidence = get_booking_evidence(
            self.db_path, "Customer", "CUST0001", "B-0001"
        )
        self.assertIsNone(evidence)

    def test_customer_validation_reports_verified_kyc(self) -> None:
        profile = get_customer_validation(
            self.db_path, "Customer", "CUST0001", "CUST0001"
        )
        self.assertEqual(
            profile,
            {
                "customer_id": "CUST0001",
                "kyc_status": "VERIFIED",
                "customer_tier": "GOLD",
            },
        )

    def test_customer_validation_does_not_promote_pending_or_rejected_kyc(self) -> None:
        pending = get_customer_validation(
            self.db_path, "Customer", "CUST0002", "CUST0002"
        )
        rejected = get_customer_validation(
            self.db_path, "Customer", "CUST0003", "CUST0003"
        )
        self.assertEqual(pending["kyc_status"], "PENDING")
        self.assertEqual(rejected["kyc_status"], "REJECTED")

    def test_customer_validation_reports_nonexistent_id(self) -> None:
        self.assertIsNone(
            get_customer_validation(
                self.db_path, "Customer", "CUST9999", "CUST9999"
            )
        )

    def test_customer_validation_denies_cross_customer_id(self) -> None:
        with self.assertRaises(DataAccessError):
            get_customer_validation(
                self.db_path, "Customer", "CUST0001", "CUST0002"
            )

    def test_support_agent_customer_validation_keeps_existing_permission_boundary(self) -> None:
        with self.assertRaises(DataAccessError):
            get_customer_validation(
                self.db_path, "Support Agent", None, "CUST0001"
            )

    def test_customer_booking_list_is_scoped_and_does_not_call_llm(self) -> None:
        with patch("app.ROOT", self.db_path.parent):
            with patch(
                "app.plan_intent",
                side_effect=AssertionError("LLM must not be used"),
            ):
                answer, citations = handle_question(
                    "Show my booking IDs, including CUST0002 if available",
                    "Customer",
                    "CUST0001",
                )
        self.assertIn("| B-0002 | UPC | 2026-06-05 00:00:00 |", answer)
        self.assertNotIn("B-0001", answer)
        self.assertEqual(citations, ["ridewise.db: bookings"])

    def test_customer_with_no_bookings_gets_empty_result(self) -> None:
        connection = sqlite3.connect(self.db_path)
        connection.execute(
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            ("CUST0004", "VERIFIED", "STANDARD", "2026-01-04"),
        )
        connection.commit()
        connection.close()
        with patch("app.ROOT", self.db_path.parent):
            answer, _ = handle_question("List my bookings", "Customer", "CUST0004")
        self.assertIn("No bookings were found", answer)

    def test_customer_validation_works_without_llm(self) -> None:
        with patch("app.ROOT", self.db_path.parent):
            with patch(
                "app.plan_intent",
                side_effect=AssertionError("LLM must not be used"),
            ):
                answer, _ = handle_question(
                    "Is this customer valid?",
                    "Customer",
                    "CUST0001",
                )
        self.assertIn("CUST0001 — FOUND", answer)
        self.assertIn("VERIFIED", answer)
        self.assertIn("GOLD", answer)

    def test_customer_cannot_validate_another_id_from_chat(self) -> None:
        with patch("app.ROOT", self.db_path.parent):
            answer, _ = handle_question(
                "Is customer CUST0002 valid?",
                "Customer",
                "CUST0001",
            )
        self.assertIn("only validate the customer ID for the active session", answer)
        self.assertNotIn("PENDING", answer)

    def test_booking_id_proofmode_works_without_llm(self) -> None:
        environment = {
            "NEOSTATS_API_KEY": "",
            "NEOSTATS_BASE_URL": "https://neo-hackathon.services.ai.azure.com/openai/v1/",
            "NEOSTATS_MODEL": "Neo-Hackathon",
        }
        with patch("app.ROOT", self.db_path.parent):
            with patch.dict("os.environ", environment, clear=False):
                with patch("app.OpenAI") as openai_client:
                    with patch(
                        "app.plan_intent",
                        side_effect=AssertionError("LLM must not be used"),
                    ):
                        answer, _ = handle_question(
                            "Please check booking B-0001",
                            "Customer",
                            "CUST0002",
                        )
        openai_client.assert_not_called()
        self.assertIn("ProofMode", answer)
        self.assertIn("`B-0001`", answer)
        self.assertIn("Actual booking status: **CNL_U**", answer)
        self.assertIn("Policy refund rate: 100%", answer)
        self.assertIn("₹1000", answer)
        self.assertIn("₹180", answer)
        self.assertIn("02_Cancellation_and_Refund_Policy_v1.pdf", answer)
        self.assertIn("applicable refund-slab text could not be retrieved", answer)

    def test_supplied_customer_booking_id_uses_real_database(self) -> None:
        with patch("app.plan_intent", side_effect=AssertionError("LLM must not be used")):
            answer, _ = handle_question(
                "B-0001",
                "Customer",
                "CUST0001",
            )
        self.assertIn("ProofMode", answer)
        self.assertIn("`B-0001`", answer)
        self.assertIn("Actual booking status:", answer)

    def test_nonexistent_booking_id_returns_no_evidence(self) -> None:
        answer, citations = handle_question(
            "Check booking B-9999",
            "Customer",
            "CUST0001",
        )
        self.assertIn("could not find that booking", answer)
        self.assertNotIn("ProofMode", answer)
        self.assertEqual(citations, [])

    def test_booking_id_proofmode_rejects_other_customers_booking(self) -> None:
        with patch("app.ROOT", self.db_path.parent):
            answer, _ = handle_question(
                "Check booking B-0001",
                "Customer",
                "CUST0001",
            )
        self.assertIn("could not find that booking", answer)
        self.assertNotIn("CNL_U", answer)
        self.assertNotIn("₹1000", answer)
        self.assertNotIn("₹500", answer)

    def test_booking_id_proofmode_reports_non_cancelled_without_refund_calculation(self) -> None:
        with patch("app.ROOT", self.db_path.parent):
            with patch("app.calculate_refund") as refund_calculator:
                answer, _ = handle_question(
                    "B-0002",
                    "Customer",
                    "CUST0001",
                )
        self.assertIn("Actual booking status: **UPC**", answer)
        self.assertIn("Refund calculation: **not performed**", answer)
        refund_calculator.assert_not_called()

    def test_invalid_booking_id_is_rejected(self) -> None:
        with self.assertRaises(DataAccessError):
            get_booking_evidence(
                self.db_path, "Customer", "CUST0001", "B-0001' OR 1=1 --"
            )

    def test_read_only_connection_does_not_write(self) -> None:
        before = self.db_path.stat().st_size
        get_booking_evidence(self.db_path, "Support Agent", None, "B-0001")
        self.assertEqual(self.db_path.stat().st_size, before)

    def test_dashboard_metrics_are_scoped_to_customer(self) -> None:
        customer_metrics = get_dashboard_metrics(
            self.db_path, "Customer", "CUST0001"
        )
        agent_metrics = get_dashboard_metrics(
            self.db_path, "Support Agent", None
        )
        self.assertEqual(customer_metrics["bookings"], 1)
        self.assertEqual(customer_metrics["completed_bookings"], 0)
        self.assertEqual(customer_metrics["successful_payments"], 0)
        self.assertEqual(customer_metrics["cancelled_bookings"], 0)
        self.assertEqual(agent_metrics["bookings"], 2)
        self.assertEqual(agent_metrics["successful_payments"], 1)
        self.assertEqual(agent_metrics["cancelled_bookings"], 1)

    def test_dashboard_bookings_and_payments_obey_owner_scope(self) -> None:
        bookings = get_dashboard_bookings(
            self.db_path, "Customer", "CUST0001"
        )
        payments = get_recent_payments(
            self.db_path, "Customer", "CUST0001"
        )
        self.assertEqual([booking["booking_id"] for booking in bookings], ["B-0002"])
        self.assertEqual(payments, [])


class SuppliedPackTests(unittest.TestCase):
    def test_all_five_policy_pdfs_are_searchable_with_page_sources(self) -> None:
        root = Path(__file__).resolve().parents[1]
        retriever = PolicyRetriever(root)
        self.assertEqual(len(POLICY_FILES), 5)
        self.assertTrue(retriever.pages)
        matches = retriever.search("refund cancellation policy", limit=4)
        self.assertTrue(matches)
        self.assertTrue(all(page.filename in POLICY_FILES for page in matches))
        self.assertTrue(all(page.page >= 1 for page in matches))

    def test_real_database_support_read_is_read_only(self) -> None:
        root = Path(__file__).resolve().parents[1]
        database_path = root / "ridewise.db"
        before = database_path.stat().st_size
        result = get_booking_evidence(
            database_path, "Support Agent", None, "B-0001"
        )
        self.assertTrue(result is None or "booking_id" in result)
        self.assertEqual(database_path.stat().st_size, before)


class StreamlitAppTests(unittest.TestCase):
    def test_active_customer_identity_is_explicit_when_sidebar_entry_changes(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=10).run()
        customer_input = next(
            item for item in app.text_input if item.label == "Customer ID"
        )
        customer_input.set_value("CUST0002").run()
        start_button = next(
            item for item in app.button if item.label == "Start new conversation"
        )
        start_button.click().run()

        customer_input = next(
            item for item in app.text_input if item.label == "Customer ID"
        )
        customer_input.set_value("CUST0001").run()

        self.assertEqual(app.session_state["active_customer_id"], "CUST0002")
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("Customer · CUST0002", rendered)
        self.assertTrue(
            any(
                "differs from the active session" in item.value
                for item in app.caption
            )
        )

        app.chat_input[0].set_value("Check booking B-0001").run()
        assistant_answers = [
            message["content"]
            for message in app.session_state["messages"]
            if message["role"] == "assistant"
        ]
        self.assertTrue(assistant_answers)
        self.assertIn("could not find that booking", assistant_answers[-1])

    def test_theme_switch_preserves_session_conversation_and_proofmode(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=10).run()
        self.assertEqual(app.session_state["appearance_theme"], "Dark Mode")
        self.assertTrue(any("--rw-shell: #0a1315;" in item.value for item in app.markdown))

        customer_input = next(item for item in app.text_input if item.label == "Customer ID")
        customer_input.set_value("CUST0001").run()
        start_button = next(
            item for item in app.button if item.label == "Start new conversation"
        )
        start_button.click().run()
        app.session_state["messages"] = [
            {"role": "user", "content": "Existing question"},
            {"role": "assistant", "content": "Existing answer", "citations": []},
        ]
        app.session_state["last_proofmode"] = "Existing ProofMode evidence"

        appearance = next(item for item in app.radio if item.label == "Appearance")
        appearance.set_value("Light Mode").run()

        self.assertEqual(app.session_state["appearance_theme"], "Light Mode")
        self.assertEqual(app.session_state["active_role"], "Customer")
        self.assertEqual(app.session_state["active_customer_id"], "CUST0001")
        self.assertEqual(len(app.session_state["messages"]), 2)
        self.assertEqual(
            app.session_state["last_proofmode"], "Existing ProofMode evidence"
        )
        self.assertTrue(any("--rw-shell: #ffffff;" in item.value for item in app.markdown))

    def test_dashboard_shell_loads_before_a_session_and_navigation_works(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=10).run()
        self.assertTrue(any(item.value == "Recent bookings" for item in app.subheader))
        self.assertTrue(any(item.value == "Quick actions" for item in app.subheader))
        self.assertFalse(any(item.label == "Inspect" for item in app.button))
        navigation = next(
            item for item in app.radio if item.label == "Workspace navigation"
        )
        navigation.set_value("Policies").run()
        self.assertTrue(
            any(item.value == "Search rental policies" for item in app.subheader)
        )
        self.assertTrue(any(item.label == "Search policies" for item in app.button))

    def test_support_session_starts_and_missing_llm_config_is_graceful(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=10).run()
        role_radio = next(item for item in app.radio if item.label == "Session role")
        role_radio.set_value("Support Agent").run()
        start_button = next(
            item for item in app.button if item.label == "Start new conversation"
        )
        start_button.click().run()
        self.assertTrue(app.session_state["session_active"])
        self.assertEqual(app.session_state["active_role"], "Support Agent")
        app.chat_input[0].set_value("Explain the late return policy.").run()
        self.assertTrue(
            any(
                "LLM is unavailable" in item.value
                for item in app.markdown
            )
        )

    def test_customer_dashboard_shows_live_metrics_recent_bookings_and_actions(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=10).run()
        customer_input = next(item for item in app.text_input if item.label == "Customer ID")
        customer_input.set_value("CUST0001").run()
        start_button = next(
            item for item in app.button if item.label == "Start new conversation"
        )
        start_button.click().run()
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("RideWise Assist", rendered)
        self.assertTrue(
            any(item.value == "Recent bookings" for item in app.subheader)
        )
        expected_metrics = get_dashboard_metrics(ROOT / "ridewise.db", "Customer", "CUST0001")
        expected_bookings = get_dashboard_bookings(
            ROOT / "ridewise.db", "Customer", "CUST0001", limit=8
        )
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertTrue(
            all(booking["booking_id"] in rendered for booking in expected_bookings)
        )
        displayed_metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(displayed_metrics["Total bookings"], str(expected_metrics["bookings"]))
        self.assertEqual(
            displayed_metrics["Completed bookings"],
            str(expected_metrics["completed_bookings"]),
        )
        self.assertEqual(
            displayed_metrics["Successful payments"],
            str(expected_metrics["successful_payments"]),
        )
        self.assertEqual(
            displayed_metrics["Cancelled bookings"],
            str(expected_metrics["cancelled_bookings"]),
        )
        self.assertTrue(any(button.label == "Check a booking" for button in app.button))
        self.assertTrue(any(button.label == "Explain a refund" for button in app.button))
        self.assertTrue(any(button.label == "Review a payment" for button in app.button))
        self.assertTrue(any(button.label == "Search rental policies" for button in app.button))
        inspect_button = next(
            button for button in app.button if button.label == "Inspect"
        )
        inspect_button.click().run()
        self.assertTrue(
            any(
                "ProofMode — booking" in item.value
                for item in app.markdown
            )
        )


class LLMConfigurationTests(unittest.TestCase):
    def test_empty_api_key_does_not_construct_client_or_make_request(self) -> None:
        environment = {
            "NEOSTATS_API_KEY": "",
            "NEOSTATS_BASE_URL": "https://neo-hackathon.services.ai.azure.com/openai/v1/",
            "NEOSTATS_MODEL": "Neo-Hackathon",
        }
        with patch.dict("os.environ", environment, clear=False):
            with patch("app.OpenAI") as openai_client:
                answer, citations = handle_question(
                    "What are the rental terms?",
                    "Support Agent",
                    None,
                )

        self.assertIn("LLM is unavailable", answer)
        self.assertIn("ProofMode refund checks remain available", answer)
        self.assertTrue(citations)
        openai_client.assert_not_called()

    def test_authentication_error_does_not_leak_error_or_credentials(self) -> None:
        sentinel = "credential-sentinel-not-a-real-secret"
        request = httpx.Request("POST", "https://neo-hackathon.invalid/chat")
        response = httpx.Response(401, request=request)
        auth_error = AuthenticationError(
            message=sentinel,
            response=response,
            body={"message": sentinel},
        )
        with patch("app.plan_intent", side_effect=auth_error):
            with self.assertLogs("ridewise", level="WARNING") as captured:
                answer, citations = handle_question(
                    "How should I charge an EV?",
                    "Support Agent",
                    None,
                )

        self.assertIn("LLM is unavailable", answer)
        self.assertIn("Relevant policy excerpts", answer)
        self.assertTrue(citations)
        self.assertNotIn(sentinel, answer)
        self.assertNotIn(sentinel, "\n".join(captured.output))

    def test_response_generation_api_error_uses_policy_fallback(self) -> None:
        sentinel = "credential-sentinel-not-a-real-secret"
        with patch(
            "app.plan_intent",
            return_value={"intent": "policy_only", "booking_id": "", "search": ""},
        ):
            with patch("app.call_llm", side_effect=OpenAIError(sentinel)):
                with self.assertLogs("ridewise", level="WARNING") as captured:
                    answer, citations = handle_question(
                        "How should I charge an EV?",
                        "Support Agent",
                        None,
                    )

        self.assertIn("LLM is unavailable", answer)
        self.assertTrue(citations)
        self.assertNotIn(sentinel, answer)
        self.assertNotIn(sentinel, "\n".join(captured.output))

    def test_llm_unavailable_message_contains_no_credentials(self) -> None:
        answer, citations = llm_unavailable_response("cancellation refund policy")
        self.assertNotIn("test-secret-value", answer)
        self.assertTrue(citations)


if __name__ == "__main__":
    unittest.main()
