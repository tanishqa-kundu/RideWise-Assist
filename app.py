from __future__ import annotations

import json
import html
import logging
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError
import streamlit as st

from policy_retrieval import PolicyPage, PolicyRetriever, PolicyRetrievalError, format_policy_sources
from proofmode import calculate_refund
from ridewise_db import (
    DataAccessError,
    execute_intent,
    get_booking_evidence,
    get_customer_validation,
    get_dashboard_bookings,
    get_dashboard_metrics,
    get_my_bookings,
    get_recent_payments,
)


ROOT = Path(__file__).resolve().parent
BOOKING_PATTERN = re.compile(r"\bB-[0-9]+\b", re.IGNORECASE)
CUSTOMER_ID_PATTERN = re.compile(r"\bCUST[0-9]+\b", re.IGNORECASE)
CUSTOMER_VALIDATION_TERMS = re.compile(
    r"\b(?:valid|validate|verification|verify|exists|existence)\b.*\bcustomer\b"
    r"|\bcustomer\b.*\b(?:valid|validate|verification|verify|exists|existence)\b",
    re.IGNORECASE,
)
BOOKING_LIST_TERMS = re.compile(
    r"\bmy\s+bookings?\b|\bmy\s+booking\s+(?:ids|history)\b"
    r"|\b(?:list|show|view)\s+(?:my\s+)?bookings?\b"
    r"|\bbooking\s+(?:ids|history)\b|^\s*bookings?\s*[?.!]*\s*$",
    re.IGNORECASE,
)
REFUND_TERMS = re.compile(r"\brefund(?:s|ed|ing)?\b|\bcancel(?:lation|led|ed)?\b", re.I)
LOGGER = logging.getLogger("ridewise")

load_dotenv(ROOT / ".env")


class LLMConfigurationError(RuntimeError):
    """Raised when required NeoStats environment configuration is missing."""


INTENT_INSTRUCTIONS = """
Choose exactly one intent from this allowlist:
policy_only, my_bookings, customer_profile, booking_lookup, payment_lookup,
damage_lookup, vehicle_lookup, agent_counts.
Return only JSON with keys "intent" and "booking_id" and "search".
Use a booking ID only if explicitly present in the request. For a vehicle lookup,
put the short model, vehicle ID, or hub query in "search". Never return SQL.
"""


def get_client() -> OpenAI:
    api_key = os.getenv("NEOSTATS_API_KEY", "").strip()
    base_url = os.getenv("NEOSTATS_BASE_URL", "").strip()
    model = os.getenv("NEOSTATS_MODEL", "").strip()
    if not api_key:
        raise LLMConfigurationError(
            "NeoStats is not configured yet: set NEOSTATS_API_KEY in the project "
            ".env file to enable chat responses. Policy search and ProofMode "
            "refund checks remain available."
        )
    if not base_url or not model:
        raise LLMConfigurationError(
            "NeoStats configuration is incomplete: set NEOSTATS_BASE_URL and "
            "NEOSTATS_MODEL in the project .env file."
        )
    return OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=1)


def call_llm(messages: list[dict[str, str]]) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=os.environ["NEOSTATS_MODEL"].strip(),
        messages=messages,
        temperature=0.1,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The NeoStats model returned an empty response.")
    return content


def plan_intent(
    question: str, role: str, history: list[dict[str, str]]
) -> dict[str, str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You route user questions to fixed application operations. "
                "Do not answer the user or generate SQL. "
                + INTENT_INSTRUCTIONS
            ),
        }
    ]
    messages.extend(
        message
        for message in history[-6:]
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
    )
    messages.append(
        {
            "role": "user",
            "content": json.dumps({"role": role, "request": question}),
        }
    )
    response = call_llm(messages)
    try:
        candidate = json.loads(response)
    except json.JSONDecodeError:
        return {"intent": "policy_only", "booking_id": "", "search": ""}
    if not isinstance(candidate, dict):
        return {"intent": "policy_only", "booking_id": "", "search": ""}
    allowed = {
        "policy_only",
        "my_bookings",
        "customer_profile",
        "booking_lookup",
        "payment_lookup",
        "damage_lookup",
        "vehicle_lookup",
        "agent_counts",
    }
    intent = candidate.get("intent")
    if intent not in allowed:
        intent = "policy_only"
    booking_id = candidate.get("booking_id", "")
    search = candidate.get("search", "")
    return {
        "intent": intent,
        "booking_id": booking_id if isinstance(booking_id, str) else "",
        "search": search if isinstance(search, str) else "",
    }


@st.cache_resource(show_spinner=False)
def load_policy_retriever(root: str) -> PolicyRetriever:
    return PolicyRetriever(Path(root))


def retrieve_pages(question: str) -> tuple[list[PolicyPage], str | None]:
    try:
        retriever = load_policy_retriever(str(ROOT))
        pages = retriever.search(question, limit=4)
        return pages, None
    except PolicyRetrievalError as error:
        return [], str(error)


def llm_unavailable_response(question: str) -> tuple[str, list[str]]:
    pages, retrieval_error = retrieve_pages(question)
    answer = (
        "The NeoStats LLM is unavailable because its credentials are missing or the "
        "service could not be reached. Database-backed ProofMode refund checks remain "
        "available. Relevant policy excerpts are included below when found."
    )
    citations = [page.source for page in pages]
    if pages:
        answer += "\n\nRelevant policy excerpts:\n\n" + format_policy_sources(pages)
    elif retrieval_error:
        answer += (
            "\n\nPolicy search is temporarily unavailable because the supplied "
            "documents could not be read."
        )
    else:
        answer += "\n\nNo matching policy passage was found."
    return answer, citations


def cancellation_policy_sources(version: str) -> tuple[list[PolicyPage], bool]:
    filename = (
        "02_Cancellation_and_Refund_Policy_v1.pdf"
        if version == "v1"
        else "03_Cancellation_and_Refund_Policy_v2.pdf"
    )
    try:
        retriever = load_policy_retriever(str(ROOT))
    except PolicyRetrievalError:
        return [], False
    candidates = [
        page
        for page in retriever.pages
        if page.filename == filename
    ]
    matched = [
        page
        for page in candidates
        if any(
            term in page.text.lower()
            for term in ("refund", "cancellation", "cancelled", "cancelled")
        )
    ]
    thresholds = ("24", "6", "100%", "50%") if version == "v1" else (
        "48",
        "12",
        "2",
        "100%",
        "75%",
        "25%",
    )
    schedule_text = any(
        all(threshold in page.text for threshold in thresholds) for page in candidates
    )
    return matched, schedule_text


def format_booking_proof(evidence: dict[str, Any]) -> str:
    status = evidence["booking_status"]
    lines = [
        f"### ProofMode — booking `{evidence['booking_id']}`",
        f"- Actual booking status: **{status}**",
        f"- Booked at (UTC): `{evidence.get('booked_at') or 'not recorded'}`",
        f"- Scheduled pickup (UTC): `{evidence.get('scheduled_start') or 'not recorded'}`",
        f"- Cancellation time (UTC): `{evidence.get('cancelled_at') or 'not recorded'}`",
    ]
    payments = evidence.get("payments", [])
    lines.append("- Payment evidence:")
    if payments:
        for payment in payments:
            lines.append(
                "  - "
                + str(payment.get("payment_status") or "status not recorded")
                + " "
                + str(payment.get("payment_type") or "payment type not recorded")
                + ": ₹"
                + str(payment.get("amount") if payment.get("amount") is not None else "not recorded")
                + " + GST ₹"
                + str(payment.get("gst_amount") if payment.get("gst_amount") is not None else "not recorded")
                + " ("
                + str(payment.get("paid_at") or "time not recorded")
                + ")"
            )
    else:
        lines.append("  - No payment records found.")

    if status not in {"CNL_U", "CNL_S"}:
        lines.append(
            "- Refund calculation: **not performed** — the booking is not recorded "
            "as cancelled. No cancellation refund was calculated."
        )
        if status == "NSH":
            lines.append(
                "- A no-show is not a cancellation; this ProofMode path does not "
                "calculate a cancellation refund."
            )
        return "\n".join(lines)

    result = calculate_refund(evidence)
    version = result.get("policy_version")
    policy_pages, schedule_text_available = (
        cancellation_policy_sources(version) if version else ([], False)
    )
    lines.append(
        f"- Applicable cancellation policy: **Version {version or 'undetermined'}** "
        "(selected from `booked_at`; cutoff is 1 July 2026, 00:00 IST)."
    )
    if result["status"] == "ambiguous":
        lines.append(f"- Calculation: **not determined** — {result['message']}")
    else:
        lines.extend(
            [
                f"- Cancellation rule: {result['timing']}",
                f"- Policy refund rate: {result['refund_rate_percent']}%",
                f"- Rental refund: ₹{result['rental_refund_inr']}",
                f"- GST refunded proportionally: ₹{result['gst_refund_inr']}",
                f"- Total calculated refund: **₹{result['gross_refund_inr']}**",
                f"- Successful refund already recorded: ₹{result['already_recorded_refund_inr']}",
            ]
        )
        if result.get("remaining_refund_inr") is not None:
            lines.append(
                f"- Remaining calculated refund: **₹{result['remaining_refund_inr']}**"
            )
        if result.get("caveat"):
            lines.append(f"- Caution: {result['caveat']}")
        if result.get("deposit_payments"):
            lines.append(
                "- Deposit evidence is listed above but excluded from the rental-refund "
                "calculation; the database does not show a separate deposit-return confirmation."
            )
    lines.append("- Calculation steps:")
    if result["status"] == "calculated":
        lines.extend(
            [
                f"  1. Eligible paid rental amount = ₹{result['rental_paid_inr']}.",
                f"  2. Apply the policy rate of {result['refund_rate_percent']}% "
                f"to get ₹{result['rental_refund_inr']} rental refund.",
                f"  3. Apply the same rate proportionally to recorded GST "
                f"(₹{result['gst_paid_inr']}) to get ₹{result['gst_refund_inr']} GST refund.",
                f"  4. Total calculated refund = ₹{result['gross_refund_inr']}.",
            ]
        )
    if result.get("policy_source"):
        lines.append(
            f"- Refund schedule source: `{result['policy_source']}`, "
            f"page {result.get('policy_page', 'not available')}."
        )
    if policy_pages:
        lines.append(
            "- Retrieved policy context: "
            + "; ".join(f"`{page.source}`" for page in policy_pages)
        )
    if not schedule_text_available:
        lines.append(
            "- Limitation: the applicable refund-slab text could not be retrieved "
            "as searchable text from the PDF. The cited schedule page is provided "
            "for manual verification; no policy passage is quoted."
        )
    lines.append("- Database evidence source: `ridewise.db`, tables `bookings` and `payments`.")
    damage_reports = evidence.get("damage_reports", [])
    if damage_reports:
        lines.append("- Damage reports:")
        for damage in damage_reports:
            lines.append(
                "  - "
                + str(damage.get("damage_category") or "category not recorded")
                + ", "
                + str(damage.get("severity") or "severity not recorded")
                + ", assessed ₹"
                + str(damage.get("assessed_cost") if damage.get("assessed_cost") is not None else "not recorded")
                + ", status "
                + str(damage.get("claim_status") or "not recorded")
            )
    lines.append(
        "- ProofMode reports calculated policy amounts; it does not initiate a payment or confirm gateway settlement."
    )
    return "\n".join(lines)


def handle_question(
    question: str,
    role: str,
    customer_id: str | None,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, list[str]]:
    history = history or []
    if CUSTOMER_VALIDATION_TERMS.search(question):
        id_match = CUSTOMER_ID_PATTERN.search(question)
        requested_id = (
            id_match.group(0).upper()
            if id_match
            else customer_id if role == "Customer" else ""
        )
        if not requested_id:
            return "Provide a customer ID to validate.", []
        try:
            profile = get_customer_validation(
                ROOT / "ridewise.db", role, customer_id, requested_id
            )
        except DataAccessError as error:
            return str(error), []
        if profile is None:
            return (
                f"Customer validation: **{requested_id} — NOT FOUND**. "
                "No customer record exists for this ID.",
                ["ridewise.db: customers"],
            )
        kyc_status = str(profile.get("kyc_status") or "NOT RECORDED")
        kyc_verified = kyc_status.upper() == "VERIFIED"
        return (
            f"Customer validation: **{profile['customer_id']} — FOUND**\n"
            f"- KYC status: **{kyc_status}** "
            f"({'verified' if kyc_verified else 'not verified'})\n"
            f"- Customer tier: **{profile.get('customer_tier') or 'not recorded'}**",
            ["ridewise.db: customers"],
        )
    if BOOKING_LIST_TERMS.search(question):
        if role != "Customer":
            return "Booking history is available through a Customer session.", []
        try:
            bookings = get_my_bookings(
                ROOT / "ridewise.db", role, customer_id, limit=50
            )
        except DataAccessError as error:
            return str(error), []
        if not bookings:
            return "No bookings were found for the active customer session.", [
                "ridewise.db: bookings"
            ]
        rows = [
            "| Booking ID | Status | Scheduled pickup |",
            "|---|---|---|",
        ]
        rows.extend(
            f"| {booking['booking_id']} | {booking.get('booking_status') or 'not recorded'} "
            f"| {booking.get('scheduled_start') or 'not recorded'} |"
            for booking in bookings
        )
        return (
            "Bookings for the active customer session:\n\n" + "\n".join(rows),
            ["ridewise.db: bookings"],
        )
    booking_match = BOOKING_PATTERN.search(question)
    is_refund = bool(REFUND_TERMS.search(question))
    if booking_match:
        booking_id = booking_match.group(0).upper()
        evidence = get_booking_evidence(
            ROOT / "ridewise.db", role, customer_id, booking_id
        )
        if evidence is None:
            return (
                "I could not find that booking within the records available to this session.",
                [],
            )
        return format_booking_proof(evidence), [
            "ridewise.db: bookings and payments"
        ]
    if is_refund and not booking_match:
        for message in reversed(history):
            if message.get("role") == "user":
                booking_match = BOOKING_PATTERN.search(message.get("content", ""))
                if booking_match:
                    break
    if is_refund:
        if not booking_match:
            if role == "Customer":
                bookings = get_my_bookings(ROOT / "ridewise.db", role, customer_id)
                if not bookings:
                    return (
                        "I could not find bookings for this customer ID. Please verify the ID with support.",
                        [],
                    )
                own_ids = ", ".join(f"`{item['booking_id']}`" for item in bookings)
                return (
                    "To calculate a refund, tell me the booking ID. I found these recent bookings "
                    f"for your account: {own_ids}.",
                    [],
                )
            return ("Please provide the booking ID so I can verify the refund evidence.", [])
    try:
        plan = plan_intent(question, role, history)
    except LLMConfigurationError:
        return llm_unavailable_response(question)
    except OpenAIError as error:
        LOGGER.warning("Intent routing unavailable (%s).", type(error).__name__)
        return llm_unavailable_response(question)
    if role == "Customer" and plan["intent"] in {
        "vehicle_lookup",
        "agent_counts",
    }:
        plan["intent"] = "policy_only"
    arguments: dict[str, Any] = {}
    if plan["booking_id"]:
        arguments["booking_id"] = plan["booking_id"].upper()
    if plan["search"]:
        arguments["search"] = plan["search"]

    data: dict[str, Any] = {}
    try:
        if plan["intent"] != "policy_only":
            data = execute_intent(
                ROOT / "ridewise.db",
                role,
                customer_id,
                plan["intent"],
                arguments,
            )
    except DataAccessError as error:
        return str(error), []

    retrieval_query = " ".join(
        [message.get("content", "") for message in history[-4:]] + [question]
    )
    pages, retrieval_error = retrieve_pages(retrieval_query)
    citations = [page.source for page in pages]
    policy_context = format_policy_sources(pages)
    if retrieval_error:
        policy_context = "Policy documents are unavailable: " + retrieval_error
    elif not pages:
        policy_context = "No matching policy text was found in the supplied PDFs."

    system_message = (
        "You are RideWise Assist. Answer from the supplied evidence only. "
        "Do not invent database values or policy rules. If evidence is missing, say so "
        "and ask a concise clarification. Never expose data not present in the evidence. "
        "Cite policy filename and page when policy context is used. The database result "
        "was fetched by fixed read-only operations; do not claim to have run SQL."
    )
    context_messages = [
        message
        for message in history[-6:]
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
    ]
    context_messages.append(
        {
            "role": "user",
            "content": json.dumps(
                {
                    "request": question,
                    "role": role,
                    "database_evidence": data,
                    "policy_pages": policy_context,
                },
                default=str,
            ),
        }
    )
    try:
        answer = call_llm(
            [{"role": "system", "content": system_message}, *context_messages]
        )
    except LLMConfigurationError:
        answer, fallback_citations = llm_unavailable_response(question)
        citations.extend(item for item in fallback_citations if item not in citations)
    except OpenAIError as error:
        LOGGER.warning("LLM request unavailable (%s).", type(error).__name__)
        answer, fallback_citations = llm_unavailable_response(question)
        citations.extend(item for item in fallback_citations if item not in citations)
    return answer, citations


def _render_message(message: dict[str, Any]) -> None:
    is_proofmode = (
        message.get("role") == "assistant"
        and str(message.get("content", "")).startswith("### ProofMode")
    )
    if is_proofmode:
        with st.container(border=True):
            st.markdown(message["content"])
            for citation in message.get("citations", []):
                st.caption("Evidence source: " + citation)
        return
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for citation in message.get("citations", []):
            st.caption("Source: " + citation)


def _submit_question(
    question: str, role: str, customer_id: str | None
) -> None:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking permitted records and policy sources..."):
            try:
                history = st.session_state["messages"][-7:-1]
                answer, citations = handle_question(
                    question, role, customer_id, history
                )
            except (DataAccessError, PolicyRetrievalError) as error:
                answer, citations = str(error), []
            except Exception as error:
                LOGGER.warning("Request failed (%s).", type(error).__name__)
                answer = "I could not complete that request safely. Please try again."
                citations = []
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "citations": citations,
            }
            _render_message(assistant_message)
    st.session_state["messages"].append(assistant_message)
    if "ProofMode — booking" in answer:
        st.session_state["last_proofmode"] = answer
        with st.container(border=True):
            st.markdown(
                '<div class="rw-kicker">ProofMode evidence</div>',
                unsafe_allow_html=True,
            )
            st.markdown(answer)


def _dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rw-bg: #0b1118;
            --rw-panel: #111c25;
            --rw-panel-soft: #14232d;
            --rw-line: rgba(151, 184, 194, .16);
            --rw-text: #edf5f5;
            --rw-muted: #91a7ad;
            --rw-teal: #39d0ba;
            --rw-teal-soft: rgba(57, 208, 186, .12);
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse at 12% -10%, rgba(38, 130, 127, .18), transparent 38%),
                var(--rw-bg);
            color: var(--rw-text);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] {
            max-width: 1360px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: #0e1720;
            border-right: 1px solid var(--rw-line);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--rw-muted);
        }
        [data-testid="stRadio"] label:has(input:checked) {
            background: rgba(77, 224, 194, .10);
            border-radius: 9px;
        }
        [data-testid="stRadio"] label:has(input:checked) p {
            color: var(--rw-teal);
            font-weight: 720;
        }
        h1, h2, h3, p, label { color: var(--rw-text); }
        h1 { letter-spacing: -.04em; font-weight: 720; }
        h2, h3 { letter-spacing: -.025em; }
        [data-testid="stCaptionContainer"] p { color: var(--rw-muted); }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, var(--rw-panel-soft), var(--rw-panel));
            border: 1px solid var(--rw-line);
            border-radius: 14px;
            padding: 15px 17px;
            min-height: 104px;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--rw-muted);
            font-size: .78rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: .08em;
        }
        [data-testid="stMetricValue"] { color: var(--rw-text); }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--rw-line);
            border-radius: 14px;
            background: rgba(17, 28, 37, .72);
        }
        [data-testid="stChatMessage"] {
            background: rgba(17, 28, 37, .78);
            border: 1px solid var(--rw-line);
            border-radius: 14px;
            padding: .85rem 1rem;
            margin-bottom: .55rem;
        }
        [data-testid="stChatInput"] {
            border: 1px solid rgba(57, 208, 186, .38);
            border-radius: 14px;
            background: #111c25;
        }
        [data-testid="stChatInput"] textarea { color: var(--rw-text); }
        .rw-brand { display:flex; align-items:center; gap:.65rem; }
        .rw-mark {
            display:flex; align-items:center; justify-content:center;
            width:2.3rem; height:2.3rem; border-radius:11px;
            background:var(--rw-teal-soft); color:var(--rw-teal);
            font-size:1.25rem; border:1px solid rgba(57,208,186,.28);
        }
        .rw-brand-title { font-size:1.18rem; font-weight:720; color:var(--rw-text); }
        .rw-tagline { color:var(--rw-muted); font-size:.88rem; margin-top:.12rem; }
        .rw-role {
            display:inline-block; border:1px solid rgba(57,208,186,.28);
            color:var(--rw-teal); background:var(--rw-teal-soft);
            border-radius:999px; padding:.36rem .72rem; font-size:.78rem;
            font-weight:700; letter-spacing:.05em; text-transform:uppercase;
        }
        .rw-section-kicker {
            color:var(--rw-teal); font-size:.72rem; letter-spacing:.13em;
            font-weight:750; text-transform:uppercase; margin-bottom:.25rem;
        }
        .rw-muted { color:var(--rw-muted); font-size:.9rem; }
        .rw-booking-row {
            display:flex; justify-content:space-between; gap:.8rem;
            align-items:center; padding:.7rem 0; border-bottom:1px solid var(--rw-line);
        }
        .rw-booking-row:last-child { border-bottom:0; }
        .rw-booking-id { color:var(--rw-text); font-weight:700; }
        .rw-booking-meta { color:var(--rw-muted); font-size:.8rem; margin-top:.12rem; }
        .rw-status {
            display:inline-block; padding:.25rem .52rem; border:1px solid var(--rw-line);
            border-radius:999px; color:#b9d1d2; font-size:.73rem; font-weight:700;
        }
        @media (max-width: 720px) {
            [data-testid="stMainBlockContainer"] { padding: 1rem .8rem 2rem; }
            .rw-tagline { font-size:.78rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _reference_dashboard_styles(theme: str) -> None:
    is_light = theme == "Light Mode"
    if is_light:
        page_background = "linear-gradient(145deg, #e8f2f1 0%, #f4f7f8 52%, #e7efef 100%)"
        shell = "#ffffff"
        sidebar = "#ffffff"
        panel = "#ffffff"
        panel_gradient = "linear-gradient(150deg, #ffffff, #f7faf9)"
        text = "#172b35"
        muted = "#61747a"
        line = "rgba(30, 67, 72, .13)"
        shell_shadow = "0 24px 75px rgba(28, 65, 68, .12)"
        art_background = "radial-gradient(circle at 82% 50%, rgba(61,196,167,.12), transparent 34%), linear-gradient(120deg,#eff8f6,#f7faf9)"
        user_input = "#ffffff"
        accent = "#087f72"
        accent_soft = "rgba(8, 127, 114, .10)"
        status_text = "#38515a"
    else:
        page_background = "radial-gradient(ellipse at 50% -20%, rgba(44, 153, 134, .34), transparent 58%), linear-gradient(145deg, #07514b 0%, #063a39 48%, #052e32 100%)"
        shell = "#0a1315"
        sidebar = "#081516"
        panel = "#10191b"
        panel_gradient = "linear-gradient(150deg, #142224, #10191b)"
        text = "#ecf5f2"
        muted = "#91a8a4"
        line = "rgba(151, 202, 195, .13)"
        shell_shadow = "0 24px 85px rgba(0, 13, 13, .28)"
        art_background = "radial-gradient(circle at 82% 50%, rgba(61,196,167,.18), transparent 34%), linear-gradient(120deg,#102326,#0d191b)"
        user_input = "#0e191b"
        accent = "#4de0c2"
        accent_soft = "rgba(77, 224, 194, .12)"
        status_text = "#c0d2ce"

    theme_css = """
        <style>
        :root {
            --rw-page-background: __PAGE_BACKGROUND__;
            --rw-shell: __SHELL__;
            --rw-sidebar: __SIDEBAR__;
            --rw-panel: __PANEL__;
            --rw-panel-gradient: __PANEL_GRADIENT__;
            --rw-line: __LINE__;
            --rw-text: __TEXT__;
            --rw-muted: __MUTED__;
            --rw-shell-shadow: __SHELL_SHADOW__;
            --rw-art-background: __ART_BACKGROUND__;
            --rw-input: __INPUT__;
            --rw-teal: __ACCENT__;
            --rw-teal-soft: __ACCENT_SOFT__;
            --rw-status-text: __STATUS_TEXT__;
        }
        [data-testid="stAppViewContainer"] {
            background: var(--rw-page-background);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMain"] { background: transparent; }
        [data-testid="stMainBlockContainer"] {
            max-width: 1510px;
            margin: 1.15rem auto 2rem;
            padding: 1.45rem 1.8rem 2rem;
            background: var(--rw-shell);
            border: 1px solid var(--rw-line);
            border-radius: 25px;
            box-shadow: var(--rw-shell-shadow);
        }
        [data-testid="stSidebar"] {
            background: var(--rw-sidebar);
            border-right: 1px solid var(--rw-line);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--rw-muted); }
        [data-testid="stRadio"] input { accent-color: var(--rw-teal); }
        [data-testid="stRadio"] label:has(input:checked) {
            background: var(--rw-teal-soft); border-radius: 9px;
        }
        [data-testid="stRadio"] label:has(input:checked) p { color: var(--rw-teal); font-weight: 720; }
        [data-testid="stRadio"] label p, [data-testid="stWidgetLabel"] p { color: var(--rw-muted); }
        h1, h2, h3, h4, p, label { color: var(--rw-text); }
        h1 { letter-spacing: -.045em; font-weight: 740; }
        h2, h3 { letter-spacing: -.03em; }
        [data-testid="stCaptionContainer"] p { color: var(--rw-muted); }
        [data-testid="stMetric"] {
            background: var(--rw-panel-gradient);
            border: 1px solid var(--rw-line);
            border-radius: 15px;
            min-height: 103px;
            padding: 15px 17px;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--rw-muted);
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .085em;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] { color: var(--rw-text); }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--rw-line);
            border-radius: 17px;
            background: var(--rw-panel);
        }
        [data-testid="stDataFrame"] { border: 1px solid var(--rw-line); border-radius: 12px; }
        [data-testid="stChatMessage"] {
            background: var(--rw-panel); border: 1px solid var(--rw-line);
            border-radius: 14px; padding: .8rem 1rem; margin-bottom: .5rem;
        }
        [data-testid="stChatInput"] {
            border: 1px solid rgba(77, 224, 194, .34);
            border-radius: 13px; background: var(--rw-panel);
        }
        [data-testid="stChatInput"] textarea { color: var(--rw-text); }
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: var(--rw-input); color:var(--rw-text); border-color:var(--rw-line);
        }
        [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus,
        [data-testid="stNumberInput"] input:focus {
            border-color: var(--rw-teal); box-shadow: 0 0 0 1px var(--rw-teal);
        }
        [data-testid="stAlert"] {
            background: var(--rw-panel); border: 1px solid var(--rw-line); color:var(--rw-text);
        }
        [data-testid="stDialog"] > div {
            background: var(--rw-panel); border: 1px solid var(--rw-line); color:var(--rw-text);
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background: var(--rw-panel); border: 1px solid var(--rw-line); border-radius: 12px;
        }
        [data-testid="stDataFrame"] * { color: var(--rw-text); }
        button[kind="primary"] {
            background:var(--rw-teal); border-color:var(--rw-teal); color:#ffffff;
        }
        button[kind="primary"]:hover { filter:brightness(.88); color:#ffffff; }
        button[kind="secondary"] {
            background:var(--rw-panel); border-color:var(--rw-line); color:var(--rw-text);
        }
        button[kind="secondary"]:hover { border-color:var(--rw-teal); color:var(--rw-teal); }
        button:disabled { opacity:.56; cursor:not-allowed; }
        button:focus-visible, input:focus-visible {
            outline:2px solid var(--rw-teal); outline-offset:2px;
        }
        [data-testid="stForm"] { border-color:var(--rw-line); }
        .rw-brand { display:flex; align-items:center; gap:.68rem; padding:.45rem .4rem 1.2rem; }
        .rw-mark {
            display:flex; align-items:center; justify-content:center; width:2.35rem; height:2.35rem;
            border-radius:12px; background:var(--rw-teal-soft); color:var(--rw-teal);
            font-size:1.22rem; border:1px solid rgba(77,224,194,.28);
        }
        .rw-brand-title { font-size:1rem; font-weight:760; color:var(--rw-text); }
        .rw-tagline { color:var(--rw-muted); font-size:.77rem; margin-top:.16rem; }
        .rw-topbar { display:flex; align-items:center; gap:1rem; margin-bottom:1.1rem; }
        .rw-overline { color:var(--rw-teal); font-size:.7rem; letter-spacing:.15em; font-weight:750; text-transform:uppercase; }
        .rw-top-title { color:var(--rw-text); font-size:1.55rem; font-weight:740; letter-spacing:-.04em; line-height:1.2; }
        .rw-top-note { color:var(--rw-muted); font-size:.82rem; margin-top:.28rem; }
        .rw-chip {
            display:inline-flex; align-items:center; gap:.45rem; border:1px solid rgba(77,224,194,.24);
            color:var(--rw-teal); background:var(--rw-teal-soft); border-radius:999px;
            padding:.42rem .7rem; font-size:.7rem; font-weight:750; letter-spacing:.06em;
            text-transform:uppercase; white-space:nowrap;
        }
        .rw-kicker { color:var(--rw-teal); font-size:.68rem; letter-spacing:.14em; font-weight:760; text-transform:uppercase; margin-bottom:.35rem; }
        .rw-muted { color:var(--rw-muted); font-size:.86rem; }
        .rw-proof-art {
            min-height:164px; display:flex; align-items:center; justify-content:space-between; gap:1rem;
            margin-top:.7rem; padding:1.1rem 1.2rem; border-radius:14px; overflow:hidden;
            background:var(--rw-art-background);
            border:1px solid rgba(77,224,194,.15);
        }
        .rw-proof-art-title { color:var(--rw-text); font-size:.85rem; font-weight:700; }
        .rw-proof-art-copy { color:var(--rw-muted); font-size:.75rem; line-height:1.6; max-width:270px; margin-top:.4rem; }
        .rw-orbit {
            display:flex; align-items:center; justify-content:center; flex:0 0 82px; height:82px;
            border:1px solid rgba(77,224,194,.27); border-radius:50%; color:var(--rw-teal);
            font-size:2.1rem; box-shadow:0 0 0 12px rgba(77,224,194,.035), 0 0 38px rgba(77,224,194,.13);
        }
        .rw-nav-note { color:var(--rw-muted); font-size:.68rem; letter-spacing:.12em; font-weight:750; text-transform:uppercase; padding:.2rem .55rem .45rem; }
        .rw-session-indicator { border-top:1px solid var(--rw-line); margin-top:1.2rem; padding:.9rem .55rem .25rem; color:var(--rw-muted); font-size:.78rem; }
        .rw-booking-head { color:var(--rw-muted); font-size:.66rem; font-weight:740; letter-spacing:.09em; text-transform:uppercase; padding:.5rem .25rem; }
        .rw-table-row { border-top:1px solid var(--rw-line); padding:.15rem 0; }
        .rw-booking-id { color:var(--rw-text); font-weight:730; }
        .rw-table-meta { color:var(--rw-muted); font-size:.78rem; }
        .rw-status {
            display:inline-block; padding:.24rem .5rem; border:1px solid var(--rw-line);
            border-radius:999px; color:var(--rw-status-text); font-size:.69rem; font-weight:740;
        }
        @media (max-width: 850px) {
            [data-testid="stMainBlockContainer"] { margin:.3rem .35rem 1rem; padding:1rem .85rem 1.4rem; border-radius:18px; }
            .rw-top-title { font-size:1.28rem; }
            .rw-proof-art { min-height:130px; }
        }
        @media (max-width: 560px) {
            [data-testid="stMainBlockContainer"] { margin:0; padding:.8rem .65rem 1.2rem; border-radius:0; border-left:0; border-right:0; }
            .rw-topbar { align-items:flex-start; flex-wrap:wrap; }
            .rw-chip { margin-top:.25rem; }
            .rw-proof-art { padding:.85rem; }
            .rw-orbit { flex-basis:60px; height:60px; font-size:1.6rem; }
        }
        </style>
    """
    for token, value in (
        ("__PAGE_BACKGROUND__", page_background),
        ("__SHELL__", shell),
        ("__SIDEBAR__", sidebar),
        ("__PANEL__", panel),
        ("__PANEL_GRADIENT__", panel_gradient),
        ("__LINE__", line),
        ("__TEXT__", text),
        ("__MUTED__", muted),
        ("__SHELL_SHADOW__", shell_shadow),
        ("__ART_BACKGROUND__", art_background),
        ("__INPUT__", user_input),
        ("__ACCENT__", accent),
        ("__ACCENT_SOFT__", accent_soft),
        ("__STATUS_TEXT__", status_text),
    ):
        theme_css = theme_css.replace(token, value)
    st.markdown(theme_css, unsafe_allow_html=True)


def _render_sidebar() -> tuple[str, str, str, str]:
    with st.sidebar:
        st.markdown(
            """
            <div class="rw-brand">
              <div class="rw-mark">↗</div>
              <div><div class="rw-brand-title">RideWise Assist</div>
              <div class="rw-tagline">Electric mobility support</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="rw-nav-note">Workspace</div>', unsafe_allow_html=True)
        section = st.radio(
            "Workspace navigation",
            ("Dashboard", "Bookings", "Payments", "ProofMode", "Policies", "Support"),
            key="workspace_section",
            label_visibility="collapsed",
        )
        st.divider()
        selected_role = st.radio(
            "Session role",
            ("Customer", "Support Agent"),
            key="selected_role",
        )
        entered_customer_id = ""
        if selected_role == "Customer":
            entered_customer_id = st.text_input(
                "Customer ID",
                placeholder="Enter your customer ID",
                key="customer_id_entry",
            ).strip().upper()
        if st.button(
            "Start new conversation",
            type="primary",
            use_container_width=True,
        ):
            if selected_role == "Customer" and not entered_customer_id:
                st.warning("Enter your customer ID to start a customer session.")
            else:
                st.session_state["active_role"] = selected_role
                st.session_state["active_customer_id"] = (
                    entered_customer_id if selected_role == "Customer" else None
                )
                st.session_state["messages"] = []
                st.session_state["last_proofmode"] = None
                st.session_state["session_active"] = True
        appearance = st.radio(
            "Appearance",
            ("Dark Mode", "Light Mode"),
            key="appearance_theme",
            horizontal=True,
        )
        active = st.session_state.get("session_active", False)
        active_role = st.session_state.get("active_role", "Session not started")
        role_label = active_role
        if not active:
            role_label = "Session not started"
        elif active_role == "Customer" and st.session_state.get(
            "active_customer_id"
        ):
            role_label = (
                f"Customer · {st.session_state['active_customer_id']}"
            )
        st.markdown(
            f'<div class="rw-session-indicator">● &nbsp; Active session<br><strong>{html.escape(role_label)}</strong></div>',
            unsafe_allow_html=True,
        )
        if active and (
            selected_role != active_role
            or (
                active_role == "Customer"
                and entered_customer_id
                != st.session_state.get("active_customer_id")
            )
        ):
            st.caption(
                "Sidebar selection differs from the active session. Start a new "
                "conversation to switch identities."
            )
    return section, selected_role, entered_customer_id, appearance


def _render_topbar(section: str, role: str | None) -> None:
    greeting = "Your mobility support, with evidence." if role else "Welcome to RideWise Assist"
    role_chip = role or "Choose a session"
    llm_state = (
        "AI READY"
        if os.getenv("NEOSTATS_API_KEY", "").strip()
        else "AI UNAVAILABLE · DATA + POLICIES READY"
    )
    st.markdown(
        f"""
        <div class="rw-topbar">
          <div style="flex:1">
            <div class="rw-overline">{html.escape(section)} · DON'T JUST ANSWER. PROVE IT.</div>
            <div class="rw-top-title">{html.escape(greeting)}</div>
            <div class="rw-top-note">Booking records, payment evidence and policy-grounded help in one workspace.</div>
          </div>
          <span class="rw-chip">{html.escape(role_chip)}</span>
          <span class="rw-chip">{html.escape(llm_state)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview_metrics(
    role: str | None, customer_id: str | None
) -> None:
    labels = (
        ("Total bookings", "Completed bookings", "Successful payments", "Cancelled bookings")
    )
    metric_keys = (
        "bookings",
        "completed_bookings",
        "successful_payments",
        "cancelled_bookings",
    )
    metrics: dict[str, int] = {}
    if role:
        try:
            metrics = get_dashboard_metrics(ROOT / "ridewise.db", role, customer_id)
        except DataAccessError as error:
            st.warning("Overview metrics are unavailable: " + str(error))
    columns = st.columns(4)
    for column, label, metric_key in zip(columns, labels, metric_keys):
        value = f"{metrics[metric_key]:,}" if role and metric_key in metrics else "—"
        column.metric(label, value)


def _render_proofmode_feature(role: str | None) -> str | None:
    with st.container(border=True):
        st.markdown('<div class="rw-kicker">ProofMode · evidence first</div>', unsafe_allow_html=True)
        st.subheader("Every answer, backed by the record.")
        st.markdown(
            '<div class="rw-muted">Verify the booking, payments and applicable policy before explaining what happened.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="rw-proof-art">
              <div>
                <div class="rw-proof-art-title">VERIFIED AGAINST RIDEWISE RECORDS</div>
                <div class="rw-proof-art-copy">Booking status · payment and GST facts · cancellation rule · transparent calculation</div>
              </div>
              <div class="rw-orbit">⚡</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("proofmode_inspect_form"):
            booking_id = st.text_input(
                "Booking ID",
                placeholder="Enter booking ID, e.g. B-0001",
                key="proofmode_booking_input",
            )
            inspect = st.form_submit_button(
                "Inspect booking",
                type="primary",
                use_container_width=True,
                disabled=role is None,
            )
    if inspect and booking_id.strip():
        return f"Check booking {booking_id.strip().upper()}"
    if inspect:
        st.warning("Enter a booking ID to inspect.")
    return None


def _render_quick_actions(role: str | None) -> tuple[str | None, str]:
    action_prompt: str | None = None
    with st.container(border=True):
        st.markdown('<div class="rw-kicker">Shortcuts</div>', unsafe_allow_html=True)
        st.subheader("Quick actions")
        booking_id = st.text_input(
            "Booking ID for quick actions",
            placeholder="Booking ID",
            key="quick_action_booking_input",
            disabled=role is None,
        ).strip().upper()
        actions = (
            "Check a booking",
            "Explain a refund",
            "Review a payment",
            "Search rental policies",
        )
        prompts = {
            "Check a booking": f"Check booking {booking_id}",
            "Explain a refund": f"Explain the refund for booking {booking_id}",
            "Review a payment": f"Review the payment for booking {booking_id}",
            "Search rental policies": "Explain the relevant RideWise policy and cite the source.",
        }
        for row in (actions[:2], actions[2:]):
            columns = st.columns(2)
            for column, label in zip(columns, row):
                if column.button(
                    label,
                    key=f"dashboard_action_{label}",
                    use_container_width=True,
                    disabled=role is None,
                ):
                    if label != "Search rental policies" and not BOOKING_PATTERN.fullmatch(booking_id):
                        st.warning("Enter a valid booking ID (for example, B-0001) first.")
                    else:
                        action_prompt = prompts[label]
    return action_prompt, booking_id


def _money(value: Any) -> str:
    if value is None:
        return "—"
    return f"₹{float(value):,.2f}"


def _render_booking_table(
    role: str | None, customer_id: str | None, limit: int = 10
) -> str | None:
    st.markdown('<div class="rw-kicker">Live from the booking ledger</div>', unsafe_allow_html=True)
    st.subheader("Recent bookings")
    if not role:
        st.info("Start a Customer or Support Agent session to view authorised bookings.")
        return None
    try:
        bookings = get_dashboard_bookings(
            ROOT / "ridewise.db", role, customer_id, limit=limit
        )
    except DataAccessError as error:
        st.warning("Bookings are unavailable: " + str(error))
        return None
    if not bookings:
        st.info("No bookings are available for this session.")
        return None

    headers = ("BOOKING ID", "STATUS", "SCHEDULED PICKUP", "VEHICLE", "PAYMENT SUMMARY", "")
    widths = [1.0, .8, 1.4, 1.0, 1.7, .8]
    columns = st.columns(widths)
    for column, label in zip(columns, headers):
        column.markdown(f'<div class="rw-booking-head">{label}</div>', unsafe_allow_html=True)

    selected: str | None = None
    for index, booking in enumerate(bookings):
        columns = st.columns(widths, vertical_alignment="center")
        columns[0].markdown(
            f'<div class="rw-booking-id">{html.escape(str(booking["booking_id"]))}</div>',
            unsafe_allow_html=True,
        )
        columns[1].markdown(
            f'<span class="rw-status">{html.escape(str(booking["booking_status"]))}</span>',
            unsafe_allow_html=True,
        )
        columns[2].markdown(
            f'<div class="rw-table-meta">{html.escape(str(booking.get("scheduled_start") or "Not scheduled"))}</div>',
            unsafe_allow_html=True,
        )
        vehicle = booking.get("model") or booking.get("vehicle_id") or "Not recorded"
        columns[3].markdown(
            f'<div class="rw-table-meta">{html.escape(str(vehicle))}</div>',
            unsafe_allow_html=True,
        )
        payment_summary = "No successful payment"
        if booking["successful_payment_count"]:
            payment_summary = (
                f'{booking["successful_payment_count"]} successful · '
                f'{_money(booking["successful_payment_amount"])} + GST '
                f'{_money(booking["successful_payment_gst"])}'
            )
        columns[4].markdown(
            f'<div class="rw-table-meta">{html.escape(payment_summary)}</div>',
            unsafe_allow_html=True,
        )
        if columns[5].button(
            "Inspect",
            key=f"booking_inspect_{index}_{booking['booking_id']}",
            use_container_width=True,
        ):
            selected = str(booking["booking_id"])
    return selected


def _render_payments(role: str | None, customer_id: str | None) -> str | None:
    st.markdown('<div class="rw-kicker">Payment ledger</div>', unsafe_allow_html=True)
    st.subheader("Recent payments")
    if not role:
        st.info("Start a session to view authorised payment records.")
        return None
    try:
        payments = get_recent_payments(
            ROOT / "ridewise.db", role, customer_id, limit=30
        )
    except DataAccessError as error:
        st.warning("Payments are unavailable: " + str(error))
        return None
    if not payments:
        st.info("No payment records are available for this session.")
        return None
    selected: str | None = None
    for index, payment in enumerate(payments):
        with st.container(border=True):
            columns = st.columns([1, 1, 1, 1, 1.2, .8], vertical_alignment="center")
            columns[0].markdown(f'**{html.escape(str(payment["booking_id"]))}**')
            columns[1].markdown(html.escape(str(payment["payment_type"] or "Payment")))
            columns[2].markdown(html.escape(str(payment["payment_status"] or "Status unavailable")))
            columns[3].markdown(_money(payment.get("amount")))
            columns[4].markdown("GST " + _money(payment.get("gst_amount")))
            columns[5].markdown(html.escape(str(payment.get("paid_at") or "Date unavailable")))
            if st.button(
                "Inspect booking evidence",
                key=f"payment_inspect_{index}_{payment['booking_id']}",
            ):
                selected = str(payment["booking_id"])
    return selected


def _render_policy_search() -> None:
    st.markdown('<div class="rw-kicker">Supplied policy library</div>', unsafe_allow_html=True)
    st.subheader("Search rental policies")
    with st.form("policy_search_form"):
        query = st.text_input(
            "Policy question",
            placeholder="e.g. cancellation, refund, late return, charging",
        )
        submitted = st.form_submit_button("Search policies", type="primary")
    if submitted and query.strip():
        pages, error = retrieve_pages(query.strip())
        if error:
            st.warning("Policy search is unavailable: " + error)
        elif not pages:
            st.info("No matching passage was found in the supplied policy PDFs.")
        else:
            for page in pages:
                with st.container(border=True):
                    st.markdown(f"**{page.source}**")
                    st.write(page.text)


def _render_conversation() -> None:
    st.markdown('<div class="rw-kicker">Conversation history</div>', unsafe_allow_html=True)
    with st.container(height=380, border=True):
        messages = st.session_state.setdefault("messages", [])
        if not messages:
            st.markdown(
                '<div class="rw-muted">Ask about a booking or policy. Booking inspections use ProofMode directly.</div>',
                unsafe_allow_html=True,
            )
        for message in messages:
            _render_message(message)


def _render_saved_proofmode() -> None:
    proof = st.session_state.get("last_proofmode")
    if proof:
        with st.container(border=True):
            st.markdown('<div class="rw-kicker">ProofMode evidence</div>', unsafe_allow_html=True)
            st.markdown(proof)


def main() -> None:
    st.set_page_config(
        page_title="RideWise Assist",
        page_icon="🛵",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state.setdefault("appearance_theme", "Dark Mode")
    section, _, _, appearance = _render_sidebar()
    _dashboard_styles()
    _reference_dashboard_styles(appearance)
    active = st.session_state.get("session_active", False)
    role = st.session_state.get("active_role") if active else None
    customer_id = st.session_state.get("active_customer_id") if active else None
    _render_topbar(section, role)

    if not os.getenv("NEOSTATS_API_KEY", "").strip():
        st.caption(
            "● NeoStats AI is unavailable — database-backed support and policy "
            "retrieval remain available."
        )

    _render_overview_metrics(role, customer_id)
    st.markdown("")

    inspect_prompt: str | None = None
    section_prompt: str | None = None
    section_booking_id: str | None = None

    if section == "Dashboard":
        feature_column, action_column = st.columns([1.28, 1], gap="large")
        with feature_column:
            inspect_prompt = _render_proofmode_feature(role)
        with action_column:
            section_prompt, section_booking_id = _render_quick_actions(role)
        selected_booking = _render_booking_table(role, customer_id, limit=8)
        if selected_booking:
            inspect_prompt = f"Check booking {selected_booking}"
    elif section == "Bookings":
        selected_booking = _render_booking_table(role, customer_id, limit=50)
        if selected_booking:
            inspect_prompt = f"Check booking {selected_booking}"
    elif section == "Payments":
        selected_booking = _render_payments(role, customer_id)
        if selected_booking:
            inspect_prompt = f"Review the payment for booking {selected_booking}"
    elif section == "ProofMode":
        inspect_prompt = _render_proofmode_feature(role)
    elif section == "Policies":
        _render_policy_search()
    elif section == "Support":
        _render_conversation()

    if section in {"Dashboard", "Bookings", "Payments", "ProofMode"}:
        _render_saved_proofmode()
    if section != "Support":
        _render_conversation()

    question = st.chat_input("Ask RideWise Assist…")
    action = inspect_prompt or section_prompt
    if section_prompt and section_booking_id:
        st.session_state["quick_action_booking_input"] = section_booking_id
    if action and role:
        _submit_question(action, role, customer_id)
    elif question and role:
        _submit_question(question, role, customer_id)
    elif question:
        st.info("Start a session before asking about account or booking records.")


if __name__ == "__main__":
    main()
