"""Deterministic refund evidence and calculations for RideWise ProofMode."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from zoneinfo import ZoneInfo
from typing import Any


IST = ZoneInfo("Asia/Kolkata")
POLICY_CUTOFF = datetime(2026, 7, 1, 0, 0, tzinfo=IST)
CENT = Decimal("0.01")

POLICY_SOURCES = {
    "v1": ("02_Cancellation_and_Refund_Policy_v1.pdf", 1),
    "v2": ("03_Cancellation_and_Refund_Policy_v2.pdf", 2),
}


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _policy_refund_rate(version: str, hours_before_pickup: Decimal) -> Decimal:
    if version == "v1":
        if hours_before_pickup >= 24:
            return Decimal("1")
        if hours_before_pickup >= 6:
            return Decimal("0.50")
        return Decimal("0")
    if hours_before_pickup >= 48:
        return Decimal("1")
    if hours_before_pickup >= 12:
        return Decimal("0.75")
    if hours_before_pickup >= 2:
        return Decimal("0.25")
    return Decimal("0")


def calculate_refund(evidence: dict[str, Any]) -> dict[str, Any]:
    """Calculate a refund from a booking evidence object, without an LLM."""
    required_fields = ("booked_at", "booking_status", "scheduled_start")
    if any(not evidence.get(field) for field in required_fields):
        return {
            "status": "ambiguous",
            "message": "Required booking timestamps or status are missing.",
        }
    try:
        booked_at = _parse_utc(evidence["booked_at"])
    except (TypeError, ValueError):
        return {
            "status": "ambiguous",
            "message": "The booking timestamp is not a valid ISO timestamp.",
        }
    version = "v1" if booked_at.astimezone(IST) < POLICY_CUTOFF else "v2"
    source_file, policy_page = POLICY_SOURCES[version]
    status = evidence["booking_status"]
    cancelled_at_value = evidence.get("cancelled_at")
    payments = evidence.get("payments", [])

    if status not in {"CNL_U", "CNL_S", "NSH"}:
        return {
            "status": "ambiguous",
            "message": "The booking record does not show a completed cancellation or no-show.",
            "policy_version": version,
            "policy_source": source_file,
            "policy_page": policy_page,
        }

    if status == "CNL_U" and not cancelled_at_value:
        return {
            "status": "ambiguous",
            "message": "The customer-cancellation record has no cancellation timestamp.",
            "policy_version": version,
            "policy_source": source_file,
            "policy_page": policy_page,
        }

    if status == "CNL_S":
        rate = Decimal("1")
        timing = "RideWise cancellation: 100% refund regardless of timing."
    elif status == "NSH":
        rate = Decimal("0")
        timing = "No-show: no rental refund."
    else:
        try:
            scheduled = _parse_utc(evidence["scheduled_start"])
            cancelled = _parse_utc(cancelled_at_value)
        except (TypeError, ValueError):
            return {
                "status": "ambiguous",
                "message": "The pickup or cancellation timestamp is not a valid ISO timestamp.",
                "policy_version": version,
                "policy_source": source_file,
                "policy_page": policy_page,
            }
        if cancelled > scheduled:
            return {
                "status": "ambiguous",
                "message": "Cancellation is recorded after scheduled pickup; the refund window cannot be safely determined.",
                "policy_version": version,
                "policy_source": source_file,
                "policy_page": policy_page,
            }
        hours_before_pickup = Decimal(
            str((scheduled - cancelled).total_seconds() / 3600)
        )
        rate = _policy_refund_rate(version, hours_before_pickup)
        timing = (
            "Cancellation was "
            + str(_money(hours_before_pickup))
            + " hours before scheduled pickup."
        )

    successful_booking_payments = [
        payment
        for payment in payments
        if payment.get("payment_type") == "BOOKING"
        and payment.get("payment_status") == "SUCCESS"
    ]
    successful_refunds = [
        payment
        for payment in payments
        if payment.get("payment_type") == "REFUND"
        and payment.get("payment_status") == "SUCCESS"
    ]
    if any(
        payment.get("amount") is None or payment.get("gst_amount") is None
        for payment in successful_booking_payments + successful_refunds
    ):
        return {
            "status": "ambiguous",
            "message": "A successful rental or refund payment is missing its amount or GST evidence.",
            "policy_version": version,
            "policy_source": source_file,
            "policy_page": policy_page,
            "timing": timing,
        }
    successful_deposits = [
        payment
        for payment in payments
        if payment.get("payment_type") == "DEPOSIT"
        and payment.get("payment_status") == "SUCCESS"
    ]
    if not successful_booking_payments:
        if status == "NSH":
            return {
                "status": "calculated",
                "policy_version": version,
                "policy_source": source_file,
                "policy_page": policy_page,
                "refund_rate_percent": 0,
                "timing": timing,
                "rental_paid_inr": "0.00",
                "gst_paid_inr": "0.00",
                "rental_refund_inr": "0.00",
                "gst_refund_inr": "0.00",
                "gross_refund_inr": "0.00",
                "already_recorded_refund_inr": "0.00",
                "remaining_refund_inr": "0.00",
                "booking_payments": [],
                "refund_payments": successful_refunds,
                "pending_booking_payment_count": 0,
                "booking_status": status,
                "all_payment_evidence": payments,
            }
        return {
            "status": "ambiguous",
            "message": "No successful rental-charge payment is recorded, so the refundable amount cannot be established.",
            "policy_version": version,
            "policy_source": source_file,
            "policy_page": policy_page,
            "timing": timing,
        }

    rental_paid = sum(
        (Decimal(str(payment["amount"])) for payment in successful_booking_payments),
        Decimal("0"),
    )
    gst_paid = sum(
        (
            Decimal(str(payment.get("gst_amount") or 0))
            for payment in successful_booking_payments
        ),
        Decimal("0"),
    )
    already_refunded = sum(
        (
            abs(Decimal(str(payment["amount"])))
            + abs(Decimal(str(payment.get("gst_amount") or 0)))
            for payment in successful_refunds
        ),
        Decimal("0"),
    )
    rental_refund = _money(rental_paid * rate)
    gst_refund = _money(gst_paid * rate)
    gross_refund = rental_refund + gst_refund
    refund_allocation_ambiguous = bool(successful_refunds and successful_deposits)
    remaining_refund = (
        None
        if refund_allocation_ambiguous
        else max(Decimal("0"), gross_refund - already_refunded)
    )

    pending_booking_payments = [
        payment
        for payment in payments
        if payment.get("payment_type") == "BOOKING"
        and payment.get("payment_status") == "PENDING"
    ]
    result = {
        "status": "calculated",
        "policy_version": version,
        "policy_source": source_file,
        "policy_page": policy_page,
        "refund_rate_percent": int(rate * 100),
        "timing": timing,
        "rental_paid_inr": str(_money(rental_paid)),
        "gst_paid_inr": str(_money(gst_paid)),
        "rental_refund_inr": str(rental_refund),
        "gst_refund_inr": str(gst_refund),
        "gross_refund_inr": str(_money(gross_refund)),
        "already_recorded_refund_inr": str(_money(already_refunded)),
        "remaining_refund_inr": (
            None if remaining_refund is None else str(_money(remaining_refund))
        ),
        "booking_payments": successful_booking_payments,
        "refund_payments": successful_refunds,
        "deposit_payments": successful_deposits,
        "all_payment_evidence": payments,
        "refund_allocation_ambiguous": refund_allocation_ambiguous,
        "pending_booking_payment_count": len(pending_booking_payments),
        "booking_status": status,
    }
    caveats: list[str] = []
    if pending_booking_payments:
        caveats.append(
            "Pending rental payments are not counted as paid. Confirm payment settlement "
            "before treating this amount as final."
        )
    if refund_allocation_ambiguous:
        caveats.append(
            "The record includes both a deposit payment and refund payment(s), but the "
            "database does not link refund amounts to rental charges versus deposits. "
            "The remaining rental refund cannot be determined safely."
        )
    if caveats:
        result["caveat"] = " ".join(caveats)
    return result
