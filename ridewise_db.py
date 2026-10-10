"""Read-only, role-scoped access to the supplied RideWise SQLite database."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
from typing import Any
from collections.abc import Iterator


BOOKING_ID_PATTERN = re.compile(r"^B-[0-9]+$")
CUSTOMER_ID_PATTERN = re.compile(r"^CUST[0-9]+$")


class DataAccessError(Exception):
    """Raised when a data request cannot be safely completed."""


def _open_database(db_path: Path) -> sqlite3.Connection:
    path = db_path.resolve()
    if not path.is_file():
        raise DataAccessError("The RideWise database file is unavailable.")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


@contextmanager
def _read_database(db_path: Path) -> Iterator[sqlite3.Connection]:
    connection = _open_database(db_path)
    try:
        yield connection
    finally:
        connection.close()


def _validate_role(role: str, customer_id: str | None) -> None:
    if role not in {"Customer", "Support Agent"}:
        raise DataAccessError("Unsupported session role.")
    if role == "Customer":
        if not customer_id or not CUSTOMER_ID_PATTERN.fullmatch(customer_id):
            raise DataAccessError("Enter a valid customer ID to access customer data.")


def _owner_clause(role: str, customer_id: str | None) -> tuple[str, tuple[str, ...]]:
    if role == "Customer":
        return " AND b.customer_id = ?", (customer_id or "",)
    return "", ()


def _dict_rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def get_my_bookings(
    db_path: Path, role: str, customer_id: str | None, limit: int = 10
) -> list[dict[str, Any]]:
    _validate_role(role, customer_id)
    if role != "Customer":
        raise DataAccessError("This request is only available in Customer mode.")
    with _read_database(db_path) as connection:
        rows = connection.execute(
            """
            SELECT b.booking_id, b.booked_at, b.scheduled_start, b.scheduled_end,
                   b.actual_return_at, b.cancelled_at, b.booking_status,
                   v.model, v.vehicle_type, v.hub_name
            FROM bookings AS b
            LEFT JOIN vehicles AS v ON v.vehicle_id = b.vehicle_id
            WHERE b.customer_id = ?
            ORDER BY b.booked_at DESC
            LIMIT ?
            """,
            (customer_id, max(1, min(limit, 50))),
        ).fetchall()
    return _dict_rows(rows)


def get_customer_validation(
    db_path: Path,
    role: str,
    active_customer_id: str | None,
    requested_customer_id: str,
) -> dict[str, Any] | None:
    """Return minimal customer validation facts within the active role's scope."""
    _validate_role(role, active_customer_id)
    if role != "Customer":
        raise DataAccessError("Customer validation is only available in Customer mode.")
    if not CUSTOMER_ID_PATTERN.fullmatch(requested_customer_id):
        raise DataAccessError("Enter a valid customer ID to check.")
    if role == "Customer" and requested_customer_id != active_customer_id:
        raise DataAccessError(
            "Customer sessions can only validate the customer ID for the active session."
        )
    with _read_database(db_path) as connection:
        row = connection.execute(
            """
            SELECT customer_id, kyc_status, customer_tier
            FROM customers
            WHERE customer_id = ?
            """,
            (requested_customer_id,),
        ).fetchone()
    return dict(row) if row else None


def get_dashboard_bookings(
    db_path: Path, role: str, customer_id: str | None, limit: int = 10
) -> list[dict[str, Any]]:
    """Return recent bookings and successful-payment summaries within role scope."""
    _validate_role(role, customer_id)
    owner_sql, owner_params = _owner_clause(role, customer_id)
    with _read_database(db_path) as connection:
        rows = connection.execute(
            """
            SELECT b.booking_id, b.vehicle_id, b.booked_at, b.scheduled_start,
                   b.booking_status, v.model,
                   (
                       SELECT COUNT(*)
                       FROM payments AS p
                       WHERE p.booking_id = b.booking_id
                         AND p.payment_status = 'SUCCESS'
                   ) AS successful_payment_count,
                   (
                       SELECT SUM(COALESCE(p.amount, 0))
                       FROM payments AS p
                       WHERE p.booking_id = b.booking_id
                         AND p.payment_status = 'SUCCESS'
                   ) AS successful_payment_amount,
                   (
                       SELECT SUM(COALESCE(p.gst_amount, 0))
                       FROM payments AS p
                       WHERE p.booking_id = b.booking_id
                         AND p.payment_status = 'SUCCESS'
                   ) AS successful_payment_gst
            FROM bookings AS b
            LEFT JOIN vehicles AS v ON v.vehicle_id = b.vehicle_id
            WHERE 1 = 1
            """
            + owner_sql
            + """
            ORDER BY b.booked_at DESC
            LIMIT ?
            """,
            (*owner_params, max(1, min(limit, 50))),
        ).fetchall()
    return _dict_rows(rows)


def get_recent_payments(
    db_path: Path, role: str, customer_id: str | None, limit: int = 25
) -> list[dict[str, Any]]:
    """Return recent payment rows after applying booking ownership in SQL."""
    _validate_role(role, customer_id)
    owner_sql, owner_params = _owner_clause(role, customer_id)
    with _read_database(db_path) as connection:
        rows = connection.execute(
            """
            SELECT p.booking_id, p.payment_type, p.amount, p.gst_amount,
                   p.payment_status, p.paid_at
            FROM payments AS p
            JOIN bookings AS b ON b.booking_id = p.booking_id
            WHERE 1 = 1
            """
            + owner_sql
            + """
            ORDER BY p.paid_at DESC
            LIMIT ?
            """,
            (*owner_params, max(1, min(limit, 100))),
        ).fetchall()
    return _dict_rows(rows)


def get_dashboard_metrics(
    db_path: Path, role: str, customer_id: str | None
) -> dict[str, int]:
    """Return aggregate dashboard counts, scoped to a customer when applicable."""
    _validate_role(role, customer_id)
    with _read_database(db_path) as connection:
        if role == "Customer":
            booking_filter = " WHERE customer_id = ?"
            params = (customer_id,)
            booking_count = connection.execute(
                "SELECT COUNT(*) FROM bookings" + booking_filter, params
            ).fetchone()[0]
            completed_count = connection.execute(
                "SELECT COUNT(*) FROM bookings"
                + booking_filter
                + " AND booking_status = 'CMP'",
                params,
            ).fetchone()[0]
            successful_payment_count = connection.execute(
                "SELECT COUNT(*) FROM payments WHERE payment_status = 'SUCCESS' "
                "AND booking_id IN "
                "(SELECT booking_id FROM bookings WHERE customer_id = ?)",
                params,
            ).fetchone()[0]
            cancelled_count = connection.execute(
                "SELECT COUNT(*) FROM bookings"
                + booking_filter
                + " AND booking_status IN ('CNL_U', 'CNL_S')",
                params,
            ).fetchone()[0]
        else:
            booking_count = connection.execute(
                "SELECT COUNT(*) FROM bookings"
            ).fetchone()[0]
            completed_count = connection.execute(
                "SELECT COUNT(*) FROM bookings WHERE booking_status = 'CMP'"
            ).fetchone()[0]
            successful_payment_count = connection.execute(
                "SELECT COUNT(*) FROM payments WHERE payment_status = 'SUCCESS'"
            ).fetchone()[0]
            cancelled_count = connection.execute(
                "SELECT COUNT(*) FROM bookings "
                "WHERE booking_status IN ('CNL_U', 'CNL_S')"
            ).fetchone()[0]
    return {
        "bookings": booking_count,
        "completed_bookings": completed_count,
        "successful_payments": successful_payment_count,
        "cancelled_bookings": cancelled_count,
    }


def get_booking_evidence(
    db_path: Path, role: str, customer_id: str | None, booking_id: str
) -> dict[str, Any] | None:
    _validate_role(role, customer_id)
    if not BOOKING_ID_PATTERN.fullmatch(booking_id):
        raise DataAccessError("Enter a booking ID such as B-0001.")
    owner_sql, owner_params = _owner_clause(role, customer_id)
    with _read_database(db_path) as connection:
        booking = connection.execute(
            """
            SELECT b.booking_id, b.customer_id, b.vehicle_id, b.booked_at,
                   b.scheduled_start, b.scheduled_end, b.actual_return_at,
                   b.cancelled_at, b.booking_status, b.return_battery_pct,
                   v.model, v.vehicle_type, v.hub_name
            FROM bookings AS b
            LEFT JOIN vehicles AS v ON v.vehicle_id = b.vehicle_id
            WHERE b.booking_id = ?
            """
            + owner_sql,
            (booking_id, *owner_params),
        ).fetchone()
        if booking is None:
            return None
        payments = connection.execute(
            """
            SELECT p.payment_type, p.amount, p.gst_amount, p.payment_status,
                   p.paid_at
            FROM payments AS p
            WHERE p.booking_id = ?
            ORDER BY p.paid_at, p.payment_type
            """,
            (booking_id,),
        ).fetchall()
        damage_reports = connection.execute(
            """
            SELECT d.report_id, d.damage_category, d.severity, d.assessed_cost,
                   d.claim_status, d.reported_at
            FROM damage_reports AS d
            WHERE d.booking_id = ?
            ORDER BY d.reported_at
            """,
            (booking_id,),
        ).fetchall()
    evidence = dict(booking)
    evidence.pop("customer_id", None)
    evidence["payments"] = _dict_rows(payments)
    evidence["damage_reports"] = _dict_rows(damage_reports)
    return evidence


def execute_intent(
    db_path: Path,
    role: str,
    customer_id: str | None,
    intent: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Execute one supported data operation; all SQL is fixed in this module."""
    _validate_role(role, customer_id)
    if intent == "my_bookings":
        if role != "Customer":
            raise DataAccessError("This request is only available in Customer mode.")
        return {"bookings": get_my_bookings(db_path, role, customer_id)}

    if intent == "customer_profile":
        if role != "Customer":
            raise DataAccessError("This request is only available in Customer mode.")
        with _read_database(db_path) as connection:
            row = connection.execute(
                """
                SELECT customer_id, kyc_status, customer_tier, created_at
                FROM customers WHERE customer_id = ?
                """,
                (customer_id,),
            ).fetchone()
        return {"profile": dict(row) if row else None}

    if intent in {"booking_lookup", "payment_lookup", "damage_lookup"}:
        booking_id = arguments.get("booking_id")
        if not isinstance(booking_id, str) or not BOOKING_ID_PATTERN.fullmatch(booking_id):
            raise DataAccessError("A valid booking ID is needed for this lookup.")
        owner_sql, owner_params = _owner_clause(role, customer_id)
        with _read_database(db_path) as connection:
            exists = connection.execute(
                "SELECT b.booking_id FROM bookings AS b "
                "WHERE b.booking_id = ?" + owner_sql,
                (booking_id, *owner_params),
            ).fetchone()
            if exists is None:
                return {"found": False, "records": []}
            if intent == "booking_lookup":
                rows = connection.execute(
                    """
                    SELECT b.booking_id, b.booked_at, b.scheduled_start,
                           b.scheduled_end, b.actual_return_at, b.cancelled_at,
                           b.booking_status, v.model, v.vehicle_type, v.hub_name
                    FROM bookings AS b
                    LEFT JOIN vehicles AS v ON v.vehicle_id = b.vehicle_id
                    WHERE b.booking_id = ?
                    """,
                    (booking_id,),
                ).fetchall()
            elif intent == "payment_lookup":
                rows = connection.execute(
                    """
                    SELECT payment_type, amount, gst_amount, payment_status, paid_at
                    FROM payments
                    WHERE booking_id = ?
                    ORDER BY paid_at
                    """,
                    (booking_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT report_id, damage_category, severity, assessed_cost,
                           claim_status, reported_at
                    FROM damage_reports
                    WHERE booking_id = ?
                    ORDER BY reported_at
                    """,
                    (booking_id,),
                ).fetchall()
        return {"found": True, "records": _dict_rows(rows)}

    if intent == "vehicle_lookup":
        if role != "Support Agent":
            raise DataAccessError("Vehicle operations lookups require Support Agent mode.")
        search = arguments.get("search", "")
        if not isinstance(search, str):
            raise DataAccessError("Vehicle search must be text.")
        with _read_database(db_path) as connection:
            rows = connection.execute(
                """
                SELECT vehicle_id, vehicle_type, model, hub_name, status,
                       hourly_rate, daily_rate, battery_capacity_kwh
                FROM vehicles
                WHERE vehicle_id LIKE ? OR model LIKE ? OR hub_name LIKE ?
                ORDER BY vehicle_id
                LIMIT 25
                """,
                (f"%{search[:80]}%", f"%{search[:80]}%", f"%{search[:80]}%"),
            ).fetchall()
        return {"vehicles": _dict_rows(rows)}

    if intent == "agent_counts":
        if role != "Support Agent":
            raise DataAccessError("Operational counts require Support Agent mode.")
        counts: dict[str, int] = {}
        with _open_database(db_path) as connection:
            for table in (
                "customers",
                "vehicles",
                "bookings",
                "payments",
                "damage_reports",
            ):
                counts[table] = connection.execute(
                    "SELECT COUNT(*) FROM " + table
                ).fetchone()[0]
        return {"table_counts": counts}

    if intent == "policy_only":
        return {}

    raise DataAccessError("The request could not be mapped to a supported data lookup.")
