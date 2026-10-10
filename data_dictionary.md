# RideWise Database - Data Dictionary

`ridewise.db` is a SQLite database with 5 tables.

**Conventions**
- All timestamps are stored as text in the format `YYYY-MM-DD HH:MM:SS`, in **UTC**.
- Dates are stored as text in the format `YYYY-MM-DD`.
- All money values are in Indian Rupees (₹).

## customers
| # | Column | Type | Description |
|---|---|---|---|
| 1 | customer_id | TEXT (PK) | Unique customer ID, format CUST0001 |
| 2 | full_name | TEXT | Customer's full name |
| 3 | phone | TEXT | Registered mobile number (10 digits) |
| 4 | email | TEXT | Registered email address |
| 5 | city | TEXT | City the customer registered from |
| 6 | date_of_birth | TEXT (DATE) | Date of birth, YYYY-MM-DD |
| 7 | driving_license_no | TEXT | Driving licence number |
| 8 | kyc_status | TEXT | VERIFIED, PENDING or REJECTED |
| 9 | customer_tier | TEXT | STANDARD, SILVER or GOLD (tiers affect certain policy benefits) |
| 10 | created_at | TEXT (TIMESTAMP) | Account creation time (UTC) |

## vehicles
| # | Column | Type | Description |
|---|---|---|---|
| 1 | vehicle_id | TEXT (PK) | Unique vehicle ID, format VEH001 |
| 2 | registration_no | TEXT | Registration plate, e.g. KA-05-EV-221 |
| 3 | vehicle_type | TEXT | BIKE, SCOOTER, EV_SCOOTER or EV_BIKE |
| 4 | model | TEXT | Make and model name |
| 5 | hub_name | TEXT | Pickup hub, e.g. Koramangala, Whitefield |
| 6 | hourly_rate | REAL | Rental rate per hour (₹) |
| 7 | daily_rate | REAL | Rental rate per 24 hours (₹) |
| 8 | battery_capacity_kwh | REAL | Battery capacity; NULL for non-EV vehicles |
| 9 | status | TEXT | ACTIVE, MAINTENANCE or RETIRED |
| 10 | commissioned_on | TEXT (DATE) | Date the vehicle joined the fleet |

## bookings
| # | Column | Type | Description |
|---|---|---|---|
| 1 | booking_id | TEXT (PK) | Unique booking ID, format B-0001 |
| 2 | customer_id | TEXT (FK → customers) | Customer who made the booking |
| 3 | vehicle_id | TEXT (FK → vehicles) | Vehicle booked |
| 4 | booked_at | TEXT (TIMESTAMP) | When the booking was made (UTC) |
| 5 | scheduled_start | TEXT (TIMESTAMP) | Planned pickup time (UTC) |
| 6 | scheduled_end | TEXT (TIMESTAMP) | Planned return time (UTC) |
| 7 | actual_return_at | TEXT (TIMESTAMP) | Actual return time (UTC); NULL if not returned |
| 8 | cancelled_at | TEXT (TIMESTAMP) | Cancellation time (UTC); NULL if not cancelled |
| 9 | booking_status | TEXT | Booking status code (see below) |
| 10 | return_battery_pct | INTEGER | Battery % at return (EVs only); NULL otherwise |

**booking_status codes**

| Code | Meaning |
|---|---|
| UPC | Upcoming |
| ONG | Ongoing |
| CMP | Completed |
| CNL_U | Cancelled by customer |
| CNL_S | Cancelled by RideWise |
| NSH | No-show |

## payments
| # | Column | Type | Description |
|---|---|---|---|
| 1 | payment_id | TEXT (PK) | Unique payment ID, format PAY00001 |
| 2 | booking_id | TEXT (FK → bookings) | Booking this payment belongs to |
| 3 | payment_type | TEXT | BOOKING, DEPOSIT, PENALTY or REFUND |
| 4 | amount | REAL | Amount in ₹, excluding GST (refunds are recorded as negative values) |
| 5 | gst_amount | REAL | GST charged on this transaction (₹) |
| 6 | payment_method | TEXT | UPI, CARD, WALLET or CASH |
| 7 | payment_status | TEXT | SUCCESS, FAILED or PENDING |
| 8 | transaction_ref | TEXT | Gateway transaction reference |
| 9 | paid_at | TEXT (TIMESTAMP) | Transaction time (UTC) |
| 10 | remarks | TEXT | Free-text notes from the payment system or staff |

## damage_reports
| # | Column | Type | Description |
|---|---|---|---|
| 1 | report_id | TEXT (PK) | Unique report ID, format DMG0001 |
| 2 | booking_id | TEXT (FK → bookings) | Booking during which the damage was found |
| 3 | vehicle_id | TEXT (FK → vehicles) | Vehicle affected |
| 4 | reported_by | TEXT | CUSTOMER or HUB_STAFF |
| 5 | damage_category | TEXT | SCRATCH, DENT, TYRE, BATTERY, LOST_KEY or OTHER |
| 6 | severity | TEXT | LOW, MEDIUM or HIGH |
| 7 | description | TEXT | Free-text description of the damage |
| 8 | assessed_cost | REAL | Repair cost assessed by the hub (₹) |
| 9 | claim_status | TEXT | OPEN, CHARGED, WAIVED or DISPUTED |
| 10 | reported_at | TEXT (TIMESTAMP) | Report time (UTC) |
