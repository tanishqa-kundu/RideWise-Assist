# RideWise Assist — Design Decisions

## 1. Fixed read-only database operations

The model never generates or executes SQL. `ridewise_db.py` exposes a small allowlist of parameterised queries and opens the supplied SQLite database in read-only mode. Customer booking and payment evidence is filtered by the active customer ID in the database layer before any result is sent to the model. This trades unrestricted natural-language querying for a smaller, safer MVP surface.

## 2. Deterministic ProofMode for refunds

Refund arithmetic and cancellation-version selection are ordinary Python logic, not model output. The version is selected by booking creation time (`booked_at`), with the exact cutoff at 1 July 2026, 00:00 IST. Only successful rental payments contribute to the refund base; GST is refunded proportionally. Existing successful refund records are shown separately. ProofMode does not claim to initiate or settle payments.

## 3. Page-level policy retrieval with citations

The five supplied policy PDFs are searched by page using lightweight lexical overlap; responses include source filenames and page numbers. This is easy to inspect and does not require a vector service or embedding model. It may miss image-only or semantically phrased passages, so uncertain retrieval should be improved before production.

## Rejected alternatives

- Arbitrary SQL from the LLM was rejected because it is difficult to constrain and could disclose unrelated customer data.
- LLM-generated refund calculations were rejected because payment evidence, policy cutoffs, and money rounding need to be reproducible.
- External LLMs or embedding providers were rejected; the event requires use of the NeoStats-hosted model.

## Current weaknesses

Customer ID selection is not authentication. Lexical retrieval is basic and may fail on scanned PDF pages. Intent routing depends on the model and gracefully degrades when it is unavailable, but conversational database queries are limited to the fixed operations. Refunds with inconsistent timestamps, missing successful payments, or unclear status are surfaced as ambiguous rather than guessed.

## Next steps with another week

Add authenticated customer identity, expand regression tests using unseen and adversarial prompts, improve retrieval for scanned PDFs and contradictory policy versions, and run a privacy review of support-agent access and logs.
