# RideWise Assist — ProofMode MVP

A local Streamlit chat assistant for RideWise customers and support agents. It uses fixed, parameterised, read-only SQLite operations, lexical PDF-page retrieval, and the NeoStats OpenAI-compatible endpoint. Refund answers use deterministic ProofMode calculations rather than model-generated arithmetic.

## Run locally

1. Use Python 3.10 or later.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Copy `.env.example` to `.env` in the project root, then set the event key:

   ```dotenv
   NEOSTATS_API_KEY=your-event-key
   NEOSTATS_BASE_URL=https://neo-hackathon.services.ai.azure.com/openai/v1/
   NEOSTATS_MODEL=Neo-Hackathon
   ```

   Keep `.env` private. It is excluded by `.gitignore`; do not commit it.
5. Start the interface: `streamlit run app.py`.

The application expects `ridewise.db` and the five supplied policy PDFs beside `app.py`. It never writes to the supplied database or PDFs. Customer sessions are selected by customer ID for hackathon evaluation; this is not production authentication.

## Architecture

- `app.py` renders the role-selected chat, calls the NeoStats model for intent routing and grounded conversational responses, and catches unavailable-model errors.
- `ridewise_db.py` is the data-access boundary. It opens SQLite with `mode=ro`, sets `query_only`, accepts only fixed allowlisted operations, binds user-supplied values as parameters, and scopes every customer booking lookup to that session's customer ID. No LLM-generated SQL is executed.
- `policy_retrieval.py` extracts searchable text page-by-page from all five PDFs and returns filename/page citations.
- `proofmode.py` selects cancellation policy by `booked_at` relative to 1 July 2026, 00:00 IST; it computes rental and proportional GST refunds from successful payment evidence and reports recorded refunds separately.

ProofMode reports a calculated amount, not a payment instruction or gateway settlement. Failed and pending payments are not counted as successful rental payments. Security deposits are distinct from rental charges and are not included in the cancellation refund schedule calculation.

## Checks

Run the standard-library tests with `python -m unittest discover -s tests -v`.

## Limitations

- Session role selection and customer ID entry are evaluation controls, not identity verification.
- PDF retrieval is lexical and may not find relevant text inside image-only scans; always verify cited policy passages.
- Ambiguous or incomplete payment/cancellation evidence is reported rather than guessed.
- The interface does not initiate cancellations, refunds, or other database writes.
