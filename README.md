# FactCheck — Evidence Layer for PDF Claims

A Streamlit web app that extracts factual claims from a text-based PDF, searches live web sources, and produces an evidence-linked assessment for each claim.

## What it does

1. Extracts selectable text from a PDF page-by-page (PyMuPDF).
2. Uses an LLM to identify checkable claims without silently correcting the original wording.
3. Runs two complementary live searches through Tavily (exact-claim and broader official-data / methodology query), then deduplicates results by URL.
4. Uses an LLM to compare the claim with retrieved excerpts and classify it as:
   - `VERIFIED`
   - `INACCURATE`
   - `FALSE`
   - `INSUFFICIENT EVIDENCE`
   - `CONFLICTING EVIDENCE`
   - `REVIEW REQUIRED` when a provider call fails
5. Displays source titles, URLs, excerpts, the query used, confidence, corrected fact (when established), and limitations.
6. Exports JSON, CSV and Markdown reports.

**Important policy:** no search result is not proof of falsehood. The app only uses `FALSE` when retrieved evidence directly contradicts the claim. Unsupported claims are `INSUFFICIENT EVIDENCE`.

## Stack

- Python + Streamlit UI
- PyMuPDF for text-based PDF extraction
- Groq Chat Completions API for claim extraction and source-grounded adjudication
- Tavily Search API for live web retrieval
- pytest for deterministic unit tests

This intentionally keeps the MVP simple to deploy. Search and adjudication are separate service layers, so providers can be swapped later.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
export GROQ_API_KEY="your-key"
export TAVILY_API_KEY="your-key"
export GROQ_MODEL="llama-3.3-70b-versatile"  # optional override
streamlit run app.py
```

You can also enter both API keys in the app sidebar. For a public deployment, configure secrets in the hosting platform rather than committing keys.

## Deploy to Streamlit Community Cloud

1. Push this directory to a GitHub repository (the repository root should contain `app.py`, `requirements.txt`, `factcheck/` and `.streamlit/config.toml`).
2. In Streamlit Community Cloud, create an app from the repository and select `app.py` as the entry point.
3. In **App settings → Secrets**, add:

```toml
GROQ_API_KEY = "your-groq-key"
TAVILY_API_KEY = "your-tavily-key"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

4. Configure a Groq API key from https://console.groq.com/keys and a Tavily API key. Deploy and test with `sample_trap_document.pdf` and a separate PDF containing claims unknown to the app.

The deployed app needs Groq and Tavily API credentials configured in Streamlit Secrets before live verification can run.

## Tests

```bash
python -m pytest -q
```

The unit tests cover text extraction, conservative numeric-candidate extraction, scanned-PDF handling, source-domain hints and safe status separation. They do **not** constitute a live API integration test. Before submission, test with valid API keys and retain screenshots or logs of the live run.

## Limits and responsible use

- Text-based PDFs only; scanned/image-only PDFs need OCR and are rejected with a clear message.
- Current limits: 20 MB, 60 pages, 50,000 extracted characters, up to 25 claims per document.
- Live search results are not comprehensive. Search engines may miss primary sources or return stale snippets.
- The model can misread context. Inspect the source URL and excerpt before relying on a verdict. Decisive verdicts are downgraded when the cited evidence is only one non-primary source; the app looks for an authoritative source or corroboration across independent domains.
- Current `confidence` is a qualitative model assessment, not a calibrated statistical probability.
- This is decision support, not legal, financial, scientific or medical certification.
- PDF text and claims are transmitted to the configured AI/search providers to perform the requested analysis. Do not upload confidential or personal data without appropriate approval.
- The app does not persist uploaded PDFs or reports on a server; session state holds results during the active app session.

## Repository layout

```text
.
├── app.py
├── factcheck/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── requirements.txt
├── .streamlit/config.toml
├── sample_trap_document.pdf
└── README.md
```
