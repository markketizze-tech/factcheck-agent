from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from factcheck.core import (
    DEFAULT_GROQ_MODEL,
    LEGACY_GROQ_MODEL,
    analyze_pdf,
    extract_pdf_pages,
    resolve_groq_model,
)

st.set_page_config(page_title="FactCheck | Evidence Layer", page_icon="🔎", layout="wide")

# Minimal custom styling; no external assets or remote scripts are loaded.
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.8rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(115deg,#101d35,#183d58); color: white; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin-bottom: .2rem; font-size: 2.1rem;}
.hero p {color: #d7e8f4; margin-bottom: 0;}
.small-muted {color: #64748b; font-size: .88rem;}
.status {font-weight: 700; letter-spacing: .02em;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>FactCheck</h1>
  <p>An evidence layer for claims in documents. Extract → Search live sources → Review the verdict.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Configuration")
    def secret_or_env(name: str, default: str = "") -> str:
        try:
            return str(st.secrets.get(name, os.getenv(name, default)))
        except Exception:
            return os.getenv(name, default)
    groq_key = secret_or_env("GROQ_API_KEY")
    tavily_key = secret_or_env("TAVILY_API_KEY")
    configured_model = secret_or_env("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    model = resolve_groq_model(configured_model)
    if configured_model == LEGACY_GROQ_MODEL:
        st.warning(f"The previous default model is not enabled for this account; using {DEFAULT_GROQ_MODEL}.")
    model = st.text_input("Groq model", value=model)
    max_claims = st.slider("Maximum claims to verify", min_value=1, max_value=25, value=12)
    st.divider()
    st.caption("API keys are loaded from server configuration and are never displayed in the app.")
    st.caption("MVP limits: text-based PDFs, 20 MB, 60 pages. Scanned PDFs need OCR, which is not enabled.")

uploaded = st.file_uploader("Upload a PDF to fact-check", type=["pdf"], help="Text-based PDFs only. Up to 20 MB and 60 pages.")

if uploaded:
    pdf_bytes = uploaded.getvalue()
    c1, c2, c3 = st.columns(3)
    c1.metric("File", uploaded.name[:28])
    c2.metric("Size", f"{len(pdf_bytes)/1024:.0f} KB")
    try:
        pages = extract_pdf_pages(pdf_bytes)
        c3.metric("Pages", len(pages))
        with st.expander("Preview extracted text", expanded=False):
            for p in pages[:3]:
                st.markdown(f"**Page {p['page_number']}**")
                st.write(p["text"][:1800] or "No selectable text found on this page.")
        if st.button("Run fact-check", type="primary", use_container_width=True):
            if not groq_key or not tavily_key:
                st.error("Configure GROQ_API_KEY and TAVILY_API_KEY in Streamlit Secrets or server environment variables before running verification.")
            else:
                with st.status("Fact-checking document...", expanded=True) as status_box:
                    st.write("Extracting factual claims from the PDF")
                    st.write("Searching live web sources for each claim")
                    st.write("Comparing the claim with retrieved evidence")
                    try:
                        result = analyze_pdf(pdf_bytes, tavily_key, groq_key, model, max_claims, True)
                        st.session_state["factcheck_result"] = result
                        st.session_state["factcheck_filename"] = uploaded.name
                        status_box.update(label="Fact-check complete", state="complete", expanded=False)
                    except Exception as exc:
                        status_box.update(label="Fact-check could not complete", state="error", expanded=True)
                        st.error(str(exc))
    except Exception as exc:
        st.error(str(exc))

result = st.session_state.get("factcheck_result")
if result:
    st.divider()
    st.subheader("Results")
    results = result.get("results", [])
    counts = {s: sum(1 for r in results if r.get("status") == s) for s in ["VERIFIED", "INACCURATE", "FALSE", "INSUFFICIENT EVIDENCE", "CONFLICTING EVIDENCE", "REVIEW REQUIRED"]}
    metric_cols = st.columns(6)
    for col, label in zip(metric_cols, counts):
        col.metric(label.title(), counts[label])
    st.caption("A lack of search evidence is shown as Insufficient Evidence, not False. All verdicts should be reviewed before publication or legal use.")

    rows = []
    for i, r in enumerate(results, start=1):
        rows.append({"#": i, "Page": r.get("page_number"), "Status": r.get("status"), "Confidence": r.get("confidence"), "Claim": r.get("original_claim"), "Corrected fact": r.get("corrected_fact")})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for i, r in enumerate(results, start=1):
        status = r.get("status", "REVIEW REQUIRED")
        title = f"{i}. [{status}] {r.get('original_claim', '')[:115]}"
        with st.expander(title, expanded=(status in {"FALSE", "INACCURATE", "CONFLICTING EVIDENCE"} or i == 1)):
            left, right = st.columns([3, 1])
            left.markdown("**Original PDF claim**")
            left.write(r.get("original_claim", ""))
            left.markdown("**Normalized claim**")
            left.write(r.get("normalized_claim", ""))
            left.markdown("**Assessment**")
            left.write(r.get("verdict", ""))
            if r.get("corrected_fact") and r.get("corrected_fact") != "Not established":
                left.markdown("**Corrected / current fact from retrieved evidence**")
                left.write(r.get("corrected_fact"))
            right.metric("Confidence", r.get("confidence", "Low"))
            right.caption(f"PDF page {r.get('page_number', 'N/A')}")
            right.caption(f"Claim type: {r.get('claim_type', 'N/A')}")
            st.markdown("**Live search queries (two passes)**")
            st.code(r.get("search_query", "Not run"), language="text")
            cited_ids = set(r.get("cited_source_ids", []))
            st.markdown("**Retrieved sources**")
            sources = r.get("sources", [])
            if not sources:
                st.info("No usable sources were retrieved.")
            for src in sources:
                marker = " ✓ used in verdict" if src.get("source_id") in cited_ids else ""
                st.markdown(f"**{src.get('source_id')} · {src.get('title')}**{marker}")
                st.caption(f"{src.get('quality_hint', '')} · Published date: {src.get('published_date', 'Not supplied')}")
                st.write(src.get("snippet", ""))
                st.markdown(f"[Open source]({src.get('url')})")
            if r.get("reasoning_checks"):
                st.markdown("**Checks considered**")
                for check in r["reasoning_checks"]:
                    st.write(f"- {check}")
            if r.get("limitations"):
                st.caption(f"Limitation: {r.get('limitations')}")

    st.divider()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = {
        "file": st.session_state.get("factcheck_filename", "uploaded.pdf"),
        "generated_at": stamp,
        "disclaimer": "AI-assisted evidence triage. Verify primary sources and context before relying on findings.",
        "summary": counts,
        "results": results,
    }
    st.download_button("Download JSON report", json.dumps(report, indent=2, ensure_ascii=False), file_name="factcheck_report.json", mime="application/json")
    csv_buf = io.StringIO()
    fieldnames = ["page_number", "status", "confidence", "original_claim", "normalized_claim", "verdict", "corrected_fact", "search_query"]
    writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow({k: r.get(k, "") for k in fieldnames})
    st.download_button("Download CSV summary", csv_buf.getvalue(), file_name="factcheck_summary.csv", mime="text/csv")
    md_lines = [f"# Fact-check report: {report['file']}", f"Generated: {stamp}", "", "> AI-assisted evidence triage. Verify primary sources and context before relying on findings.", ""]
    for i, r in enumerate(results, start=1):
        md_lines += [f"## {i}. {r.get('status')} — page {r.get('page_number')}", f"**Claim:** {r.get('original_claim')}", f"**Assessment:** {r.get('verdict')}", f"**Corrected fact:** {r.get('corrected_fact')}", "**Sources:**"]
        for s in r.get("sources", []):
            md_lines.append(f"- [{s.get('title')}]({s.get('url')}) — {s.get('snippet','')[:300]}")
        md_lines.append("")
    st.download_button("Download Markdown report", "\n".join(md_lines), file_name="factcheck_report.md", mime="text/markdown")

st.divider()
with st.expander("How the verdicts work"):
    st.markdown("""
- **Verified:** retrieved credible evidence materially supports the same date, scope, unit and definition.
- **Inaccurate:** the claim is outdated, incomplete, misdated or materially imprecise.
- **False:** credible evidence directly contradicts the claim. Lack of evidence alone is not enough.
- **Insufficient evidence:** available sources do not establish the claim either way.
- **Conflicting evidence:** credible sources materially disagree.

The system uses an LLM to interpret retrieved passages; it does not turn the model's prior knowledge into evidence. Source links and excerpts are displayed for human inspection. Search indexes may omit relevant evidence, and source excerpts can be incomplete.
""")
