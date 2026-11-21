import io
import pymupdf
import pytest

from factcheck.core import (
    extract_pdf_pages,
    fallback_extract_claims,
    _source_quality_hint,
    ALLOWED_STATUSES,
)


def make_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_pdf_pages_reads_text():
    pages = extract_pdf_pages(make_pdf("The Eiffel Tower stands 500 metres tall."))
    assert len(pages) == 1
    assert "500 metres" in pages[0]["text"]


def test_fallback_extracts_numeric_claim_candidate():
    pages = [{"page_number": 2, "text": "In 2024, revenue increased by 35 percent. The product is popular."}]
    claims = fallback_extract_claims(pages)
    assert len(claims) == 1
    assert claims[0]["page_number"] == 2
    assert "35 percent" in claims[0]["original_claim"]


def test_fallback_does_not_extract_opinion_without_number():
    pages = [{"page_number": 1, "text": "This is an excellent product for modern teams."}]
    assert fallback_extract_claims(pages) == []


def test_scanned_pdf_returns_empty_text_for_app_to_handle():
    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    pages = extract_pdf_pages(data)
    assert pages and pages[0]["text"] == ""


def test_source_quality_hint_identifies_government():
    assert "Government" in _source_quality_hint("https://www.nasa.gov/example")


def test_allowed_statuses_keep_insufficient_separate_from_false():
    assert "INSUFFICIENT EVIDENCE" in ALLOWED_STATUSES
    assert "FALSE" in ALLOWED_STATUSES
    assert "INSUFFICIENT EVIDENCE" != "FALSE"


def test_empty_pdf_rejected():
    with pytest.raises(ValueError):
        extract_pdf_pages(b"")


def test_no_search_results_never_returns_false(monkeypatch):
    from factcheck import core
    monkeypatch.setattr(core, "tavily_search", lambda query, api_key, max_results=6: [])
    result = core.verify_claim(
        {"original_claim": "The market grew by 80% in 2025.", "normalized_claim": "The market grew by 80% in 2025.", "page_number": 1},
        tavily_key="test", openai_key="test"
    )
    assert result["status"] == "INSUFFICIENT EVIDENCE"
    assert "not proof" in result["verdict"].lower()


def test_false_without_cited_source_is_downgraded(monkeypatch):
    from factcheck import core
    monkeypatch.setattr(core, "tavily_search", lambda query, api_key, max_results=6: [{
        "source_id": "S1", "title": "One page", "url": "https://example.com/article",
        "snippet": "An unrelated source excerpt", "published_date": "2026-01-01", "score": 0.7
    }])
    monkeypatch.setattr(core, "_openai_json", lambda prompt, api_key, model, timeout=60: {
        "status": "FALSE", "confidence": "High", "verdict": "This is false.",
        "corrected_fact": "Not established", "cited_source_ids": [], "reasoning_checks": [], "limitations": ""
    })
    result = core.verify_claim(
        {"original_claim": "The market grew by 80% in 2025.", "normalized_claim": "The market grew by 80% in 2025.", "page_number": 1},
        tavily_key="test", openai_key="test"
    )
    assert result["status"] == "INSUFFICIENT EVIDENCE"


def test_decisive_verdict_requires_primary_or_independent_sources(monkeypatch):
    from factcheck import core
    calls = {"n": 0}
    def one_secondary_source(query, api_key, max_results=6):
        calls["n"] += 1
        return [{"source_id": "S1", "title": "One page", "url": "https://example.com/article",
                 "snippet": "The excerpt contradicts the statement.", "published_date": "2026-01-01", "score": 0.7}]
    monkeypatch.setattr(core, "tavily_search", one_secondary_source)
    monkeypatch.setattr(core, "_openai_json", lambda prompt, api_key, model, timeout=60: {
        "status": "FALSE", "confidence": "High", "verdict": "Contradicted.",
        "corrected_fact": "A different value", "cited_source_ids": ["S1"], "reasoning_checks": [], "limitations": ""
    })
    result = core.verify_claim(
        {"original_claim": "The market grew by 80% in 2025.", "normalized_claim": "The market grew by 80% in 2025.", "page_number": 1},
        tavily_key="test", openai_key="test"
    )
    assert calls["n"] == 2
    assert result["status"] == "INSUFFICIENT EVIDENCE"
