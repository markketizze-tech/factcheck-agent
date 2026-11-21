import io
from types import SimpleNamespace

import pymupdf
import pytest

from factcheck.core import (
    DEFAULT_GROQ_MODEL,
    LEGACY_GROQ_MODEL,
    extract_pdf_pages,
    fallback_extract_claims,
    _source_quality_hint,
    ALLOWED_STATUSES,
    resolve_groq_model,
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


def test_ai_extraction_preserves_context_for_search_disambiguation(monkeypatch):
    from factcheck import core
    captured = {}
    monkeypatch.setattr(core, "_groq_json", lambda prompt, api_key, model, **kwargs: captured.update(
        prompt=prompt, schema=kwargs.get("response_schema")
    ) or {
        "claims": [{
            "page_number": 1,
            "original_claim": "45% of NCR sales",
            "normalized_claim": "45% of NCR sales",
            "context": "In the Gurugram luxury housing market, 45% of NCR sales were in this segment.",
            "entities": ["NCR", "Gurugram"],
        }]
    })
    claims = core.extract_claims_with_ai(
        [{"page_number": 1, "text": "In the Gurugram luxury housing market, 45% of NCR sales were in this segment."}],
        "test-key",
    )
    assert claims[0]["context"].startswith("In the Gurugram")
    assert "Do not return fragments" in captured["prompt"]
    assert captured["schema"] == core.CLAIM_EXTRACTION_SCHEMA


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


def test_legacy_groq_default_resolves_to_available_model():
    assert resolve_groq_model(LEGACY_GROQ_MODEL) == DEFAULT_GROQ_MODEL


def test_groq_json_uses_groq_chat_completions(monkeypatch):
    from factcheck import core
    captured = {}
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": '{"ok": true}'}}]},
            text="",
        )

    monkeypatch.setattr(core.requests, "post", fake_post)
    assert core._groq_json("prompt", "test-key", "test-model", response_schema=schema) == {"ok": True}
    assert captured["url"] == core.GROQ_CHAT_URL
    assert captured["json"]["model"] == "test-model"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["response_format"]["type"] == "json_schema"
    assert captured["json"]["response_format"]["json_schema"]["strict"] is True
    assert captured["json"]["response_format"]["json_schema"]["schema"] == schema
    assert captured["json"]["reasoning_effort"] == "low"
    assert captured["json"]["max_completion_tokens"] == 2048


def test_groq_json_reports_provider_errors_without_fallback(monkeypatch):
    from factcheck import core
    monkeypatch.setattr(
        core.requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(status_code=429, text="rate limit"),
    )
    with pytest.raises(RuntimeError, match="Groq rate limit reached"):
        core._groq_json("prompt", "test-key", "test-model")


def test_groq_json_retries_without_json_mode_after_validation_error(monkeypatch):
    from factcheck import core
    requests = []
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}

    def fake_post(url, **kwargs):
        requests.append({**kwargs, "json": dict(kwargs["json"])})
        if len(requests) == 1:
            return SimpleNamespace(
                status_code=400,
                text='{"error":{"code":"json_validate_failed"}}',
            )
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": '{"ok": true}'}}]},
            text="",
        )

    monkeypatch.setattr(core.requests, "post", fake_post)
    assert core._groq_json("prompt", "test-key", "test-model", response_schema=schema) == {"ok": True}
    assert requests[0]["json"]["response_format"]["type"] == "json_schema"
    assert requests[1]["json"]["response_format"] == {"type": "json_object"}
    assert "exactly one valid JSON object" in requests[1]["json"]["messages"][1]["content"]


def test_groq_json_extracts_object_from_markdown_response(monkeypatch):
    from factcheck import core
    monkeypatch.setattr(
        core.requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [{
                    "message": {"content": 'Result:\n```json\n{"ok": true}\n```'}
                }]
            },
            text="",
        ),
    )
    assert core._groq_json("prompt", "test-key", "test-model") == {"ok": True}


def test_groq_json_rejects_non_json_response(monkeypatch):
    from factcheck import core
    monkeypatch.setattr(
        core.requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "not JSON"}}]},
            text="",
        ),
    )
    with pytest.raises(RuntimeError, match="not valid JSON"):
        core._groq_json("prompt", "test-key", "test-model")


def test_no_search_results_never_returns_false(monkeypatch):
    from factcheck import core
    monkeypatch.setattr(core, "tavily_search", lambda query, api_key, max_results=6: [])
    result = core.verify_claim(
        {"original_claim": "The market grew by 80% in 2025.", "normalized_claim": "The market grew by 80% in 2025.", "page_number": 1},
        tavily_key="test", groq_key="test"
    )
    assert result["status"] == "INSUFFICIENT EVIDENCE"
    assert "not proof" in result["verdict"].lower()


def test_verify_claim_uses_pdf_context_to_disambiguate_search(monkeypatch):
    from factcheck import core
    queries = []
    monkeypatch.setattr(
        core,
        "tavily_search",
        lambda query, api_key, max_results=6: queries.append(query) or [],
    )
    core.verify_claim(
        {
            "original_claim": "45% of NCR sales",
            "normalized_claim": "45% of NCR sales",
            "context": "Gurugram luxury housing sales in the National Capital Region",
            "entities": ["Gurugram", "National Capital Region"],
        },
        tavily_key="test",
        groq_key="test",
    )
    assert len(queries) == 2
    assert all("Gurugram" in query for query in queries)


def test_false_without_cited_source_is_downgraded(monkeypatch):
    from factcheck import core
    monkeypatch.setattr(core, "tavily_search", lambda query, api_key, max_results=6: [{
        "source_id": "S1", "title": "One page", "url": "https://example.com/article",
        "snippet": "An unrelated source excerpt", "published_date": "2026-01-01", "score": 0.7
    }])
    monkeypatch.setattr(core, "_groq_json", lambda prompt, api_key, model, timeout=60, **kwargs: {
        "status": "FALSE", "confidence": "High", "verdict": "This is false.",
        "corrected_fact": "Not established", "cited_source_ids": [], "reasoning_checks": [], "limitations": ""
    })
    result = core.verify_claim(
        {"original_claim": "The market grew by 80% in 2025.", "normalized_claim": "The market grew by 80% in 2025.", "page_number": 1},
        tavily_key="test", groq_key="test"
    )
    assert result["status"] == "INSUFFICIENT EVIDENCE"


def test_decisive_verdict_requires_primary_or_independent_sources(monkeypatch):
    from factcheck import core
    calls = {"n": 0}
    def one_secondary_source(query, api_key, max_results=6):
        calls["n"] += 1
        assert max_results == 3
        return [{"source_id": "S1", "title": "One page", "url": "https://example.com/article",
                 "snippet": "The excerpt contradicts the statement.", "published_date": "2026-01-01", "score": 0.7}]
    monkeypatch.setattr(core, "tavily_search", one_secondary_source)
    monkeypatch.setattr(core, "_groq_json", lambda prompt, api_key, model, timeout=60, **kwargs: {
        "status": "FALSE", "confidence": "High", "verdict": "Contradicted.",
        "corrected_fact": "A different value", "cited_source_ids": ["S1"], "reasoning_checks": [], "limitations": ""
    })
    result = core.verify_claim(
        {"original_claim": "The market grew by 80% in 2025.", "normalized_claim": "The market grew by 80% in 2025.", "page_number": 1},
        tavily_key="test", groq_key="test"
    )
    assert calls["n"] == 2
    assert result["status"] == "INSUFFICIENT EVIDENCE"


def test_analyze_pdf_stops_searching_after_groq_rate_limit(monkeypatch):
    from factcheck import core
    claims = [
        {"page_number": 1, "original_claim": f"Claim {i}", "normalized_claim": f"Claim {i}"}
        for i in range(3)
    ]
    calls = []
    progress = []
    monkeypatch.setattr(core, "extract_pdf_pages", lambda pdf_bytes: [{"page_number": 1, "text": "claims"}])
    monkeypatch.setattr(core, "extract_claims_with_ai", lambda *args: claims)

    def rate_limited(claim, *args):
        calls.append(claim)
        return {
            **claim,
            "status": "REVIEW REQUIRED",
            "verdict": "Groq rate limit reached.",
            "_stop_reason": "groq_rate_limit",
        }

    monkeypatch.setattr(core, "verify_claim", rate_limited)
    result = core.analyze_pdf(
        b"pdf",
        "tavily-key",
        "groq-key",
        max_claims=3,
        progress_callback=lambda current, total, message: progress.append((current, total, message)),
    )
    assert len(calls) == 1
    assert len(result["results"]) == 3
    assert result["results"][1]["search_query"] == "Not run"
    assert "not attempted" in result["results"][1]["verdict"].lower()
    assert any("Stopped early" in message for _, _, message in progress)
