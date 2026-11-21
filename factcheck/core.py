"""Core extraction and evidence verification logic for the Fact-Check Agent.

The module intentionally separates extraction, retrieval and verdict adjudication.
Absence of evidence is never treated as proof that a claim is false.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import pymupdf
import requests

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
LEGACY_GROQ_MODEL = "llama-3.3-70b-versatile"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
ALLOWED_STATUSES = {"VERIFIED", "INACCURATE", "FALSE", "INSUFFICIENT EVIDENCE", "CONFLICTING EVIDENCE", "REVIEW REQUIRED"}


def resolve_groq_model(model: str) -> str:
    """Replace the earlier enterprise-only default with the supported default."""
    if model == LEGACY_GROQ_MODEL:
        return DEFAULT_GROQ_MODEL
    return model


def extract_pdf_pages(pdf_bytes: bytes, max_pages: int = 60, max_chars: int = 50000) -> List[Dict[str, Any]]:
    """Extract text page-by-page. Scanned/image-only pages are reported as empty."""
    if not pdf_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(pdf_bytes) > 20 * 1024 * 1024:
        raise ValueError("PDF is larger than the 20 MB upload limit.")
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("This file could not be opened as a valid PDF.") from exc
    if doc.page_count == 0:
        raise ValueError("The PDF contains no pages.")
    if doc.page_count > max_pages:
        raise ValueError(f"The PDF has {doc.page_count} pages; the current limit is {max_pages} pages.")
    pages: List[Dict[str, Any]] = []
    remaining = max_chars
    for index, page in enumerate(doc):
        text = page.get_text("text").strip()
        if remaining <= 0:
            break
        text = text[:remaining]
        remaining -= len(text)
        pages.append({"page_number": index + 1, "text": text})
    return pages


def fallback_extract_claims(pages: Iterable[Dict[str, Any]], limit: int = 25) -> List[Dict[str, Any]]:
    """Conservative local fallback: extracts numeric/date-bearing sentences for review.

    This does not claim to understand or verify them; it creates candidates only.
    """
    claims: List[Dict[str, Any]] = []
    number_pattern = re.compile(r"(?:\d[\d,.]*\s?(?:%|percent|million|billion|trillion|km|m|kg|\$|₹|USD|INR)?|\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)", re.I)
    for page in pages:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", page.get("text", "")):
            sentence = re.sub(r"\s+", " ", sentence).strip(" •\t")
            if len(sentence) < 24 or not number_pattern.search(sentence):
                continue
            claims.append({
                "page_number": page["page_number"],
                "original_claim": sentence[:650],
                "normalized_claim": sentence[:650],
                "claim_type": "numeric/date candidate",
                "time_reference": "Not inferred",
                "entities": [],
            })
            if len(claims) >= limit:
                return claims
    return claims


def _parse_json_object(content: str) -> Dict[str, Any]:
    if not isinstance(content, str):
        raise RuntimeError("Groq returned a response that was not valid JSON.")
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        result = None
        for index, character in enumerate(content):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(content, index)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                result = candidate
                break
    if not isinstance(result, dict):
        raise RuntimeError("Groq returned a response that was not valid JSON.")
    return result


def _groq_json(prompt: str, api_key: str, model: str, timeout: int = 60) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    request_data = {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return one valid JSON object only. Be conservative, distinguish evidence from inference, and do not invent sources or facts."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(GROQ_CHAT_URL, headers=headers, json=request_data, timeout=timeout)
    if response.status_code == 400:
        try:
            error_data = json.loads(response.text)
        except json.JSONDecodeError:
            error_data = {}
        error = error_data.get("error", {}) if isinstance(error_data, dict) else {}
        if isinstance(error, dict) and error.get("code") == "json_validate_failed":
            request_data.pop("response_format")
            request_data["messages"][1]["content"] = (
                f"{prompt}\n\nOutput-format reminder: return exactly one valid JSON object. "
                "Do not include explanations, markdown fences, or any text outside the JSON object."
            )
            response = requests.post(
                GROQ_CHAT_URL, headers=headers, json=request_data, timeout=timeout
            )
    if response.status_code == 429:
        raise RuntimeError(
            "Groq rate limit reached. Wait for the provider limit to reset or verify fewer claims at once."
        )
    if response.status_code >= 400:
        detail = response.text[:500]
        raise RuntimeError(f"Groq API error ({response.status_code}): {detail}")
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return _parse_json_object(content)


def extract_claims_with_ai(pages: List[Dict[str, Any]], api_key: str,
                           model: str = DEFAULT_GROQ_MODEL, limit: int = 25) -> List[Dict[str, Any]]:
    text = "\n\n".join(f"[PAGE {p['page_number']}]\n{p['text']}" for p in pages)
    prompt = f"""
Extract up to {limit} independently checkable factual claims from the supplied PDF text.
Prioritise statistics, percentages, dates, financial figures, technical specifications, rankings,
quantities, and factual statements about named entities. Ignore opinions, predictions, slogans,
questions, and unsupported recommendations unless they contain a checkable factual assertion.
Preserve exact wording. Do not repair or silently correct claims. Assign the page number shown.
For each claim return: page_number (integer), original_claim (string), normalized_claim (string),
claim_type (string), time_reference (string or 'Not stated'), entities (array of strings).
If no claims are present return {{"claims": []}}.

PDF TEXT:
{text[:50000]}
"""
    payload = _groq_json(prompt, api_key, model)
    claims = payload.get("claims", [])
    out: List[Dict[str, Any]] = []
    for item in claims[:limit]:
        if not isinstance(item, dict) or not str(item.get("original_claim", "")).strip():
            continue
        try:
            page_num = int(item.get("page_number", 1))
        except (ValueError, TypeError):
            page_num = 1
        out.append({
            "page_number": max(1, page_num),
            "original_claim": str(item.get("original_claim", ""))[:650],
            "normalized_claim": str(item.get("normalized_claim", item.get("original_claim", "")))[:650],
            "claim_type": str(item.get("claim_type", "factual claim"))[:80],
            "time_reference": str(item.get("time_reference", "Not stated"))[:100],
            "entities": [str(x)[:80] for x in item.get("entities", [])[:8]],
        })
    return out


def tavily_search(query: str, api_key: str, max_results: int = 6) -> List[Dict[str, Any]]:
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured.")
    response = requests.post(
        TAVILY_SEARCH_URL,
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "topic": "general",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=45,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Live search API error ({response.status_code}): {response.text[:400]}")
    payload = response.json()
    results = []
    for idx, item in enumerate(payload.get("results", []), start=1):
        url = str(item.get("url", ""))
        if not url.startswith(("https://", "http://")):
            continue
        results.append({
            "source_id": f"S{idx}",
            "title": str(item.get("title", "Untitled source"))[:250],
            "url": url,
            "snippet": str(item.get("content", ""))[:900],
            "published_date": str(item.get("published_date", "") or "Not supplied"),
            "score": item.get("score"),
        })
    return results


def _source_quality_hint(url: str) -> str:
    host = re.sub(r"^https?://", "", url.lower()).split("/")[0]
    if host.endswith(".gov") or ".gov." in host or host.endswith(".gov.in"):
        return "Government / public-sector domain"
    if any(x in host for x in ["who.int", "un.org", "worldbank.org", "oecd.org", "nasa.gov", "nist.gov", "ec.europa.eu"]):
        return "Recognised institution / primary information"
    if any(x in host for x in ["nature.com", "science.org", "nejm.org", "thelancet.com", "arxiv.org"]):
        return "Research / academic source (check peer-review status)"
    return "Secondary or publisher source - assess provenance"


def verify_claim(claim: Dict[str, Any], tavily_key: str, groq_key: str,
                 model: str = DEFAULT_GROQ_MODEL) -> Dict[str, Any]:
    """Search the live web, then adjudicate only against retrieved source material."""
    base_query = str(claim.get("normalized_claim") or claim.get("original_claim", "")).strip()
    if not base_query:
        raise ValueError("Claim text is empty.")
    query_exact = f'"{base_query}" official source fact check'
    query_context = f'{base_query} latest official data original publication methodology date'
    # Two complementary live searches reduce dependence on a single search-result ranking.
    raw_sources: List[Dict[str, Any]] = []
    search_errors: List[str] = []
    for q in (query_exact, query_context):
        try:
            raw_sources.extend(tavily_search(q, tavily_key, max_results=3))
        except Exception as exc:
            search_errors.append(str(exc)[:180])
    # Deduplicate by canonicalized URL while preserving the first useful excerpt.
    seen_urls = set()
    enriched = []
    for item in raw_sources:
        url = str(item.get("url", "")).strip()
        normalized_url = url.rstrip("/").lower().split("#", 1)[0]
        if not url.startswith(("https://", "http://")) or normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        src = dict(item)
        src["source_id"] = f"S{len(enriched)+1}"
        src["quality_hint"] = _source_quality_hint(url)
        enriched.append(src)
        if len(enriched) >= 4:
            break
    query = query_exact + "\n" + query_context
    if not enriched:
        return {
            **claim, "status": "INSUFFICIENT EVIDENCE", "confidence": "Low",
            "verdict": "The live searches returned no usable source results. This is not proof that the claim is false.",
            "corrected_fact": "Not established", "search_query": query, "sources": [],
            "limitations": "No source evidence was retrieved. " + "; ".join(search_errors),
        }
    evidence_text = "\n\n".join(
        f"[{s['source_id']}] {s['title']}\nURL: {s['url']}\nSource hint: {s['quality_hint']}\nDate: {s['published_date']}\nEvidence excerpt: {s['snippet']}"
        for s in enriched
    )
    prompt = f"""
You are an evidence adjudicator, not a creative writer. Decide whether the PDF claim is supported by
CURRENT retrieved web evidence. The PDF's statement is the claim to test; the sources below are the
only evidence you may rely on. Do not use prior knowledge to invent a correction.

CLAIM:
Original: {claim.get('original_claim')}
Normalized: {claim.get('normalized_claim')}
Claim type: {claim.get('claim_type')}
Time reference: {claim.get('time_reference')}
PDF page: {claim.get('page_number')}

RETRIEVED SOURCES:
{evidence_text}

Choose exactly one status:
- VERIFIED: credible source evidence materially supports the claim for the same date, definition, unit and geography.
- INACCURATE: evidence shows the statement is outdated, misdated, incomplete, imprecise, or materially misstates a definition/unit/period.
- FALSE: credible evidence directly contradicts the claim. Prefer one authoritative primary source or two independent credible sources. Lack of evidence alone is NEVER FALSE.
- INSUFFICIENT EVIDENCE: sources are missing, weak, irrelevant, or do not establish the claim either way.
- CONFLICTING EVIDENCE: credible sources materially disagree and cannot be reconciled from the excerpts.

Return JSON keys: status, confidence (High/Medium/Low), verdict (2-4 sentences), corrected_fact
(string; say 'Not established' if unsupported), cited_source_ids (array of source IDs only),
reasoning_checks (array; address date, scope, units/denominator where relevant), limitations.
Be conservative. Do not cite a source unless its excerpt actually supports the stated point.
If a claim was historically true but has since become outdated, prefer INACCURATE rather than FALSE.
"""
    try:
        judgment = _groq_json(prompt, groq_key, model)
    except Exception as exc:
        return {
            **claim, "status": "REVIEW REQUIRED", "confidence": "Low",
            "verdict": f"Sources were retrieved, but automated adjudication failed: {str(exc)[:250]}",
            "corrected_fact": "Not established", "search_query": query, "sources": enriched,
            "limitations": "Human review is required; no automated verdict was produced.",
        }
    status = str(judgment.get("status", "INSUFFICIENT EVIDENCE")).upper().strip()
    if status not in ALLOWED_STATUSES:
        status = "INSUFFICIENT EVIDENCE"
    valid_ids = {s["source_id"] for s in enriched}
    cited = [sid for sid in judgment.get("cited_source_ids", []) if sid in valid_ids]
    cited_sources = [s for s in enriched if s["source_id"] in cited]
    # Defensive guard: a supported/contradicted verdict cannot stand without cited retrieved evidence.
    if status in {"VERIFIED", "INACCURATE", "FALSE"} and not cited_sources:
        status = "INSUFFICIENT EVIDENCE"
        judgment["verdict"] = "The model did not tie its conclusion to a retrieved source. Treat this as insufficient evidence."
        judgment["corrected_fact"] = "Not established"
    if status == "FALSE" and len(cited_sources) == 0:
        status = "INSUFFICIENT EVIDENCE"
    # Require either an authoritative source hint or two distinct source domains for a decisive verdict.
    if status in {"VERIFIED", "INACCURATE", "FALSE"}:
        cited_hosts = {re.sub(r"^https?://", "", s["url"].lower()).split("/")[0].removeprefix("www.") for s in cited_sources}
        has_authoritative_hint = any(s.get("quality_hint") != "Secondary or publisher source - assess provenance" for s in cited_sources)
        if not has_authoritative_hint and len(cited_hosts) < 2:
            status = "INSUFFICIENT EVIDENCE"
            judgment["verdict"] = "Only one non-primary source was tied to the verdict. A second independent source or an authoritative primary source is needed."
            judgment["corrected_fact"] = "Not established"
    if search_errors:
        judgment["limitations"] = (str(judgment.get("limitations", "")) + " Partial search error(s): " + "; ".join(search_errors))[:600]
    return {
        **claim,
        "status": status,
        "confidence": str(judgment.get("confidence", "Low")).title(),
        "verdict": str(judgment.get("verdict", "No conclusion supplied."))[:1200],
        "corrected_fact": str(judgment.get("corrected_fact", "Not established"))[:600],
        "search_query": query,
        "sources": enriched,
        "cited_source_ids": cited,
        "reasoning_checks": judgment.get("reasoning_checks", [])[:8],
        "limitations": str(judgment.get("limitations", ""))[:600],
    }


def analyze_pdf(pdf_bytes: bytes, tavily_key: str, groq_key: str,
                model: str = DEFAULT_GROQ_MODEL, max_claims: int = 20,
                use_ai_extraction: bool = True) -> Dict[str, Any]:
    pages = extract_pdf_pages(pdf_bytes)
    if not any(p["text"].strip() for p in pages):
        raise ValueError("No selectable text was found. This appears to be a scanned/image-only PDF; OCR is not enabled in this MVP.")
    if use_ai_extraction:
        claims = extract_claims_with_ai(pages, groq_key, model, max_claims)
    else:
        claims = fallback_extract_claims(pages, max_claims)
    if not claims:
        return {"pages": pages, "claims": [], "results": [], "message": "No factual numeric/date claims were identified."}
    results = []
    for claim in claims:
        try:
            results.append(verify_claim(claim, tavily_key, groq_key, model))
        except Exception as exc:
            results.append({**claim, "status": "REVIEW REQUIRED", "confidence": "Low",
                            "verdict": f"Verification could not be completed: {str(exc)[:250]}",
                            "corrected_fact": "Not established", "search_query": "Not run", "sources": [],
                            "limitations": "Live search or verification service failed."})
    return {"pages": pages, "claims": claims, "results": results}
