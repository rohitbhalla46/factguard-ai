import json
import os
import re

import PyPDF2
import streamlit as st
from google import genai
from google.genai import types


EXTRACTION_MODELS = ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]
GROUNDING_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash"]
ALLOWED_STATUSES = {"Verified", "Inaccurate", "False"}
CHECKABLE_VALUE_PATTERN = re.compile(
    r"(\d|%|\$|billion|million|trillion|crore|lakh|revenue|users|market share|growth|founded|launched|released|population|valuation)",
    re.IGNORECASE,
)
INSTRUCTION_PATTERN = re.compile(
    r"(build|design|define|submit|submission|deliverable|upload|interface|deployment|mandatory|"
    r"requirements?|criteria|roadmap|task|objective|ppt|github|demo video|streamlit|gradio|react|vercel|render)",
    re.IGNORECASE,
)


st.set_page_config(page_title="FactGuard AI", page_icon="FG", layout="wide")

st.markdown(
    """
<style>
* { font-family: Inter, Segoe UI, sans-serif; }
.stApp { background: #0b0f14; color: #eef3f8; }
.hero {
    padding: 1.25rem 0 1.75rem;
    border-bottom: 1px solid #23303d;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 2.45rem;
    font-weight: 750;
    margin: 0;
    letter-spacing: 0;
}
.hero p {
    color: #aab7c4;
    font-size: 1rem;
    margin-top: .35rem;
}
.claim-card {
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin: .75rem 0;
    border: 1px solid #2b3948;
    background: #111820;
}
.status {
    display: inline-block;
    padding: .18rem .55rem;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 700;
    margin-bottom: .55rem;
}
.verified { color: #65d98b; border-color: #245b39; }
.inaccurate { color: #f2b35d; border-color: #795324; }
.false { color: #ff7979; border-color: #793030; }
.source-list a { color: #8ed0ff; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>FactGuard AI</h1>
  <p>Upload a PDF, extract verifiable claims, and check them against live Google Search grounding.</p>
</div>
""",
    unsafe_allow_html=True,
)


def get_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or os.getenv(name, "")


def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def is_retryable_model_error(error_text: str) -> bool:
    retry_phrases = (
        "RESOURCE_EXHAUSTED",
        "429",
        "not found",
        "not supported",
        "unsupported",
        "INVALID_ARGUMENT",
        "400",
    )
    return any(phrase.lower() in error_text.lower() for phrase in retry_phrases)


def is_quota_error(error_text: str) -> bool:
    return "RESOURCE_EXHAUSTED" in error_text or "429" in error_text or "quota" in error_text.lower()


def generate_content_with_fallback(client: genai.Client, contents: str, config, models: list[str]):
    last_error = None
    for model_name in models:
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last_error = exc
            if not is_retryable_model_error(str(exc)):
                raise
    raise RuntimeError(
        "Gemini quota/model limit hit. The app will keep working with a safe fallback, "
        "but live verification may need a fresh API key, a new Google AI Studio project, or billing."
    ) from last_error


def extract_text_from_pdf(pdf_file) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def parse_json(raw: str):
    raw = (raw or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def is_checkable_claim(claim: str) -> bool:
    claim = (claim or "").strip()
    if not claim:
        return False
    if INSTRUCTION_PATTERN.search(claim):
        return False
    return bool(CHECKABLE_VALUE_PATTERN.search(claim))


def regex_extract_claims(text: str, limit: int = 8) -> list[dict]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    claims = []
    for sentence in sentences:
        sentence = " ".join(sentence.split())
        if len(sentence) < 25:
            continue
        if not is_checkable_claim(sentence):
            continue
        claims.append(
            {
                "claim": sentence[:500],
                "context": sentence[:500],
                "why_checkable": "Contains a date, number, percentage, money amount, or measurable factual value.",
            }
        )
        if len(claims) >= limit:
            break
    return claims


def extract_claims(text: str, api_key: str, max_claims: int) -> tuple[list[dict], str | None]:
    client = get_client(api_key)
    prompt = f"""
Extract specific factual claims from this PDF text.

Focus only on claims that can be checked externally:
- statistics and percentages
- dates and timelines
- market, financial, technical, or scientific figures
- named factual statements

Do not extract:
- goals, tasks, objectives, roadmap items, or submission instructions
- recommendations or opinions
- generic statements without a concrete factual value

Return only a JSON array. Each item must use this schema:
{{"claim": "...", "context": "...", "why_checkable": "..."}}

PDF text:
{text[:12000]}
"""
    try:
        response = generate_content_with_fallback(
            client=client,
            contents=prompt,
            models=EXTRACTION_MODELS,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        claims = parse_json(response.text or "[]")
        if not isinstance(claims, list):
            claims = []
        claims = [claim for claim in claims if is_checkable_claim(claim.get("claim", ""))]
        return claims[:max_claims], None
    except Exception as exc:
        fallback_claims = regex_extract_claims(text, max_claims)
        message = "AI extraction was unavailable, so the app used a local claim extractor."
        if is_quota_error(str(exc)):
            message = "Gemini quota is currently exhausted, so the app used a local claim extractor."
        return fallback_claims, message


def grounding_sources(response) -> list[dict]:
    sources = []
    try:
        metadata = response.candidates[0].grounding_metadata
        chunks = metadata.grounding_chunks or []
    except Exception:
        return sources

    for chunk in chunks:
        web = getattr(chunk, "web", None)
        if not web:
            continue
        title = getattr(web, "title", "") or "Source"
        uri = getattr(web, "uri", "") or ""
        if uri and not any(item["uri"] == uri for item in sources):
            sources.append({"title": title, "uri": uri})
    return sources[:5]


def verify_claims(claims: list[dict], api_key: str) -> tuple[list[dict], list[dict], str | None]:
    client = get_client(api_key)
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    claim_payload = [
        {
            "id": index,
            "claim": claim.get("claim", ""),
            "context": claim.get("context", ""),
        }
        for index, claim in enumerate(claims, start=1)
    ]

    prompt = f"""
Use live Google Search grounding to verify all claims below.

Claims:
{json.dumps(claim_payload, indent=2)}

Classify each claim as:
- Verified: reliable current sources support the claim.
- Inaccurate: the claim is partly true, outdated, or the number/date is wrong.
- False: no reliable evidence supports it, or reliable sources contradict it.

Return only a JSON array. Each item must use this schema:
{{
  "id": 1,
  "status": "Verified" | "Inaccurate" | "False",
  "explanation": "short reason based on the sources",
  "correct_fact": "correct current fact if available, otherwise N/A"
}}
"""
    try:
        response = generate_content_with_fallback(
            client=client,
            contents=prompt,
            models=GROUNDING_MODELS,
            config=types.GenerateContentConfig(
                tools=[grounding_tool],
                temperature=0.0,
            ),
        )
        results = parse_json(response.text or "[]")
        if not isinstance(results, list):
            results = []
        return results, grounding_sources(response), None
    except Exception as exc:
        message = "Live verification was unavailable. Showing extracted claims with a safe fallback status."
        if is_quota_error(str(exc)):
            message = "Gemini quota is currently exhausted. Showing extracted claims with a safe fallback status."
        fallback_results = [
            {
                "id": index,
                "status": "False",
                "explanation": message,
                "correct_fact": "Retry with available Gemini quota for the real fact.",
            }
            for index, _ in enumerate(claims, start=1)
        ]
        return fallback_results, [], message


def normalize_status(status: str) -> str:
    status = (status or "").strip()
    return status if status in ALLOWED_STATUSES else "False"


def render_result(result: dict) -> None:
    status = normalize_status(result.get("status", "False"))
    css_class = status.lower()
    sources = result.get("sources", [])
    source_html = ""
    if sources:
        links = "".join(
            f'<li><a href="{s["uri"]}" target="_blank">{s["title"]}</a></li>'
            for s in sources
        )
        source_html = f'<div class="source-list"><strong>Sources:</strong><ul>{links}</ul></div>'

    st.markdown(
        f"""
<div class="claim-card {css_class}">
  <span class="status {css_class}">{status}</span><br>
  <strong>Claim:</strong> {result.get("claim", "N/A")}<br><br>
  <strong>Finding:</strong> {result.get("explanation", "N/A")}<br>
  <strong>Correct fact:</strong> {result.get("correct_fact", "N/A")}
  {source_html}
</div>
""",
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.subheader("Configuration")
    default_key = get_secret("GEMINI_API_KEY")
    gemini_key = st.text_input(
        "Gemini API Key",
        value=default_key,
        type="password",
        placeholder="Set GEMINI_API_KEY in Streamlit secrets",
    )
    max_claims = st.slider("Claims to verify", min_value=3, max_value=8, value=5)

uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])
analyze_btn = st.button("Analyze document", type="primary", use_container_width=True)

if analyze_btn:
    if not gemini_key:
        st.error("Add your Gemini key in the sidebar or Streamlit Cloud secrets as GEMINI_API_KEY.")
        st.stop()
    if not uploaded_file:
        st.error("Upload a PDF first.")
        st.stop()

    with st.spinner("Extracting text from PDF..."):
        try:
            text = extract_text_from_pdf(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read this PDF: {exc}")
            st.stop()

    if not text:
        st.error("No readable text found in this PDF. Try a text-based PDF instead of a scanned image.")
        st.stop()
    st.success(f"Extracted {len(text):,} characters.")

    with st.spinner("Extracting checkable claims..."):
        claims, extraction_warning = extract_claims(text, gemini_key, max_claims)
    if extraction_warning:
        st.warning(extraction_warning)
    st.success(f"Found {len(claims)} claims to verify.")

    if not claims:
        st.info("No checkable claims were found.")
        st.stop()

    progress = st.progress(0)
    with st.spinner("Verifying claims against live web data..."):
        verifications, shared_sources, verification_warning = verify_claims(claims, gemini_key)
    progress.progress(1.0)
    if verification_warning:
        st.warning(verification_warning)

    verification_by_id = {
        item.get("id"): item for item in verifications if isinstance(item, dict)
    }
    results = []
    for index, claim_obj in enumerate(claims, start=1):
        verification = verification_by_id.get(index, {})
        if not verification:
            verification = {
                "status": "False",
                "explanation": "No verification result was returned for this claim.",
                "correct_fact": "N/A",
            }
        verification.pop("id", None)
        verification["status"] = normalize_status(verification.get("status", "False"))
        results.append({**claim_obj, **verification, "sources": shared_sources})

    verified = sum(1 for item in results if item.get("status") == "Verified")
    inaccurate = sum(1 for item in results if item.get("status") == "Inaccurate")
    false = sum(1 for item in results if item.get("status") == "False")

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Verified", verified)
    col2.metric("Inaccurate", inaccurate)
    col3.metric("False", false)

    st.subheader("Detailed Results")
    for result in results:
        render_result(result)

    st.download_button(
        "Download JSON report",
        data=json.dumps(results, indent=2),
        file_name="factguard_report.json",
        mime="application/json",
    )
