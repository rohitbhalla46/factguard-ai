import json
import os
import re

import PyPDF2
import streamlit as st
from google import genai
from google.genai import types


MODEL_NAMES = ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]
ALLOWED_STATUSES = {"Verified", "Inaccurate", "False"}
CHECKABLE_VALUE_PATTERN = re.compile(
    r"(\d|%|\$|₹|€|£|billion|million|trillion|crore|lakh|revenue|users|market share|growth|founded|launched)",
    re.IGNORECASE,
)
INSTRUCTION_PATTERN = re.compile(
    r"(build|design|define|submit|submission|deliverable|upload|interface|deployment|mandatory|"
    r"requirements?|criteria|roadmap|task|objective|ppt|github|demo video|streamlit|gradio|react|vercel|render)",
    re.IGNORECASE,
)


st.set_page_config(
    page_title="FactGuard AI",
    page_icon="FG",
    layout="wide",
)

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


def generate_content_with_fallback(client: genai.Client, contents: str, config):
    last_error = None
    for model_name in MODEL_NAMES:
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last_error = exc
            error_text = str(exc)
            if "RESOURCE_EXHAUSTED" not in error_text and "429" not in error_text:
                raise
    raise RuntimeError(
        "Gemini free quota is exhausted for today. Please wait for the daily reset, "
        "use another Gemini API key/project, or enable billing for higher limits."
    ) from last_error


def extract_text_from_pdf(pdf_file) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def parse_json(raw: str):
    raw = raw.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def extract_claims(text: str, api_key: str) -> list[dict]:
    client = get_client(api_key)
    prompt = f"""
Extract specific factual claims from this PDF text.

Focus only on claims that can be checked externally:
- statistics and percentages
- dates and timelines
- market, financial, technical, or scientific figures
- named factual statements

Do not extract:
- goals, tasks, objectives, or roadmap items
- recommendations or opinions
- generic statements without a concrete factual value

Return only a JSON array. Each item must use this schema:
{{"claim": "...", "context": "...", "why_checkable": "..."}}

PDF text:
{text[:12000]}
"""
    response = generate_content_with_fallback(
        client=client,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    claims = parse_json(response.text or "[]")
    if not isinstance(claims, list):
        return []
    return [claim for claim in claims if is_checkable_claim(claim.get("claim", ""))]


def is_checkable_claim(claim: str) -> bool:
    claim = (claim or "").strip()
    if not claim:
        return False
    if INSTRUCTION_PATTERN.search(claim):
        return False
    return bool(CHECKABLE_VALUE_PATTERN.search(claim))


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


def verify_claims(claims: list[dict], api_key: str) -> tuple[list[dict], list[dict]]:
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

If a provided item is not an externally verifiable factual claim, mark it as False and explain that it is not checkable evidence.

Return only a JSON array. Each item must use this schema:
{{
  "id": 1,
  "status": "Verified" | "Inaccurate" | "False",
  "explanation": "short reason based on the sources",
  "correct_fact": "correct current fact if available, otherwise N/A"
}}
"""
    response = generate_content_with_fallback(
        client=client,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.0,
        ),
    )

    results = parse_json(response.text or "[]")
    if not isinstance(results, list):
        results = []
    return results, grounding_sources(response)


def normalize_status(status: str) -> str:
    status = (status or "").strip()
    return status if status in ALLOWED_STATUSES else "False"


def render_result(result: dict) -> None:
    status = result.get("status", "False")
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

    try:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        if not text:
            st.error("No readable text found in this PDF. Try a text-based PDF instead of a scanned image.")
            st.stop()
        st.success(f"Extracted {len(text):,} characters.")

        with st.spinner("Extracting checkable claims..."):
            claims = extract_claims(text, gemini_key)
        claims = claims[:max_claims]
        st.success(f"Found {len(claims)} claims to verify.")

        if not claims:
            st.info("No checkable claims were found.")
            st.stop()

        progress = st.progress(0)
        with st.spinner("Verifying claims against live web data..."):
            verifications, shared_sources = verify_claims(claims, gemini_key)
        progress.progress(1.0)

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
    except Exception as exc:
        st.exception(exc)
