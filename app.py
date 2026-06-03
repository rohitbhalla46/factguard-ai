import json
import os
import re

import PyPDF2
import streamlit as st
from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.5-flash"


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

Return only a JSON array. Each item must use this schema:
{{"claim": "...", "context": "...", "why_checkable": "..."}}

PDF text:
{text[:12000]}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    claims = parse_json(response.text or "[]")
    return claims if isinstance(claims, list) else []


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


def verify_claim(claim: str, context: str, api_key: str) -> dict:
    client = get_client(api_key)
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    prompt = f"""
Use live Google Search grounding to verify the claim below.

Claim: {claim}
Context from PDF: {context}

Classify the claim as:
- Verified: reliable current sources support the claim.
- Inaccurate: the claim is partly true, outdated, or the number/date is wrong.
- False: no reliable evidence supports it, or reliable sources contradict it.

Return only JSON:
{{
  "status": "Verified" | "Inaccurate" | "False",
  "explanation": "short reason based on the sources",
  "correct_fact": "correct current fact if available, otherwise N/A"
}}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.0,
        ),
    )

    result = parse_json(response.text or "{}")
    if not isinstance(result, dict):
        result = {
            "status": "False",
            "explanation": "Gemini returned an unexpected verification format.",
            "correct_fact": "N/A",
        }
    result["sources"] = grounding_sources(response)
    return result


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
    max_claims = st.slider("Claims to verify", min_value=3, max_value=12, value=8)

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

        results = []
        progress = st.progress(0)
        for index, claim_obj in enumerate(claims, start=1):
            with st.spinner(f"Verifying claim {index}/{len(claims)}..."):
                claim = claim_obj.get("claim", "")
                verification = verify_claim(claim, claim_obj.get("context", ""), gemini_key)
                results.append({**claim_obj, **verification})
            progress.progress(index / len(claims))

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
