import streamlit as st
import requests
import json
import PyPDF2
import os

# Page config
st.set_page_config(
    page_title="FactGuard AI",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0a0a0f; }
.stApp { background: #0a0a0f; color: #e8e8f0; }

.hero {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #0a0a0f 0%, #1a0a2e 50%, #0a0a0f 100%);
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid #2a2a4a;
}

.hero h1 {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7c3aed, #06b6d4, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.hero p {
    color: #8888aa;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

.claim-card {
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
    border-left: 4px solid;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
}

.verified {
    background: #052e16;
    border-color: #22c55e;
    color: #86efac;
}

.inaccurate {
    background: #431407;
    border-color: #f97316;
    color: #fdba74;
}

.false {
    background: #3f0a0a;
    border-color: #ef4444;
    color: #fca5a5;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.badge-verified { background: #22c55e22; color: #22c55e; border: 1px solid #22c55e44; }
.badge-inaccurate { background: #f9731622; color: #f97316; border: 1px solid #f9731644; }
.badge-false { background: #ef444422; color: #ef4444; border: 1px solid #ef444444; }

.stats-box {
    background: #12121f;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.stats-number { font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero">
    <h1>🔍 FactGuard AI</h1>
    <p>Upload a PDF → Extract Claims → Verify Against Live Web Data</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for API keys
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    serper_key = st.text_input("Serper API Key", type="password", placeholder="your-serper-key")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. 📄 Upload PDF\n2. 🤖 AI extracts claims\n3. 🌐 Web verifies each claim\n4. ✅ Get detailed report")
    st.markdown("---")
    st.markdown("### Get Free API Keys")
    st.markdown("🔑 [Gemini API Key](https://aistudio.google.com/apikey) (Free)")
    st.markdown("🔑 [Serper API Key](https://serper.dev) (Free tier)")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_claims_gemini(text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    prompt = f"""Extract all specific factual claims from this text that can be verified. 
Focus on: statistics, dates, numbers, financial figures, scientific facts, historical events.
Return ONLY a JSON array of claims like: [{{"claim": "...", "context": "..."}}]
No explanation, just JSON. No markdown code blocks.

Text: {text[:3000]}"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000}
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

def search_web(claim, serper_key):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
    data = {"q": claim, "num": 3}
    response = requests.post(url, headers=headers, json=data)
    results = response.json()
    snippets = []
    for r in results.get("organic", [])[:3]:
        snippets.append(f"{r.get('title','')}: {r.get('snippet','')}")
    return "\n".join(snippets)

def verify_claim_gemini(claim, context, search_results, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    prompt = f"""Verify this claim using the web search results provided.

Claim: {claim}
Context: {context}

Web Search Results:
{search_results}

Respond ONLY in JSON (no markdown, no code blocks):
{{
  "status": "Verified" or "Inaccurate" or "False",
  "explanation": "brief explanation",
  "correct_fact": "what the correct fact is (if inaccurate/false)"
}}"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500}
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# Main UI
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("📄 Upload PDF Document", type=['pdf'])

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Analyze Document", use_container_width=True, type="primary")

if analyze_btn:
    if not gemini_key or not serper_key:
        st.error("⚠️ Please enter both API keys in the sidebar!")
    elif not uploaded_file:
        st.error("⚠️ Please upload a PDF file!")
    else:
        with st.spinner("📄 Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            st.success(f"✅ Extracted {len(text)} characters from PDF")

        with st.spinner("🤖 AI is identifying claims..."):
            try:
                claims = extract_claims_gemini(text, gemini_key)
                st.success(f"✅ Found {len(claims)} verifiable claims")
            except Exception as e:
                st.error(f"Error extracting claims: {e}")
                claims = []

        if claims:
            results = []
            progress = st.progress(0)
            status_text = st.empty()

            for i, claim_obj in enumerate(claims[:8]):
                status_text.text(f"🌐 Verifying claim {i+1}/{min(len(claims), 8)}...")
                try:
                    search_results = search_web(claim_obj['claim'], serper_key)
                    verification = verify_claim_gemini(
                        claim_obj['claim'],
                        claim_obj.get('context', ''),
                        search_results,
                        gemini_key
                    )
                    results.append({**claim_obj, **verification})
                except Exception as e:
                    results.append({
                        **claim_obj,
                        "status": "False",
                        "explanation": f"Could not verify: {str(e)}",
                        "correct_fact": "N/A"
                    })
                progress.progress((i + 1) / min(len(claims), 8))

            status_text.text("✅ Analysis complete!")

            # Stats
            verified = sum(1 for r in results if r['status'] == 'Verified')
            inaccurate = sum(1 for r in results if r['status'] == 'Inaccurate')
            false = sum(1 for r in results if r['status'] == 'False')

            st.markdown("### 📊 Summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="stats-box"><div class="stats-number" style="color:#22c55e">{verified}</div><div>Verified</div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="stats-box"><div class="stats-number" style="color:#f97316">{inaccurate}</div><div>Inaccurate</div></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="stats-box"><div class="stats-number" style="color:#ef4444">{false}</div><div>False</div></div>""", unsafe_allow_html=True)

            st.markdown("### 📋 Detailed Results")
            for r in results:
                status = r.get('status', 'False')
                css_class = status.lower()
                badge_class = f"badge-{css_class}"
                emoji = "✅" if status == "Verified" else "⚠️" if status == "Inaccurate" else "❌"

                st.markdown(f"""
                <div class="claim-card {css_class}">
                    <span class="badge {badge_class}">{emoji} {status}</span><br>
                    <strong>Claim:</strong> {r['claim']}<br><br>
                    <strong>Finding:</strong> {r.get('explanation', 'N/A')}<br>
                    {"<br><strong>Correct Fact:</strong> " + r.get('correct_fact', '') if r.get('correct_fact') and r.get('correct_fact') != 'N/A' else ''}
                </div>
                """, unsafe_allow_html=True)
