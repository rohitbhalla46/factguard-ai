# FactGuard AI - Automated Fact Checking Tool

A web app that automatically verifies claims in PDF documents using AI and live web search.

## Features
- Upload any PDF document
- AI extracts all verifiable claims (stats, dates, figures)
- Cross-references each claim against live web data
- Flags claims as: Verified ✅ | Inaccurate ⚠️ | False ❌

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Enter API Keys in sidebar
- **Anthropic API Key** - Get from console.anthropic.com
- **Serper API Key** - Get from serper.dev

## Deployment (Streamlit Cloud)
1. Push this repo to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Deploy!

## Tech Stack
- Frontend: Streamlit
- AI: Claude (Anthropic API)
- Web Search: Serper API
- PDF Parsing: PyPDF2
