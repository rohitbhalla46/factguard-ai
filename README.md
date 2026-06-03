# FactGuard AI

Streamlit web app for automated PDF fact-checking. It extracts verifiable claims from an uploaded PDF and verifies them using Gemini with Google Search grounding.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## API key

Local:

```bash
set GEMINI_API_KEY=your_key_here
```

Streamlit Cloud:

1. Open your app settings.
2. Go to **Secrets**.
3. Add:

```toml
GEMINI_API_KEY = "your_key_here"
```

You can also paste the key in the app sidebar while testing.

## Deployment

Push this folder to GitHub and deploy it on Streamlit Cloud. Make sure `app.py` and `requirements.txt` are in the repository root or set the Streamlit entry file to `app.py`.

## Deliverables

- Deployed Streamlit app link
- GitHub repository link
- 30-second demo video
