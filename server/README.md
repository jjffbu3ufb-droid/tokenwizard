# LLMLingua-integrated Chat (FastAPI)

This adds **prompt compression** with LLMLingua / LLMLingua-2 to your existing chat UI.

## Quick start

```bash
cd server
python -m venv .venv && source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt

# Set your key (bash)
export OPENAI_API_KEY=sk-...

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `index.html` in your browser. It will POST to `http://localhost:8000/api/chat`.
