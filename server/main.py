#ghgjghgjgh
import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

_compressor = None

def get_compressor(use_llmlingua2: bool = True, model_name: Optional[str] = None):
    global _compressor
    if _compressor is None:
        from llmlingua import PromptCompressor
        kwargs = {}
        if use_llmlingua2:
            kwargs['use_llmlingua2'] = True
            if model_name:
                kwargs['model_name'] = model_name
        _compressor = PromptCompressor(**kwargs)
    return _compressor

class Message(BaseModel):
    role: str
    content: str

class ChatBody(BaseModel):
    model: str = 'gpt-4o-mini'
    messages: List[Message]
    compress: bool = True
    rate: float = 0.5
    min_chars: int = 400
    use_llmlingua2: bool = True
    llmlingua2_model: Optional[str] = None
    debug: bool = False

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')

app = FastAPI(title='LLMLingua Chat Proxy', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Serve the repo root (which contains index.html) at "/"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

def pack_for_llmlingua(messages: List[Dict[str, str]]):
    """Split chat into instruction (system), prompt (prior context), question (latest user)."""
    system = ''
    context_lines = []
    last_user = ''

    for i, m in enumerate(messages):
        role = m.get('role', 'user')
        content = m.get('content', '')
        if role == 'system':
            system += content + '\n'
        elif role == 'user':
            last_user = content
        else:
            context_lines.append(f"{role}: {content}")

    # Use all but the last user turn for context
    for i, m in enumerate(messages[:-1] if messages else []):
        role = m.get('role', 'user')
        content = m.get('content', '')
        if role == 'system':
            continue
        context_lines.append(f"{role}: {content}")

    prompt = '\n'.join(context_lines).strip()
    question = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else last_user
    return system.strip(), prompt, question or ''

api = FastAPI(title="API")
@api.post("/chat")
def chat(body: ChatBody):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail='OPENAI_API_KEY is not set on the server.')

    msgs = [m.dict() for m in body.messages]

    system, prompt, question = pack_for_llmlingua(msgs)
    raw_len = len(system) + len(prompt) + len(question)
    do_compress = body.compress and raw_len >= body.min_chars

    compression_info = None
    chat_messages = None

    if do_compress:
        comp = get_compressor(body.use_llmlingua2, body.llmlingua2_model)
        rate = max(0.1, min(1.0, body.rate))
        result = comp.compress_prompt(
            prompt=prompt,
            instruction=system,
            question=question,
            rate=rate,
            force_tokens=['\n', '?', ':', '.', ',']
        )
        compressed_text = result.get('compressed_prompt', '')
        origin_tokens = int(result.get('origin_tokens', 0) or 0)
        compressed_tokens = int(result.get('compressed_tokens', 0) or 0)
        ratio = result.get('ratio', '')
        compression_info = f"{compressed_tokens}/{origin_tokens} tokens ({ratio})"
        chat_messages = [{ 'role': 'user', 'content': compressed_text }]
    else:
        chat_messages = msgs

    payload = {
        'model': body.model,
        'messages': chat_messages,
        'temperature': 0.7
    }
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    r = requests.post(f"{OPENAI_API_BASE}/chat/completions", headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    reply = (data.get('choices') or [{}])[0].get('message', {}).get('content', '')

    resp = {
        'reply': reply,
        'compressed': bool(do_compress),
        'compression_info': compression_info,
    }
    if body.debug and do_compress:
        resp['compressed_prompt'] = compressed_text
    return resp

app = FastAPI(title="Root")
app.mount("/api", api)  # ✅ mount API first j
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")