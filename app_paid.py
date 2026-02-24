# app.py
# beleza ai chatbot — Upload → Index → Chat (LangChain + Chroma + AWS Bedrock)
# ✅ Login screen (username + password) ALWAYS appears first (even on localhost)
# ✅ Strong tenant isolation via signed URL token + per-tenant workspace hashing
# ✅ Auto-deletes user data via TTL + idle timeout + janitor sweep

import os
import io
import json
import uuid
import time
import shutil
import hmac
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, quote

import streamlit as st

# Load .env locally only (never override Streamlit Cloud secrets)
try:
    from dotenv import load_dotenv
    if Path(".env").exists():
        load_dotenv(override=False)
except Exception:
    pass

# ---- Hard deps (fail with clear message) ----
missing = []
try:
    import pandas as pd
except Exception:
    missing.append("pandas")
try:
    from pypdf import PdfReader
except Exception:
    missing.append("pypdf")
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    missing.append("langchain-text-splitters")
try:
    from langchain_core.documents import Document
except Exception:
    missing.append("langchain-core")
try:
    from langchain_chroma import Chroma
except Exception:
    missing.append("langchain-chroma")
try:
    from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
except Exception:
    missing.append("langchain-aws")

if missing:
    st.error(
        "Missing Python packages:\n\n- " + "\n- ".join(missing) + "\n\nInstall and rerun:\n"
        "pip install -U streamlit python-dotenv pandas pypdf chromadb "
        "langchain langchain-core langchain-text-splitters langchain-chroma langchain-aws boto3"
    )
    st.stop()

# ============================
# CONFIG
# ============================
APP_TITLE = "[Your] AI Chatbot"
BRAND_PINK = "#FE5F9A"

DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2").strip()
BEDROCK_CHAT_MODEL_ID = unquote(os.getenv("BEDROCK_CHAT_MODEL_ID", "us.amazon.nova-pro-v1:0").strip())
BEDROCK_EMBED_MODEL_ID = unquote(os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0").strip())

CHROMA_ROOT = Path(os.getenv("CHROMA_ROOT", "/tmp/chroma_demo"))

MAX_FILES = int(os.getenv("MAX_FILES", "15"))
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "20"))
MAX_TOTAL_CHUNKS = int(os.getenv("MAX_TOTAL_CHUNKS", "2000"))
TOP_K = int(os.getenv("TOP_K", "4"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# Auto-delete
WORKSPACE_TTL_SECONDS = int(os.getenv("WORKSPACE_TTL_SECONDS", "7200"))  # 2 hours
IDLE_TTL_SECONDS = int(os.getenv("IDLE_TTL_SECONDS", "1800"))            # 30 minutes
JANITOR_MAX_DELETE = int(os.getenv("JANITOR_MAX_DELETE", "50"))
META_FILENAME = "_meta.json"

# Tenant signing (set in Secrets)
TENANT_SIGNING_KEY = os.getenv("TENANT_SIGNING_KEY", "").strip()

# Login creds (MUST be set; login is always required)
APP_USERNAME = os.getenv("APP_USERNAME", "").strip()
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()

SYSTEM_PROMPT = """You are an AI helpful assistant.
You MUST answer using ONLY the provided context from the uploaded files.
If the answer is not in the context, say: "I don't know based on the uploaded files."
Be concise and direct.

Return VALID JSON only in this format:
{"answer":"..."}

No markdown. No extra keys. No explanations.
"""

# ============================
# UI / STYLES (KEEP SAME LOOK)
# ============================
def inject_styles():
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&display=swap');

:root {{
  --beleza-pink: {BRAND_PINK};
}}

.gothic-title {{
  font-family: 'UnifrakturCook', cursive;
  font-size: 58px;
  line-height: 1.0;
  text-align: center;
  margin-top: 0.5rem;
  margin-bottom: 0.25rem;
}}

.gothic-sub {{
  font-family: 'UnifrakturCook', cursive;
  font-size: 22px;
  text-align: center;
  margin-bottom: 1rem;
}}

.center-note {{
  text-align: center;
  font-size: 16px;
  color: rgba(0,0,0,0.65);
  margin-bottom: 1.25rem;
}}

.small-foot {{
  text-align: center;
  font-size: 13px;
  color: rgba(0,0,0,0.65);
  margin-top: 1rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  line-height: 1.6;
}}
.small-foot em {{ font-style: italic; }}

/* Hide Streamlit chrome */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Primary button: Beleza pink + Old English font */
button[kind="primary"] {{
  background: var(--beleza-pink) !important;
  border: 1px solid var(--beleza-pink) !important;
  color: white !important;
  font-family: 'UnifrakturCook', cursive !important;
  letter-spacing: 0.2px;
}}
div.stButton > button {{
  border-radius: 12px !important;
  padding: 0.65rem 1rem !important;
}}

/* Text inputs look premium */
div[data-testid="stTextInput"] input {{
  border: 2px solid var(--beleza-pink) !important;
  border-radius: 14px !important;
  box-shadow: none !important;
}}
div[data-testid="stTextInput"] input:focus {{
  border: 2px solid var(--beleza-pink) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(254,95,154,0.18) !important;
}}

/* --- Chat input: pink outline + pink send button --- */
div[data-testid="stChatInput"] textarea {{
  border: 2px solid var(--beleza-pink) !important;
  border-radius: 14px !important;
  box-shadow: none !important;
}}
div[data-testid="stChatInput"] textarea:focus {{
  border: 2px solid var(--beleza-pink) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(254,95,154,0.18) !important;
}}
div[data-testid="stChatInput"] button {{
  border-radius: 12px !important;
  border: 1px solid var(--beleza-pink) !important;
  background: var(--beleza-pink) !important;
}}
div[data-testid="stChatInput"] button svg {{
  fill: white !important;
  stroke: white !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )

# ============================
# LOGIN GATE (ALWAYS ON)
# ============================
def login_gate() -> None:
    """
    Always require login. If creds aren't set, show a helpful error.
    """
    if st.session_state.get("authed") is True:
        return

    if not (APP_USERNAME and APP_PASSWORD):
        st.markdown('<div class="gothic-title">beleza ai chatbot</div>', unsafe_allow_html=True)
        st.markdown('<div class="center-note">private intelligence for your documents</div>', unsafe_allow_html=True)
        st.error("Missing APP_USERNAME / APP_PASSWORD. Set them in .env (local) or Streamlit Secrets (cloud).")
        st.stop()

    st.markdown('<div class="gothic-title">beleza ai chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="center-note">private intelligence for your documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="gothic-sub">login</div>', unsafe_allow_html=True)

    u = st.text_input("username", label_visibility="collapsed", placeholder="username")
    p = st.text_input("password", type="password", label_visibility="collapsed", placeholder="password")

    c1, c2 = st.columns([2, 1])
    with c1:
        go = st.button("enter", type="primary", use_container_width=True)
    with c2:
        if st.button("reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if go:
        if u == APP_USERNAME and p == APP_PASSWORD:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("invalid username or password")

    st.stop()

# ============================
# TENANT ISOLATION
# ============================
def _b(x: str) -> bytes:
    return (x or "").encode("utf-8")

def sign_token(token: str) -> str:
    key = TENANT_SIGNING_KEY or "insecure-dev-key-change-me"
    return hmac.new(_b(key), _b(token), hashlib.sha256).hexdigest()

def verify_token(token: str, sig: str) -> bool:
    if not token or not sig:
        return False
    expected = sign_token(token)
    return hmac.compare_digest(expected, sig)

def workspace_id_from_token(token: str) -> str:
    return hashlib.sha256(_b(token)).hexdigest()

def set_query_params(token: str, sig: str) -> None:
    try:
        st.query_params["t"] = token
        st.query_params["sig"] = sig
    except Exception:
        try:
            st.experimental_set_query_params(t=token, sig=sig)
        except Exception:
            pass

def get_query_params() -> Dict[str, str]:
    try:
        qp = st.query_params
        return {"t": qp.get("t", ""), "sig": qp.get("sig", "")}
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            return {"t": (qp.get("t", [""]) or [""])[0], "sig": (qp.get("sig", [""]) or [""])[0]}
        except Exception:
            return {"t": "", "sig": ""}

def ensure_tenant_context() -> str:
    qp = get_query_params()
    t = (qp.get("t") or "").strip()
    sig = (qp.get("sig") or "").strip()

    if not verify_token(t, sig):
        t = uuid.uuid4().hex
        sig = sign_token(t)
        set_query_params(t, sig)

    ws = workspace_id_from_token(t)
    st.session_state["workspace_id"] = ws
    return ws

# ============================
# AVATAR (PINK ROBOT)
# ============================
def assistant_avatar_data_uri(color_hex: str) -> str:
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
      <rect x="14" y="18" width="36" height="32" rx="10" fill="{color_hex}"/>
      <rect x="26" y="10" width="12" height="10" rx="4" fill="{color_hex}"/>
      <circle cx="26" cy="34" r="4" fill="white"/>
      <circle cx="38" cy="34" r="4" fill="white"/>
      <rect x="24" y="42" width="16" height="4" rx="2" fill="white" opacity="0.95"/>
      <circle cx="32" cy="8" r="3" fill="{color_hex}"/>
      <rect x="31" y="8" width="2" height="6" fill="{color_hex}"/>
    </svg>
    """.strip()
    return "data:image/svg+xml;utf8," + quote(svg)

ASSISTANT_AVATAR = assistant_avatar_data_uri(BRAND_PINK)

# ============================
# BEDROCK HELPERS
# ============================
def bedrock_llm(model_id: str):
    return ChatBedrockConverse(
        model=model_id,
        region_name=DEFAULT_REGION,
        temperature=0.2,
        max_tokens=600,
    )

def bedrock_embeddings():
    return BedrockEmbeddings(
        region_name=DEFAULT_REGION,
        model_id=BEDROCK_EMBED_MODEL_ID,
    )

def _extract_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "content"):
        return _extract_text(obj.content)
    if isinstance(obj, list):
        return "".join(x.get("text", "") for x in obj if isinstance(x, dict))
    return ""

def generate_streaming(llm, prompt: str) -> Tuple[str, Optional[float], float]:
    t0 = time.time()
    ttft = None
    parts: List[str] = []
    try:
        for chunk in llm.stream(prompt):
            txt = _extract_text(chunk)
            if txt:
                if ttft is None:
                    ttft = time.time() - t0
                parts.append(txt)
        return "".join(parts), ttft, round(time.time() - t0, 3)
    except Exception:
        resp = llm.invoke(prompt)
        return _extract_text(resp), None, round(time.time() - t0, 3)

# ============================
# FILE PARSING
# ============================
def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for p in reader.pages:
        texts.append(p.extract_text() or "")
    return "\n".join(texts).strip()

def read_csv(file_bytes: bytes) -> str:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.to_csv(index=False)

def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()

def file_to_text(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()
    b = uploaded_file.read()
    if name.endswith(".pdf"):
        return read_pdf(b)
    if name.endswith(".csv"):
        return read_csv(b)
    return read_txt(b)

def validate_uploads(files) -> Tuple[bool, str]:
    if not files:
        return False, "Please upload at least one file."
    if len(files) > MAX_FILES:
        return False, f"Too many files. Max is {MAX_FILES}."
    for f in files:
        size_mb = (getattr(f, "size", 0) or 0) / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            return False, f"File '{f.name}' is too large ({size_mb:.1f} MB). Max is {MAX_FILE_MB} MB."
    return True, ""

# ============================
# WORKSPACE META + JANITOR
# ============================
def _now() -> int:
    return int(time.time())

def read_workspace_meta(d: Path) -> Dict[str, Any]:
    p = d / META_FILENAME
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def write_workspace_meta(d: Path, created_at: int, last_activity: int) -> None:
    p = d / META_FILENAME
    meta = {"created_at": int(created_at), "last_activity": int(last_activity)}
    try:
        p.write_text(json.dumps(meta), encoding="utf-8")
    except Exception:
        pass

def janitor_sweep() -> int:
    if not CHROMA_ROOT.exists():
        return 0
    now = _now()
    deleted = 0
    for d in CHROMA_ROOT.iterdir():
        if not d.is_dir():
            continue
        meta = read_workspace_meta(d)
        created_at = int(meta.get("created_at") or 0)
        last_activity = int(meta.get("last_activity") or 0)
        mtime = int(d.stat().st_mtime)
        if created_at <= 0:
            created_at = mtime
        if last_activity <= 0:
            last_activity = mtime

        too_old = (now - created_at) > WORKSPACE_TTL_SECONDS
        too_idle = (now - last_activity) > IDLE_TTL_SECONDS
        if too_old or too_idle:
            try:
                shutil.rmtree(d)
                deleted += 1
            except Exception:
                pass
        if deleted >= JANITOR_MAX_DELETE:
            break
    return deleted

def touch_activity(workspace_id: str) -> None:
    now = _now()
    if "created_at" not in st.session_state:
        st.session_state.created_at = now
    st.session_state.last_activity = now

    d = CHROMA_ROOT / workspace_id
    if d.exists():
        write_workspace_meta(d, st.session_state.created_at, st.session_state.last_activity)

# ============================
# VECTOR STORE
# ============================
def workspace_dir(workspace_id: str) -> Path:
    d = CHROMA_ROOT / workspace_id
    d.mkdir(parents=True, exist_ok=True)
    created_at = int(st.session_state.get("created_at") or _now())
    last_activity = int(st.session_state.get("last_activity") or _now())
    write_workspace_meta(d, created_at, last_activity)
    return d

def get_vectordb(workspace_id: str) -> Chroma:
    d = workspace_dir(workspace_id)
    collection = f"demo_docs_{workspace_id[:12]}"
    return Chroma(
        collection_name=collection,
        persist_directory=str(d),
        embedding_function=bedrock_embeddings(),
    )

def reset_workspace() -> None:
    t = uuid.uuid4().hex
    sig = sign_token(t)
    set_query_params(t, sig)

    st.session_state.indexed = False
    st.session_state.last_index_count = 0
    st.session_state.messages = []
    st.session_state.created_at = _now()
    st.session_state.last_activity = _now()

def index_files(workspace_id: str, files) -> int:
    vectordb = get_vectordb(workspace_id)
    docs: List[Document] = []

    for f in files:
        text = file_to_text(f)
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source": f.name, "chunk": i}))
            if len(docs) >= MAX_TOTAL_CHUNKS:
                break
        if len(docs) >= MAX_TOTAL_CHUNKS:
            break

    if docs:
        vectordb.add_documents(docs)

    st.session_state.indexed = True
    st.session_state.last_index_count = len(docs)
    touch_activity(workspace_id)
    return len(docs)

# ============================
# RAG ANSWER
# ============================
def safe_json_load(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {"answer": s.strip()}

def rag_answer(workspace_id: str, question: str) -> str:
    vectordb = get_vectordb(workspace_id)
    docs = vectordb.similarity_search(question, k=TOP_K)

    context = "\n\n".join(
        f"[{d.metadata.get('source','unknown')} #{d.metadata.get('chunk',0)}] {d.page_content}"
        for d in docs
    )

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}
"""
    llm = bedrock_llm(BEDROCK_CHAT_MODEL_ID)
    raw, _, _ = generate_streaming(llm, prompt)
    obj = safe_json_load(raw)
    answer = (obj.get("answer") or "").strip() or "I don't know based on the uploaded files."
    touch_activity(workspace_id)
    return answer

# ============================
# STREAMLIT APP
# ============================
def main():
    janitor_sweep()
    st.set_page_config(APP_TITLE, layout="centered")
    inject_styles()

    # ✅ ALWAYS show login screen first
    login_gate()

    # Then tenant context + app
    workspace_id = ensure_tenant_context()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "created_at" not in st.session_state:
        st.session_state.created_at = _now()
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = _now()

    touch_activity(workspace_id)

    st.markdown('<div class="gothic-title">beleza ai chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="center-note">private intelligence for your documents</div>', unsafe_allow_html=True)

    if not st.session_state.indexed:
        st.markdown(
            '<div class="gothic-sub">upload your private files, and i can help answer questions for you</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Upload PDF, TXT, or CSV",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            index_clicked = st.button("Upload & Prepare", type="primary", use_container_width=True)
        with c2:
            if st.button("Reset", use_container_width=True):
                reset_workspace()
                st.rerun()

        if index_clicked:
            ok, msg = validate_uploads(uploaded)
            if not ok:
                st.error(msg)
            else:
                with st.spinner("Preparing your files..."):
                    n = index_files(workspace_id, uploaded)
                if n == 0:
                    st.warning("I couldn’t extract any text from those files. Try a text-based PDF, TXT, or CSV.")
                else:
                    st.success("Ready. Ask your questions below.")
                    st.rerun()

        st.markdown(
            f"""
<div class="small-foot">
<em>your files are automatically deleted after {IDLE_TTL_SECONDS//60} min idle or {WORKSPACE_TTL_SECONDS//3600} hrs • limits: {MAX_FILES} files • {MAX_FILE_MB}mb each</em><br/>
<strong>contact@belezamedia.org</strong> for your private custom chatbot
</div>
""",
            unsafe_allow_html=True,
        )
        return

    top = st.columns([3, 1])
    with top[1]:
        if st.button("Reset / New Upload", use_container_width=True):
            reset_workspace()
            st.rerun()

    for m in st.session_state.messages:
        avatar = ASSISTANT_AVATAR if m["role"] == "assistant" else None
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    q = st.chat_input("Ask a question about your uploaded files…")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Thinking..."):
                answer = rag_answer(workspace_id, q)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()