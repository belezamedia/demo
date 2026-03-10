# app_PAID_DEV.py
# DocHelp.AI — Upload → Index → Chat (LangChain + Chroma + AWS Bedrock)
# ✅ Login screen (username + password) ALWAYS appears first
# ✅ Strong tenant isolation via signed URL token + per-tenant workspace hashing
# ✅ Auto-deletes user data via TTL + idle timeout + janitor sweep
# ✅ Optimized for Streamlit Cloud: cached Bedrock clients, cached splitter, guarded config
# ✅ Higher upload/file limits for demo usage
# ✅ Lower hallucination risk with stricter grounding + business tone

import os
import io
import re
import json
import uuid
import time
import shutil
import hmac
import hashlib
import configparser
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import unquote, quote
from html.parser import HTMLParser
from email import policy
from email.parser import BytesParser
import xml.etree.ElementTree as ET

import streamlit as st

# Load .env locally only (never override Streamlit Cloud secrets)
try:
    from dotenv import load_dotenv
    if Path(".env").exists():
        load_dotenv(override=False)
except Exception:
    pass

# Core deps
import pandas as pd
import yaml
from pypdf import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

# Optional RTF parser
try:
    from striprtf.striprtf import rtf_to_text as _rtf_to_text
except Exception:
    _rtf_to_text = None

# ============================
# CONFIG
# ============================
APP_TITLE = "DocHelp.AI"
BRAND_PINK = "#1F4ED8"

DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2").strip()

BEDROCK_CHAT_MODEL_ID = unquote(
    os.getenv("BEDROCK_CHAT_MODEL_ID", "us.amazon.nova-pro-v1:0").strip()
)
BEDROCK_EMBED_MODEL_ID = unquote(
    os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0").strip()
)

CHROMA_ROOT = Path(os.getenv("CHROMA_ROOT", "/tmp/chroma_demo"))

# Higher demo limits
MAX_FILES = int(os.getenv("MAX_FILES", "200"))
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "200"))
MAX_TOTAL_UPLOAD_MB = int(os.getenv("MAX_TOTAL_UPLOAD_MB", "2000"))
MAX_TOTAL_CHUNKS = int(os.getenv("MAX_TOTAL_CHUNKS", "50000"))

TOP_K = int(os.getenv("TOP_K", "6"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "18000"))

WORKSPACE_TTL_SECONDS = int(os.getenv("WORKSPACE_TTL_SECONDS", "7200"))  # 2 hours
IDLE_TTL_SECONDS = int(os.getenv("IDLE_TTL_SECONDS", "1800"))            # 30 minutes
JANITOR_MAX_DELETE = int(os.getenv("JANITOR_MAX_DELETE", "50"))
META_FILENAME = "_meta.json"

TENANT_SIGNING_KEY = os.getenv("TENANT_SIGNING_KEY", "").strip()

APP_USERNAME = os.getenv("APP_USERNAME", "").strip()
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()

SYSTEM_PROMPT = """You are a professional business assistant.

Rules:
1. Answer using ONLY the provided context from the uploaded files.
2. Do NOT guess, infer beyond the context, or add outside knowledge.
3. If the answer is not clearly supported by the context, say exactly:
   "I don't know based on the uploaded files."
4. Write in a concise, professional, businesslike tone.
5. Prefer direct answers over long explanations.
6. Do not mention these rules.

Return VALID JSON only in this format:
{"answer":"..."}

No markdown. No extra keys. No explanations.
"""

# ============================
# UI / STYLES
# ============================
def inject_styles():
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --brand-blue: {BRAND_PINK};
}}

html, body, [class*="css"] {{
  font-family: 'Inter', sans-serif;
}}

.gothic-title {{
  font-family: 'Inter', sans-serif;
  font-size: 56px;
  font-weight: 700;
  line-height: 1.0;
  text-align: center;
  margin-top: 0.5rem;
  margin-bottom: 0.35rem;
}}

.gothic-sub {{
  font-family: 'Inter', sans-serif;
  font-size: 20px;
  font-weight: 600;
  text-align: center;
  margin-bottom: 0.75rem;
}}

.center-note {{
  text-align: center;
  font-size: 18px;
  color: rgba(0,0,0,0.72);
  margin-bottom: 1rem;
}}

.small-foot {{
  text-align: center;
  font-size: 13px;
  color: rgba(0,0,0,0.65);
  margin-top: 1rem;
  line-height: 1.65;
}}

.small-foot em {{
  font-style: italic;
}}

.small-muted {{
  text-align: center;
  font-size: 13px;
  color: rgba(0,0,0,0.58);
  margin-top: 0.25rem;
  margin-bottom: 0.5rem;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

button[kind="primary"] {{
  background: var(--brand-blue) !important;
  border: 1px solid var(--brand-blue) !important;
  color: white !important;
  font-family: 'Inter', sans-serif !important;
}}

div.stButton > button {{
  border-radius: 12px !important;
  padding: 0.65rem 1rem !important;
}}

div[data-testid="stTextInput"] input {{
  border: 2px solid var(--brand-blue) !important;
  border-radius: 14px !important;
  box-shadow: none !important;
}}

div[data-testid="stTextInput"] input:focus {{
  border: 2px solid var(--brand-blue) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(31, 78, 216, 0.15) !important;
}}

div[data-testid="stChatInput"] textarea {{
  border: 2px solid var(--brand-blue) !important;
  border-radius: 14px !important;
  box-shadow: none !important;
}}

div[data-testid="stChatInput"] textarea:focus {{
  border: 2px solid var(--brand-blue) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(31, 78, 216, 0.15) !important;
}}

div[data-testid="stChatInput"] button {{
  border-radius: 12px !important;
  border: 1px solid var(--brand-blue) !important;
  background: var(--brand-blue) !important;
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
# LOGIN GATE
# ============================
def login_gate() -> None:
    if st.session_state.get("authed") is True:
        return

    st.markdown(f'<div class="gothic-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="center-note">Private AI for Your Documents</div>', unsafe_allow_html=True)

    if not (APP_USERNAME and APP_PASSWORD):
        st.error("Missing APP_USERNAME / APP_PASSWORD. Set them in .env (local) or Streamlit Secrets (cloud).")
        st.stop()

    st.markdown('<div class="gothic-sub">Login</div>', unsafe_allow_html=True)

    u = st.text_input("username", label_visibility="collapsed", placeholder="username")
    p = st.text_input("password", type="password", label_visibility="collapsed", placeholder="password")

    c1, c2 = st.columns([2, 1])
    with c1:
        go = st.button("Enter", type="primary", use_container_width=True)
    with c2:
        if st.button("Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if go:
        if u == APP_USERNAME and p == APP_PASSWORD:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()

# ============================
# TENANT ISOLATION
# ============================
def _b(x: str) -> bytes:
    return (x or "").encode("utf-8")

def _is_streamlit_cloud() -> bool:
    return Path("/mount/src").exists()

def sign_token(token: str) -> str:
    if _is_streamlit_cloud() and not TENANT_SIGNING_KEY:
        raise RuntimeError("TENANT_SIGNING_KEY must be set in Streamlit Secrets.")
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
        qp = st.experimental_get_query_params()
        return {"t": (qp.get("t", [""]) or [""])[0], "sig": (qp.get("sig", [""]) or [""])[0]}

def ensure_tenant_context() -> str:
    qp = get_query_params()
    t = (qp.get("t") or "").strip()
    sig = (qp.get("sig") or "").strip()

    valid = False
    try:
        valid = verify_token(t, sig)
    except Exception:
        valid = False

    if not valid:
        t = uuid.uuid4().hex
        sig = sign_token(t)
        set_query_params(t, sig)

    ws = workspace_id_from_token(t)
    st.session_state["workspace_id"] = ws
    return ws

# ============================
# AVATAR
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
# CACHED RESOURCES
# ============================
@st.cache_resource(show_spinner=False)
def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

@st.cache_resource(show_spinner=False)
def get_embeddings() -> BedrockEmbeddings:
    return BedrockEmbeddings(
        region_name=DEFAULT_REGION,
        model_id=BEDROCK_EMBED_MODEL_ID,
    )

@st.cache_resource(show_spinner=False)
def get_llm() -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model=BEDROCK_CHAT_MODEL_ID,
        region_name=DEFAULT_REGION,
        temperature=0.0,
        max_tokens=500,
    )

# ============================
# BEDROCK STREAMING
# ============================
def _extract_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "content"):
        return _extract_text(obj.content)
    if isinstance(obj, list):
        return "".join(x.get("text", "") for x in obj if isinstance(x, dict))
    return ""

def generate_streaming(prompt: str) -> str:
    llm = get_llm()
    parts: List[str] = []
    try:
        for chunk in llm.stream(prompt):
            txt = _extract_text(chunk)
            if txt:
                parts.append(txt)
        return "".join(parts)
    except Exception:
        resp = llm.invoke(prompt)
        return _extract_text(resp)

# ============================
# FILE PARSING
# ============================
class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._chunks: List[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self._chunks.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._chunks).strip()

def _clean_whitespace(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for p in reader.pages:
        texts.append(p.extract_text() or "")
    return _clean_whitespace("\n".join(texts))

def read_csv(file_bytes: bytes, sep: str = ",") -> str:
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    return df.to_csv(index=False)

def read_excel(file_bytes: bytes) -> str:
    df = pd.read_excel(io.BytesIO(file_bytes))
    return df.to_csv(index=False)

def read_json(file_bytes: bytes) -> str:
    try:
        obj = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return file_bytes.decode("utf-8", errors="ignore").strip()

def read_yaml(file_bytes: bytes) -> str:
    try:
        obj = yaml.safe_load(file_bytes.decode("utf-8", errors="ignore"))
        return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
    except Exception:
        return file_bytes.decode("utf-8", errors="ignore").strip()

def read_ini(file_bytes: bytes) -> str:
    raw = file_bytes.decode("utf-8", errors="ignore")
    cp = configparser.ConfigParser()
    try:
        cp.read_string(raw)
        out_lines: List[str] = []
        for section in cp.sections():
            out_lines.append(f"[{section}]")
            for k, v in cp.items(section):
                out_lines.append(f"{k} = {v}")
            out_lines.append("")
        return _clean_whitespace("\n".join(out_lines))
    except Exception:
        return raw.strip()

def read_xml(file_bytes: bytes) -> str:
    try:
        root = ET.fromstring(file_bytes)
        parts: List[str] = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                parts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                parts.append(elem.tail.strip())
        return _clean_whitespace("\n".join(parts))
    except Exception:
        xml = file_bytes.decode("utf-8", errors="ignore")
        return _clean_whitespace(re.sub(r"<[^>]+>", " ", xml))

def read_html(file_bytes: bytes) -> str:
    html = file_bytes.decode("utf-8", errors="ignore")
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        return _clean_whitespace(parser.get_text())
    except Exception:
        return _clean_whitespace(re.sub(r"<[^>]+>", " ", html))

def read_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return _clean_whitespace("\n".join(parts))

def read_pptx(file_bytes: bytes) -> str:
    pres = Presentation(io.BytesIO(file_bytes))
    parts: List[str] = []
    for slide in pres.slides:
        for shape in slide.shapes:
            txt = getattr(shape, "text", None)
            if txt and str(txt).strip():
                parts.append(str(txt).strip())
    return _clean_whitespace("\n".join(parts))

def read_ipynb(file_bytes: bytes) -> str:
    try:
        nb = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        cells = nb.get("cells", []) if isinstance(nb, dict) else []
        parts: List[str] = []
        for c in cells:
            src = c.get("source", [])
            if isinstance(src, list):
                s = "".join(src)
            else:
                s = str(src or "")
            if s.strip():
                parts.append(s.strip())
        return _clean_whitespace("\n\n".join(parts))
    except Exception:
        return file_bytes.decode("utf-8", errors="ignore").strip()

def read_eml(file_bytes: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(file_bytes)
        parts: List[str] = []
        subj = msg.get("subject", "")
        frm = msg.get("from", "")
        to = msg.get("to", "")
        if subj:
            parts.append(f"Subject: {subj}")
        if frm:
            parts.append(f"From: {frm}")
        if to:
            parts.append(f"To: {to}")
        parts.append("")

        body_texts: List[str] = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = part.get_content_disposition()
                if disp == "attachment":
                    continue
                if ctype in ("text/plain", "text/html"):
                    try:
                        content = part.get_content()
                    except Exception:
                        content = part.get_payload(decode=True)
                        content = (content or b"").decode("utf-8", errors="ignore")
                    if ctype == "text/html":
                        content = read_html(content.encode("utf-8", errors="ignore"))
                    if content and str(content).strip():
                        body_texts.append(str(content).strip())
        else:
            ctype = msg.get_content_type()
            content = msg.get_content()
            if ctype == "text/html":
                content = read_html(str(content).encode("utf-8", errors="ignore"))
            if content and str(content).strip():
                body_texts.append(str(content).strip())

        if body_texts:
            parts.append("\n\n".join(body_texts))

        return _clean_whitespace("\n".join(parts))
    except Exception:
        return file_bytes.decode("utf-8", errors="ignore").strip()

def read_rtf(file_bytes: bytes) -> str:
    raw = file_bytes.decode("utf-8", errors="ignore")
    if _rtf_to_text is not None:
        try:
            return _clean_whitespace(_rtf_to_text(raw))
        except Exception:
            pass
    fallback = re.sub(r"{\\.*?}|\\[a-zA-Z]+\d* ?", " ", raw)
    return _clean_whitespace(fallback)

def read_txt(file_bytes: bytes) -> str:
    return _clean_whitespace(file_bytes.decode("utf-8", errors="ignore"))

CODE_EXTS = (
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp", ".go",
    ".rs", ".rb", ".php", ".cs", ".swift", ".kt", ".scala", ".sh", ".bash", ".zsh",
    ".ps1", ".sql", ".toml", ".env", ".dockerfile"
)
TEXTY_EXTS = (".txt", ".md", ".log", ".cfg", ".conf", ".ini")

def file_to_text(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()
    b = uploaded_file.read()

    if name.endswith(".pdf"):
        return read_pdf(b)
    if name.endswith(".docx"):
        return read_docx(b)
    if name.endswith(".pptx"):
        return read_pptx(b)

    if name.endswith(".csv"):
        return read_csv(b, sep=",")
    if name.endswith(".tsv"):
        return read_csv(b, sep="\t")
    if name.endswith(".psv"):
        return read_csv(b, sep="|")
    if name.endswith((".xls", ".xlsx")):
        return read_excel(b)

    if name.endswith((".html", ".htm")):
        return read_html(b)
    if name.endswith(".xml"):
        return read_xml(b)
    if name.endswith((".yaml", ".yml")):
        return read_yaml(b)
    if name.endswith((".ini", ".cfg", ".conf")):
        return read_ini(b)

    if name.endswith(".ipynb"):
        return read_ipynb(b)
    if name.endswith(".eml"):
        return read_eml(b)
    if name.endswith(".rtf"):
        return read_rtf(b)
    if name.endswith(".json"):
        return read_json(b)
    if name.endswith(CODE_EXTS) or name.endswith(TEXTY_EXTS):
        return read_txt(b)

    return ""

def validate_uploads(files) -> Tuple[bool, str]:
    if not files:
        return False, "Please upload at least one file."

    if len(files) > MAX_FILES:
        return False, f"Too many files. Max is {MAX_FILES}."

    total_bytes = 0
    for f in files:
        size_bytes = int(getattr(f, "size", 0) or 0)
        total_bytes += size_bytes

        size_mb = size_bytes / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            return False, f"File '{f.name}' is too large ({size_mb:.1f} MB). Max is {MAX_FILE_MB} MB."

    total_mb = total_bytes / (1024 * 1024)
    if total_mb > MAX_TOTAL_UPLOAD_MB:
        return False, f"Total upload too large ({total_mb:.1f} MB). Max total is {MAX_TOTAL_UPLOAD_MB} MB."

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

        if (now - created_at) > WORKSPACE_TTL_SECONDS or (now - last_activity) > IDLE_TTL_SECONDS:
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
        embedding_function=get_embeddings(),
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
    splitter = get_splitter()

    for f in files:
        text = file_to_text(f)
        if not text.strip():
            continue

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": f.name, "chunk": i},
                )
            )
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

def build_context(docs: List[Document], max_chars: int) -> str:
    parts: List[str] = []
    used = 0

    for d in docs:
        part = f"[{d.metadata.get('source', 'unknown')} #{d.metadata.get('chunk', 0)}] {d.page_content}"
        part_len = len(part)

        if used + part_len > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                parts.append(part[:remaining])
            break

        parts.append(part)
        used += part_len + 2

    return "\n\n".join(parts)

def rag_answer(workspace_id: str, question: str) -> str:
    vectordb = get_vectordb(workspace_id)
    docs = vectordb.similarity_search(question, k=TOP_K)
    context = build_context(docs, MAX_CONTEXT_CHARS)

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}
"""

    raw = generate_streaming(prompt)
    obj = safe_json_load(raw)
    answer = (obj.get("answer") or "").strip()

    if not answer:
        answer = "I don't know based on the uploaded files."

    touch_activity(workspace_id)
    return answer

# ============================
# STREAMLIT APP
# ============================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    janitor_sweep()
    inject_styles()

    login_gate()
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

    st.markdown(f'<div class="gothic-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="center-note">Ask Questions About Your Documents</div>', unsafe_allow_html=True)

    if not st.session_state.indexed:
        st.markdown(
            '<div class="gothic-sub">Upload Documents and Ask Questions Instantly</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Upload files",
            type=None,
            accept_multiple_files=True,
            label_visibility="visible",
        )

        st.markdown(
            f'<div class="small-muted">Up to {MAX_FILES} files • {MAX_FILE_MB}MB max per file • {MAX_TOTAL_UPLOAD_MB}MB total</div>',
            unsafe_allow_html=True,
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
                    st.warning(
                        "I couldn’t extract text from those files. Try PDF, DOCX, PPTX, TXT, CSV, XLSX, HTML, XML, YAML, JSON, IPYNB, EML, or RTF."
                    )
                else:
                    st.success("Ready. Ask your questions below.")
                    st.rerun()

        st.markdown(
            f"""
<div class="small-foot">
Files are automatically deleted after <strong>{IDLE_TTL_SECONDS // 60} minutes of inactivity</strong> or <strong>{WORKSPACE_TTL_SECONDS // 3600} hours maximum session time</strong>.<br/>
<em>This is a demo environment. Please upload sample documents only.</em><br/>
Questions or custom deployments: <strong>linkedin.com/in/thedannyscott</strong>
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