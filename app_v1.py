# app.py
# Beleza AI — Simple Upload → Chat RAG Demo (LangChain + Chroma + AWS Bedrock)
# ✅ Self-cleaning: automatic deletion via TTL + idle timeout + janitor sweep (no cron needed)
#
# Install:
#   pip install -U streamlit python-dotenv pandas pypdf chromadb \
#     langchain langchain-core langchain-text-splitters langchain-chroma langchain-aws boto3
#
# Run:
#   streamlit run app.py
#
# .env (example):
#   AWS_ACCESS_KEY_ID=...
#   AWS_SECRET_ACCESS_KEY=...
#   AWS_DEFAULT_REGION=us-west-2
#   BEDROCK_CHAT_MODEL_ID=us.amazon.nova-pro-v1:0
#   BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0

import os
import io
import json
import uuid
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import streamlit as st

# Optional .env support
try:
    from dotenv import load_dotenv
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
APP_TITLE = "Beleza AI"
DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

# One chat model (no UI selector) — unquote prevents accidental %3A encoding issues.
BEDROCK_CHAT_MODEL_ID = unquote(os.getenv("BEDROCK_CHAT_MODEL_ID", "us.amazon.nova-pro-v1:0").strip())

# Embeddings model — unquote prevents accidental %3A encoding issues.
BEDROCK_EMBED_MODEL_ID = unquote(os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0").strip())

# Local persistence root (ephemeral, container-local)
CHROMA_ROOT = Path(os.getenv("CHROMA_ROOT", "/tmp/chroma_demo"))

# Safety limits
MAX_FILES = int(os.getenv("MAX_FILES", "15"))
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "20"))
MAX_TOTAL_CHUNKS = int(os.getenv("MAX_TOTAL_CHUNKS", "2000"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ============================
# AUTO-DELETE (NO-MAINTENANCE)
# ============================
WORKSPACE_TTL_SECONDS = int(os.getenv("WORKSPACE_TTL_SECONDS", "7200"))  # 2 hours
IDLE_TTL_SECONDS = int(os.getenv("IDLE_TTL_SECONDS", "1800"))            # 30 minutes
JANITOR_MAX_DELETE = int(os.getenv("JANITOR_MAX_DELETE", "50"))
META_FILENAME = "_meta.json"

SYSTEM_PROMPT = """You are Beleza AI, a helpful assistant.
You MUST answer using ONLY the provided context from the uploaded files.
If the answer is not in the context, say: "I don't know based on the uploaded files."
Be concise and direct.

Return VALID JSON only in this format:
{"answer":"...", "citations":[{"source":"filename.ext","chunk":0}]}

No markdown. No extra keys. No explanations.
"""


# ============================
# BEDROCK HELPERS
# ============================
def bedrock_llm(model_id: str) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model=model_id,
        region_name=DEFAULT_REGION,
        temperature=0.2,
        max_tokens=500,
    )


def bedrock_embeddings() -> BedrockEmbeddings:
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


def generate_streaming(llm: ChatBedrockConverse, prompt: str) -> Tuple[str, Optional[float], float]:
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


def touch_activity() -> None:
    now = _now()
    if "created_at" not in st.session_state:
        st.session_state.created_at = now
    st.session_state.last_activity = now

    ws = st.session_state.get("workspace_id")
    if ws:
        d = CHROMA_ROOT / ws
        if d.exists():
            write_workspace_meta(d, st.session_state.created_at, st.session_state.last_activity)


# ============================
# VECTOR STORE
# ============================
def get_workspace_id() -> str:
    if "workspace_id" not in st.session_state:
        st.session_state.workspace_id = str(uuid.uuid4())
    return st.session_state.workspace_id


def workspace_dir(workspace_id: str) -> Path:
    d = CHROMA_ROOT / workspace_id
    d.mkdir(parents=True, exist_ok=True)
    created_at = int(st.session_state.get("created_at") or _now())
    last_activity = int(st.session_state.get("last_activity") or _now())
    write_workspace_meta(d, created_at, last_activity)
    return d


def get_vectordb() -> Chroma:
    ws = get_workspace_id()
    d = workspace_dir(ws)
    return Chroma(
        collection_name="demo_docs",
        persist_directory=str(d),
        embedding_function=bedrock_embeddings(),
    )


def reset_workspace(hard_delete: bool = True) -> None:
    ws = st.session_state.get("workspace_id")
    if ws and hard_delete:
        d = CHROMA_ROOT / ws
        try:
            if d.exists():
                shutil.rmtree(d)
        except Exception:
            pass

    st.session_state.workspace_id = str(uuid.uuid4())
    st.session_state.indexed = False
    st.session_state.last_index_count = 0
    st.session_state.messages = []
    st.session_state.created_at = _now()
    st.session_state.last_activity = _now()


def index_files(files) -> int:
    vectordb = get_vectordb()
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
    touch_activity()
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
    return {"answer": s.strip(), "citations": []}


def rag_answer(question: str) -> Dict[str, Any]:
    vectordb = get_vectordb()
    docs = vectordb.similarity_search(question, k=TOP_K)

    context_lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        chunk = d.metadata.get("chunk", 0)
        context_lines.append(f"[{src} #{chunk}] {d.page_content}")

    context = "\n\n".join(context_lines)

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}
"""

    llm = bedrock_llm(BEDROCK_CHAT_MODEL_ID)
    raw, _, _ = generate_streaming(llm, prompt)
    obj = safe_json_load(raw)

    if not isinstance(obj.get("citations"), list) or len(obj.get("citations")) == 0:
        obj["citations"] = [{"source": d.metadata.get("source", "unknown"), "chunk": d.metadata.get("chunk", 0)} for d in docs]

    obj["answer"] = (obj.get("answer") or "").strip() or "I don't know based on the uploaded files."
    touch_activity()
    return obj


# ============================
# STREAMLIT UI (CLEAN)
# ============================
def main():
    janitor_sweep()

    st.set_page_config(APP_TITLE, layout="centered")
    st.title("Beleza AI")

    # init state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "created_at" not in st.session_state:
        st.session_state.created_at = _now()
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = _now()

    get_workspace_id()
    touch_activity()

    # --- Upload first ---
    if not st.session_state.indexed:
        st.markdown("### Please upload files for Beleza AI to help you")
        uploaded = st.file_uploader(
            "Upload PDF, TXT, or CSV",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            index_clicked = st.button("Upload & Prepare", type="primary", use_container_width=True)
        with col2:
            if st.button("Reset", use_container_width=True):
                reset_workspace(hard_delete=True)
                st.rerun()

        if index_clicked:
            ok, msg = validate_uploads(uploaded)
            if not ok:
                st.error(msg)
            else:
                with st.spinner("Preparing your files..."):
                    n = index_files(uploaded)
                if n == 0:
                    st.warning("I couldn't extract any text from those files. Try a different file or a text-based PDF.")
                else:
                    st.success("Ready. Ask your questions below.")
                    st.rerun()

        st.caption(
            f"Limits: {MAX_FILES} files • {MAX_FILE_MB}MB each • "
            f"Auto-delete after {IDLE_TTL_SECONDS//60} min idle or {WORKSPACE_TTL_SECONDS//3600} hrs."
        )
        return

    # --- Chat UI ---
    top = st.columns([1, 1, 1])
    with top[2]:
        if st.button("Reset / Upload new files", use_container_width=True):
            reset_workspace(hard_delete=True)
            st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("citations"):
                with st.expander("Sources"):
                    st.json(m["citations"])

    q = st.chat_input("Ask a question about your uploaded files…")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                obj = rag_answer(q)

            st.markdown(obj["answer"])
            with st.expander("Sources"):
                st.json(obj.get("citations", []))

        st.session_state.messages.append(
            {"role": "assistant", "content": obj["answer"], "citations": obj.get("citations", [])}
        )


if __name__ == "__main__":
    main()
