"""Streamlit UI for CodeLens. Run just this:   streamlit run frontend/streamlit_app.py

It starts the FastAPI backend (backend/app.py) by itself if it is not already running.
"""
import atexit
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import streamlit as st

HERE = Path(__file__).resolve().parent
# Works if this file sits in the project folder or in its frontend/ folder
ROOT = next((p for p in (HERE, HERE.parent) if (p / "backend" / "app.py").exists()), HERE.parent)
API = os.environ.get("CODELENS_API", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RepoLens — Codebase Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling Enhancement
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'Fira Code', monospace !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0px;
    }

    /* Card Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1e293b;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Badges */
    .mode-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 8px;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stChatMessage"][data-testimonial="user"], 
    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #0f172a;
        border-color: #475569;
    }

    /* Custom Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }

    /* Custom Expander */
    .streamlit-expanderHeader {
        background-color: #0f172a;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* Source Links */
    .source-item {
        font-size: 0.85rem;
        padding: 4px 0;
        color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def backend_up():
    try:
        return requests.get(f"{API}/api/status", timeout=2).ok
    except requests.exceptions.RequestException:
        return False


@st.cache_resource
def backend_holder():
    """Remembers the backend process we started, and stops it when Streamlit exits."""
    holder = {"proc": None}
    atexit.register(lambda: holder["proc"] and holder["proc"].poll() is None and holder["proc"].terminate())
    return holder


def ensure_backend():
    """Start backend/app.py if nothing is answering on the API address (local address only)."""
    if backend_up() or not API.startswith(("http://127.0.0.1", "http://localhost")):
        return
    if not (ROOT / "backend" / "app.py").exists():
        st.error(f"Could not find backend/app.py next to this file (looked in {HERE} and {HERE.parent}). "
                 "Keep streamlit_app.py in the project folder, or in a frontend/ folder inside it.")
        st.stop()
    holder = backend_holder()
    if holder["proc"] is None or holder["proc"].poll() is not None:
        holder["proc"] = subprocess.Popen([sys.executable, str(ROOT / "backend" / "app.py")], cwd=ROOT)
    with st.spinner("🚀 Launching backend process..."):
        for _ in range(90):
            if backend_up():
                return
            if holder["proc"].poll() is not None:
                st.error("The backend stopped while starting. Check the terminal for the error "
                         "(for example, port 8000 may already be in use).")
                st.stop()
            time.sleep(1)
    st.error("The backend did not start within 90 seconds. Check the terminal for errors.")
    st.stop()


ensure_backend()


def call(method, path, timeout=30, **kwargs):
    """Call the backend. Returns (json, None) on success or (None, error message)."""
    try:
        r = requests.request(method, f"{API}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the backend. Start it with: python3 backend/app.py"
    except requests.exceptions.Timeout:
        return None, "The backend took too long to respond."
    if r.ok:
        return r.json(), None
    try:
        return None, r.json().get("detail") or f"Error {r.status_code}"
    except ValueError:
        return None, f"Error {r.status_code}"


def source_line(s, repo_url):
    """One source as a markdown line, linked to GitHub when we know the repo."""
    text = f"`{s['file']}`"
    if s.get("lines"):
        text += f", lines {s['lines']}"
        if repo_url:
            start, end = s["lines"].split("-")
            text = f"[{text}]({repo_url}/blob/HEAD/{s['file']}#L{start}-L{end})"
    if s.get("name") and s["name"] != "(top of file)":
        text += f", `{s['name']}`"
    return f"- {text} ({s['via']})"


# ---------- State ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("repo_url", None)

if "stage" not in st.session_state:
    data, err = call("GET", "/api/status", timeout=10)
    if err:
        st.error(err)
        st.stop()
    st.session_state.stage = data["stage"]

poll_every = 2 if st.session_state.stage == "indexing" else None   # only poll while indexing


@st.fragment(run_every=poll_every)
def status_box():
    data, err = call("GET", "/api/status", timeout=10)
    if err:
        st.error(err)
        return
    if data["stage"] != st.session_state.stage:
        st.session_state.stage = data["stage"]
        st.rerun()          # refresh the whole page so the chat unlocks

    if data["stage"] == "indexing":
        st.info(f"⏳ **Indexing in progress:** {data['message']}")
    elif data["stage"] == "ready":
        st.success("🟢 **Status:** Repository indexed & ready")
    elif data["stage"] == "error":
        st.error(f"🔴 **Error:** {data['message']}")
    else:
        st.info("⚪ **Status:** No repo indexed yet")


# ---------- Sidebar: index a repo ----------
with st.sidebar:
    st.title("🔍 RepoLens")
    st.caption("Codebase Intelligence & RAG Assistant")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("⚙️ Index Repository")
        kind = st.radio("Repo Source", ["GitHub URL", "Local folder"], horizontal=True)
        if kind == "GitHub URL":
            value = st.text_input("Public GitHub Repo", placeholder="https://github.com/owner/repo")
        else:
            value = st.text_input("Folder Path", placeholder="/Users/you/Projects/my-repo")

        busy = st.session_state.stage == "indexing"
        if st.button("🚀 Index Repo", type="primary", disabled=busy or not value.strip(), use_container_width=True):
            payload = {"repo_url": value.strip()} if kind == "GitHub URL" else {"path": value.strip()}
            _, err = call("POST", "/api/index", json=payload)
            if err:
                st.error(err)
            else:
                st.session_state.messages = []
                st.session_state.stage = "indexing"
                st.session_state.repo_url = value.strip().removesuffix(".git").rstrip("/") if kind == "GitHub URL" else None
                st.rerun()

    st.markdown("### 📊 Pipeline Status")
    status_box()

    st.markdown("---")
    st.caption("⚡ **Embeddings:** Local Hugging Face  \n🧠 **LLM:** OpenRouter / Nvidia API  \n🗄️ **Storage:** ChromaDB")

# ---------- Main: chat ----------
st.markdown("## 💬 Codebase Assistant")
st.caption("Ask questions about code architecture, implementation details, or dependency flows.")

ready = st.session_state.stage == "ready"
if not ready:
    st.info("💡 **Getting Started:** Index a repository using the sidebar to unlock Q&A features.")

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("mode"):
            st.markdown(f'<span class="mode-badge">MODE: {msg["mode"].upper()}</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Referenced Sources ({len(msg['sources'])})"):
                st.markdown("\n".join(source_line(s, st.session_state.repo_url) for s in msg["sources"]))

# Input Field
question = st.chat_input("Ask a question about the code...", disabled=not ready)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching vector index & generating response..."):
            data, err = call("POST", "/api/ask", timeout=300, json={"question": question})
        if err:
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "mode": data["mode"], "sources": data["sources"]}
            )
            st.rerun()
