"""FastAPI server for CodeLens.  Run:  python3 backend/app.py   then open http://127.0.0.1:8000/docs"""
import os
import re
import shutil
import subprocess

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

import ingestion_pipeline
import retrieval_pipeline
from config import INDEX_DIR, REPOS_DIR

app = FastAPI(title="Repo-lens")

# stage: idle | indexing | ready | error   (one repo at a time, one local user)
state = {"stage": "ready" if (INDEX_DIR / "chunks.json").exists() else "idle", "message": ""}


class IndexRequest(BaseModel):
    path: str | None = None       # a folder on this computer
    repo_url: str | None = None   # a public GitHub repo URL


class AskRequest(BaseModel):
    question: str


def run_index(path, repo_url):
    """Background job: (optionally) clone the repo, then index it."""
    try:
        if repo_url:
            state["message"] = "Cloning repo"
            REPOS_DIR.mkdir(exist_ok=True)
            name = re.sub(r"[^A-Za-z0-9._-]", "_", repo_url.rstrip("/").removesuffix(".git").split("/")[-1])
            path = REPOS_DIR / name
            shutil.rmtree(path, ignore_errors=True)
            subprocess.run(["git", "clone", "--depth", "1", "--", repo_url, str(path)], check=True, capture_output=True)
        ingestion_pipeline.index_repo(path, on_progress=lambda m: state.update(message=m))
        state.update(stage="ready", message="Repo is ready")
    except Exception as e:
        state.update(stage="error", message=str(e))


@app.post("/api/index")
def index(req: IndexRequest, background: BackgroundTasks):
    if state["stage"] == "indexing":
        raise HTTPException(409, "Already indexing a repo. Wait for it to finish.")
    if not (req.path or req.repo_url):
        raise HTTPException(400, "Give a local folder path or a GitHub URL.")
    if req.repo_url and not req.repo_url.startswith("https://github.com/"):
        raise HTTPException(400, "Only https://github.com/ URLs are supported.")

    state.update(stage="indexing", message="Starting")
    background.add_task(run_index, req.path, req.repo_url)
    return {"ok": True}


@app.get("/api/status")
def status():
    return state


@app.post("/api/ask")
def ask(req: AskRequest):
    if state["stage"] == "indexing":
        raise HTTPException(409, "Still indexing the repo.")
    if state["stage"] != "ready":
        raise HTTPException(400, "Index a repo first.")
    if not req.question.strip():
        raise HTTPException(400, "Type a question first.")
    try:
        return retrieval_pipeline.ask(req.question.strip())
    except Exception as e:
        raise HTTPException(500, f"Could not answer that: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)