"""Index a Python repo: code chunks -> Chroma, plus a code graph and summaries.

Run on its own:  python3 backend/ingestion_pipeline.py /path/to/repo
"""
import ast
import json
import pickle
import shutil
import sys
from pathlib import Path

import networkx as nx
from langchain_core.documents import Document

from config import INDEX_DIR, get_vectorstore

SKIP_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__", "site-packages", "build", "dist"}
BATCH_SIZE = 200        # chunks embedded per step (just for progress updates)
MAX_COMMUNITIES = 15


def find_python_files(repo):
    return [p for p in sorted(repo.rglob("*.py")) if not SKIP_DIRS & set(p.relative_to(repo).parts)]


def chunk_file(path, repo):
    """Split one file into a 'module' chunk plus one chunk per function and class.
    Also returns the names of the modules this file imports."""
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    rel = str(path.relative_to(repo))

    chunks = [{"id": f"{rel}:(top of file):1", "file": rel, "name": "(top of file)", "kind": "module",
               "start": 1, "end": min(len(lines), 60), "text": "\n".join(lines[:60]), "calls": []}]
    imports = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunks, imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[-1] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[-1])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            is_class = isinstance(node, ast.ClassDef)
            end = min(node.end_lineno, node.lineno + 29) if is_class else node.end_lineno  # classes: first 30 lines

            calls = set()   # names this function calls (approximate: matched by name only)
            if not is_class:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        f = sub.func
                        calls.add(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else "")
            chunks.append({"id": f"{rel}:{node.name}:{node.lineno}", "file": rel, "name": node.name,
                           "kind": "class" if is_class else "function", "start": node.lineno, "end": end,
                           "text": "\n".join(lines[node.lineno - 1:end]), "calls": sorted(calls - {""})})
    return chunks, imports


def build_graph(chunks, imports):
    """Code graph. Nodes: files, functions, classes.
    Edges: file --contains--> function, function --calls--> function, file --imports--> file."""
    G = nx.DiGraph()
    by_name = {}
    for c in chunks:
        if c["kind"] == "module":
            G.add_node(c["file"], file=c["file"], name=c["file"], kind="file", start=0, end=0)
        else:
            G.add_node(c["id"], file=c["file"], name=c["name"], kind=c["kind"], start=c["start"], end=c["end"])
            G.add_edge(c["file"], c["id"], type="contains")
            by_name.setdefault(c["name"], []).append(c["id"])

    for c in chunks:
        for callee in c["calls"]:
            for target in by_name.get(callee, []):
                if target != c["id"]:
                    G.add_edge(c["id"], target, type="calls")

    stems = {Path(f).stem: f for f in G.nodes if G.nodes[f]["kind"] == "file"}
    for file, modules in imports.items():
        for m in modules:
            if m in stems and stems[m] != file:
                G.add_edge(file, stems[m], type="imports")
    return G


def describe_communities(G):
    """Find tightly connected groups in the graph (Louvain). No LLM: each group is described by its names."""
    groups = [g for g in nx.community.louvain_communities(G.to_undirected(), seed=1) if len(g) >= 3]
    return [{"files": sorted({G.nodes[n]["file"] for n in g}),
             "summary": "Connected code: " + ", ".join(G.nodes[n]["name"] for n in g if G.nodes[n]["kind"] != "file")[:400]}
            for g in sorted(groups, key=len, reverse=True)[:MAX_COMMUNITIES]]


def summarize_files(repo, files):
    """No LLM: describe each file by its docstring and the names it defines. The LLM reads these at question time."""
    summaries = {}
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        doc = (ast.get_docstring(tree) or "").split("\n")[0]
        names = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        summaries[str(path.relative_to(repo))] = f"{doc} Defines: {', '.join(names[:25])}".strip()
    readme = next(iter(repo.glob("README*")), None)
    if readme:
        summaries["README"] = readme.read_text(encoding="utf-8", errors="ignore")[:4000]
    return summaries


def index_repo(repo_path, on_progress=print):
    repo = Path(repo_path).expanduser().resolve()
    if not repo.is_dir():
        raise ValueError(f"Folder not found: {repo}")
    files = find_python_files(repo)
    if not files:
        raise ValueError("No Python files found in this repo.")

    # Start fresh: one repo is indexed at a time
    try:
        get_vectorstore().delete_collection()
    except Exception:
        pass
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    INDEX_DIR.mkdir(exist_ok=True)

    on_progress("Parsing code and building the code graph")
    chunks, imports = [], {}
    for f in files:
        file_chunks, file_imports = chunk_file(f, repo)
        chunks += file_chunks
        imports[file_chunks[0]["file"]] = file_imports
    G = build_graph(chunks, imports)
    (INDEX_DIR / "graph.pkl").write_bytes(pickle.dumps(G))

    on_progress("Writing file summaries")
    (INDEX_DIR / "summaries.json").write_text(json.dumps(summarize_files(repo, files)))

    on_progress("Finding connected groups in the graph")
    (INDEX_DIR / "communities.json").write_text(json.dumps(describe_communities(G)))

    # Embed the chunks into Chroma in batches (local model: no rate limit, so no waiting)
    docs = [Document(page_content=f"# {c['file']} - {c['name']} (lines {c['start']}-{c['end']})\n{c['text']}",
                     metadata={k: c[k] for k in ("id", "file", "name", "kind", "start", "end")}) for c in chunks]
    vectorstore = get_vectorstore()
    for i in range(0, len(docs), BATCH_SIZE):
        vectorstore.add_documents(docs[i:i + BATCH_SIZE])
        on_progress(f"Embedded {min(i + BATCH_SIZE, len(docs))} of {len(docs)} chunks")

    # Written last: its existence means the index is complete
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks))
    on_progress("Done")


if __name__ == "__main__":
    index_repo(sys.argv[1])