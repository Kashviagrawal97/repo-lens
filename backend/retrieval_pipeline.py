"""Answer questions about the indexed repo.

Specific questions: hybrid search (vector + BM25), then walk the code graph up to 2 hops.
Overview questions: use the README, file summaries and graph community summaries.
"""
import json
import pickle
import re

import networkx as nx
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

from config import INDEX_DIR, get_llm, get_vectorstore

# Simple router: these phrases mean "explain the whole project"
OVERVIEW_HINTS = ("overview", "overall", "architecture", "this project", "the project", "this repo",
                  "the repo", "codebase", "whole", "entire", "poora project", "project explain", "project kya")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that explains a code repository.
Answer using ONLY the context below. When you refer to code, mention the file and lines like (file.py:21-62).
If the answer is not in the context, say you could not find it in the repository.

Context:
{context}

Question: {question}

Answer:
""")


def as_text(response):
    """Chat models return a message (.text), some local LLM wrappers return a plain string. Accept both."""
    return response if isinstance(response, str) else response.text


def with_llm_summaries(chunks, summaries):
    """Overview questions: write a proper LLM summary for each file once, then reuse it from a cache file."""
    cache_path = INDEX_DIR / "llm_summaries.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    llm = get_llm()
    result = dict(summaries)
    for f in summaries:
        if f == "README":
            continue
        if f not in cache:
            code = "\n".join(c["text"] for c in chunks if c["file"] == f)[:4000]
            try:
                cache[f] = as_text(llm.invoke(f"In 2 short sentences, explain what this Python file does.\n\nFile: {f}\n\n{code}"))
            except Exception as e:
                print(f"LLM summary failed for {f}: {e}")   # keep the simple summary, try again next time
                continue
        result[f] = cache[f]
    cache_path.write_text(json.dumps(cache))
    return result


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())   # get_vectorstore -> "get", "vectorstore"


def search(question, chunks, G):
    by_id = {c["id"]: c for c in chunks}

    # 1. Vector search (meaning)
    vector_ids = [d.metadata["id"] for d in get_vectorstore().similarity_search(question, k=6)]

    # 2. Keyword search with BM25 (exact names)
    bm25 = BM25Okapi([tokenize(c["name"] + " " + c["text"]) for c in chunks])
    scores = bm25.get_scores(tokenize(question))
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:6]
    keyword_ids = [chunks[i]["id"] for i in order if scores[i] > 0]

    # Function or class names written in the question go first
    q = question.lower()
    exact_ids = [c["id"] for c in chunks if c["kind"] != "module" and len(c["name"]) >= 4 and c["name"].lower() in q]
    keyword_ids = exact_ids + [i for i in keyword_ids if i not in exact_ids]

    # 3. Merge both rankings (reciprocal rank fusion) and keep the best 5
    fused = {}
    for ranking in (vector_ids, keyword_ids):
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
    top = [by_id[cid] for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:5] if cid in by_id]

    # 4. Graph walk: from each hit, follow call edges (callers and callees) up to 2 hops
    calls = nx.Graph([(u, v) for u, v, d in G.edges(data=True) if d["type"] == "calls"])
    top_ids = {c["id"] for c in top}
    hops = {}
    for c in top:
        if c["id"] in calls:
            for cid, dist in nx.single_source_shortest_path_length(calls, c["id"], cutoff=2).items():
                if cid not in top_ids:
                    hops[cid] = min(dist, hops.get(cid, dist))
    related = [(by_id[cid], dist) for cid, dist in sorted(hops.items(), key=lambda x: x[1])[:8] if cid in by_id]
    return top, related


def ask(question):
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text())
    G = pickle.loads((INDEX_DIR / "graph.pkl").read_bytes())
    summaries = json.loads((INDEX_DIR / "summaries.json").read_text())
    communities = json.loads((INDEX_DIR / "communities.json").read_text())

    if any(hint in question.lower() for hint in OVERVIEW_HINTS):
        mode = "overview"
        summaries = with_llm_summaries(chunks, summaries)
        files_text = "\n".join(f"{f}: {s}" for f, s in summaries.items() if f != "README")
        groups_text = "\n".join(f"Group ({', '.join(g['files'])}): {g['summary']}" for g in communities)
        context = f"README:\n{summaries.get('README', '')}\n\nConnected groups of code:\n{groups_text}\n\nFiles:\n{files_text}"
        sources = [{"file": f, "name": None, "lines": None, "via": "file summary"} for f in summaries if f != "README"]
        sources += [{"file": ", ".join(g["files"]), "name": None, "lines": None, "via": "graph community"} for g in communities]
    else:
        mode = "specific"
        top, related = search(question, chunks, G)
        context = "\n\n".join(f"[{c['file']}:{c['start']}-{c['end']}]\n{c['text']}" for c in top + [c for c, _ in related])
        sources = [{"file": c["file"], "name": c["name"], "lines": f"{c['start']}-{c['end']}", "via": "search"} for c in top]
        sources += [{"file": c["file"], "name": c["name"], "lines": f"{c['start']}-{c['end']}", "via": f"graph, {d} hop"} for c, d in related]

    response = (prompt | get_llm()).invoke({"context": context, "question": question})
    return {"mode": mode, "answer": as_text(response), "sources": sources}


if __name__ == "__main__":
    while True:
        q = input("\nAsk about the repo (or 'exit'): ")
        if q.lower() == "exit":
            break
        result = ask(q)
        print(f"\n[{result['mode']}] {result['answer']}\n")
        for s in result["sources"]:
            print(f"- {s['file']}" + (f":{s['lines']}" if s["lines"] else "") + f" ({s['via']})")