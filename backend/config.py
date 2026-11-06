"""Shared paths, settings, and the embedding model / LLM."""
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from functools import lru_cache

# import os

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = str(ROOT / "chroma_db")     # vector database
INDEX_DIR = ROOT / "index_data"      # chunks.json, graph.json, summaries.json
REPOS_DIR = ROOT / "repos"           # cloned GitHub repos
load_dotenv(ROOT / ".env")   


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

# 1. LLM Setup (For Summarization & Question Answering)
def get_llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        model="nvidia/nemotron-3.5-super-vl:free",  
        temperature=0.2
    )
@lru_cache(maxsize=1)
def get_embeddings():
    # Runs on your Mac, no API limits. lru_cache = load the model once, not on every call.
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
 
def get_vectorstore():
    return Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
