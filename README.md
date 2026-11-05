# 🔍 RepoLens — Codebase Intelligence & RAG Assistant

**RepoLens** is a hybrid Retrieval-Augmented Generation (RAG) assistant designed to index local folders or public GitHub repositories and answer questions about code architecture, implementation logic, and dependency flows with exact source line citations.

---

## ⚡ Key Architecture Highlights

* **Hybrid Ingestion Architecture:** Uses a **local Hugging Face embedding model** (`all-MiniLM-L6-v2`) for repository indexing. Embeddings run 100% offline on your machine—eliminating API costs and bypassing cloud rate-limit bottlenecks (e.g., Google Gemini's 15 RPM limit).
* **High-Efficiency Retrieval:** Query inference is routed via **OpenRouter / Nvidia API** (e.g., `nvidia/nemotron-3.5-super-vl:free`), triggering cloud API calls strictly on question retrieval.
* **Vector Storage:** Embedded code chunks are indexed locally in **ChromaDB** for instant vector searches.
* **Auto-Managing Streamlit UI:** A custom Streamlit frontend that automatically detects, launches, and handles the lifecycle of the FastAPI backend in a single workflow.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** FastAPI, Uvicorn
* **Embeddings:** Hugging Face (`sentence-transformers`)
* **Vector DB:** ChromaDB
* **LLM Engine:** OpenRouter API / Nvidia API / LangChain

---

## 📁 Project Structure

```text
repo-lens/
├── backend/
│   ├── app.py                   # FastAPI server & endpoints (/api/index, /api/ask, /api/status)
│   ├── config.py                # Pipeline settings & vectorstore configs
│   ├── ingestion_pipeline.py    # Code splitting, embedding generation & ChromaDB indexing
│   └── retrieval_pipeline.py    # Context retrieval & LLM generation logic
├── streamlit_app.py             # Streamlit UI & auto-backend orchestration
├── .env                         # API keys & environment variables
└── requirements.txt             # Project dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
Python 3.10+ recommended.

### 2. Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone [https://github.com/your-username/repo-lens.git](https://github.com/your-username/repo-lens.git)
cd repo-lens

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## 💻 Running the Application


```bash
python3 -m streamlit run streamlit_app.py
```

---

## 📖 Usage Flow

1. **Index a Repository:** 
   * Select **GitHub URL** (e.g., `https://github.com/owner/repo`) or **Local folder** in the sidebar.
   * Click **🚀 Index Repo**. The local embedding pipeline will split the code and build the vector database.
2. **Ask Questions:** 
   * Once status changes to `🟢 Repo is ready`, type any question in the main chat input (e.g., *"How does the authentication flow work?"* or *"Where is the vector store initialized?"*).
3. **Inspect Sources:** 
   * Expand the **Referenced Sources** dropdown under any response to see exact files, line numbers, and GitHub deep links.
