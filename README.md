# 🚀 Ultra-Enterprise Hybrid RAG System (v6.1 Pro)

An enterprise-grade, vendor-agnostic Retrieval-Augmented Generation (RAG) pipeline built for high-performance production environments. Powered by **Google Gemini 3.6-flash**, hybrid retrieval (**FAISS + BM25**), Reciprocal Rank Fusion (**RRF**), and **Cross-Encoder** precision re-ranking.

---

## 🏗️ Architecture & Core Components

1. **Smart Chunking Engine**: Custom text splitting with intelligent context overlap.
2. **Hybrid Retrieval**: Combines Dense Retrieval (FAISS vector search with L2 normalization) and Sparse Retrieval (BM25 lexical search).
3. **Reciprocal Rank Fusion (RRF)**: Merges and normalizes hybrid search rankings seamlessly.
4. **Precision Re-ranking**: Filters and scores candidate chunks using a `Cross-Encoder` model (`ms-marco-MiniLM-L-6-v2`) to maximize context relevance.
5. **Structured Validation & Streaming**: Utilizes Pydantic schemas for deterministic JSON outputs and supports real-time streaming via the official `google-genai` SDK.
6. **Error Resilience**: Automatic retry mechanism with exponential backoff for handling server spikes (`503 UNAVAILABLE`).

---

## 🛠️ Tech Stack
* **Python 3.10+**
* **Google GenAI SDK** (`google-genai`)
* **FAISS** (`faiss-cpu`)
* **Sentence Transformers** & **Cross-Encoders**
* **Pydantic v2**

---

## ⚙️ Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

   Set your Gemini API Key:
Windows (PowerShell): $env:GEMINI_API_KEY="AQ.YOUR_API_KEY_HERE"
Linux/macOS: export GEMINI_API_KEY="AQ.YOUR_API_KEY_HERE"

Run the Pipeline:
python main.py

