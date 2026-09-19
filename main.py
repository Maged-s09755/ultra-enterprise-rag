# ==========================================
# Ultra-Enterprise Hybrid RAG System (v6.1 Pro - Fixed Model Name)
# ==========================================
import os
import time
import logging
import math
from typing import List, Dict, Any, Generator
from pydantic import BaseModel, Field
import numpy as np
import faiss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer, CrossEncoder

# إعداد السجلات الهندسية
logger = logging.getLogger("Ultra_Enterprise_RAG_v6")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'))
    logger.addHandler(handler)

# ==========================================
# 🔑 مفتاح المصادقة الخاص بك:
# ==========================================
API_KEY = os.environ.get("GEMINI_API_KEY", "حط-مفتاح-هنا")
client = genai.Client(api_key=API_KEY)

# ==========================================
# 1. التكوينات وإدارة الهيكلة (تم تعديل النموذج إلى gemini-3.6-flash)
# ==========================================
class RAGConfig(BaseModel):
    llm_model: str = Field(default="gemini-3.6-flash", description="Production LLM")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Dense Retriever Model")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Precision Reranker")
    temperature: float = Field(default=0.0, description="Strict deterministic output")
    chunk_size: int = Field(default=300, description="Max characters per document chunk")
    chunk_overlap: int = Field(default=50, description="Overlap characters between chunks")
    top_k_retrieve: int = Field(default=10, description="Initial hybrid retrieval pool")
    top_k_final: int = Field(default=3, description="Final context window size")
    rrf_k: int = Field(default=60, description="RRF constant")

class RawDocument:
    def __init__(self, page_content: str, metadata: dict, doc_id: str):
        self.page_content = page_content
        self.metadata = metadata
        self.doc_id = doc_id

class RAGResponseSchema(BaseModel):
    answer: str = Field(description="Precise answer based solely on context.")
    confidence_score: float = Field(description="Confidence between 0.0 and 1.0.")
    reasoning: str = Field(description="Step-by-step logical justification.")
    is_grounded: bool = Field(description="True if all claims are backed by context.")

# ==========================================
# 2. نظام تقطيع المستندات الذكي (Smart Chunking Engine)
# ==========================================
class DocumentChunker:
    @staticmethod
    def chunk_documents(documents: List[RawDocument], chunk_size: int, overlap: int) -> List[RawDocument]:
        chunked_docs = []
        for doc in documents:
            text = doc.page_content
            if len(text) <= chunk_size:
                chunked_docs.append(doc)
                continue
            
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                new_metadata = doc.metadata.copy()
                new_metadata["chunk_index"] = chunk_idx
                
                chunked_docs.append(RawDocument(
                    page_content=chunk_text,
                    metadata=new_metadata,
                    doc_id=f"{doc.doc_id}_c{chunk_idx}"
                ))
                start += chunk_size - overlap
                chunk_idx += 1
        logger.info(f"Split {len(documents)} source docs into {len(chunked_docs)} optimized chunks.")
        return chunked_docs

# ==========================================
# 3. محرك البحث النصي BM25
# ==========================================
class SimpleBM25:
    def __init__(self, documents: List[RawDocument]):
        self.documents = documents
        self.corpus = [doc.page_content.lower().split() for doc in self.documents]
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / len(self.corpus) if self.corpus else 0
        self.N = len(self.corpus)
        self.idf = {}
        self._compute_idf()

    def _compute_idf(self):
        df = {}
        for doc in self.corpus:
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
        for word, freq in df.items():
            self.idf[word] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    def score(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        q_tokens = query.lower().split()
        scores = np.zeros(self.N)
        for idx, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_len[idx]
            term_freqs = {}
            for word in doc:
                term_freqs[word] = term_freqs.get(word, 0) + 1
            for token in q_tokens:
                if token in term_freqs:
                    tf = term_freqs[token]
                    idf = self.idf.get(token, 0)
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / self.avgdl))
                    score += idf * (numerator / denominator)
            scores[idx] = score
        return scores

# ==========================================
# 4. المنظومة المركزية (Ultra RAG مع FAISS + Streaming)
# ==========================================
class UltraProductionHybridRAGv6:
    def __init__(self, raw_documents: List[RawDocument], config: RAGConfig = RAGConfig()):
        self.config = config
        
        # تنفيذ التقطيع الذكي
        self.documents = DocumentChunker.chunk_documents(
            raw_documents, self.config.chunk_size, self.config.chunk_overlap
        )
        
        logger.info("Initializing neural models and FAISS Vector Database...")
        self.encoder = SentenceTransformer(self.config.embedding_model)
        self.reranker = CrossEncoder(self.config.reranker_model)
        
        # تجهيز التضمينات الكثيفة وبناء مؤشر FAISS
        texts = [doc.page_content for doc in self.documents]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        faiss.normalize_L2(embeddings)
        self.dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(embeddings)
        
        # تجهيز محرك BM25
        self.bm25 = SimpleBM25(self.documents)
        logger.info(f"FAISS Vector DB successfully built with {self.faiss_index.ntotal} vector indices!")

    def query(self, query_text: str, stream: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        metrics = {"retrieval_time": 0, "rerank_time": 0, "llm_time": 0}
        
        try:
            # 1. الاسترجاع الهجين
            ret_start = time.time()
            q_emb = self.encoder.encode(query_text, convert_to_numpy=True)
            q_emb = q_emb / np.linalg.norm(q_emb)
            q_emb = np.expand_dims(q_emb, axis=0)
            
            k_search = min(self.config.top_k_retrieve, len(self.documents))
            dense_scores, dense_indices = self.faiss_index.search(q_emb, k_search)
            dense_indices = dense_indices[0]
            
            bm25_scores = self.bm25.score(query_text)
            bm25_ranks = np.argsort(bm25_scores)[::-1]
            
            rrf_scores = {}
            k_rrf = self.config.rrf_k
            for rank, idx in enumerate(dense_indices):
                if idx != -1:
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
            for rank, idx in enumerate(bm25_ranks[:k_search]):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
                
            sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            initial_docs = [self.documents[i] for i in sorted_indices[:k_search]]
            metrics["retrieval_time"] = round(time.time() - ret_start, 4)

            # 2. Cross-Encoder Re-ranking
            rerank_start = time.time()
            pairs = [[query_text, doc.page_content] for doc in initial_docs]
            scores = self.reranker.predict(pairs)
            scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, score in scored_docs[:self.config.top_k_final]]
            metrics["rerank_time"] = round(time.time() - rerank_start, 4)

            # بناء السياق
            context = "\n\n---\n\n".join([f"ID: {doc.doc_id} | Meta: {doc.metadata}\nContent: {doc.page_content}" for doc in final_docs])
            prompt = f"""You are an elite enterprise AI chief intelligence officer. Answer the user question rigorously based on the context. If info is missing, set `is_grounded` to False.

Context:
{context}

Query: {query_text}
"""
            llm_start = time.time()
            
            if stream:
                response_stream = client.models.generate_content_stream(
                    model=self.config.llm_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=self.config.temperature),
                )
                metrics["llm_time"] = round(time.time() - llm_start, 4)
                return {
                    "stream_generator": response_stream,
                    "sources": [{"doc_id": doc.doc_id, "metadata": doc.metadata} for doc in final_docs],
                    "metrics": metrics
                }
            else:
                response = client.models.generate_content(
                    model=self.config.llm_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.config.temperature,
                        response_mime_type="application/json",
                        response_schema=RAGResponseSchema,
                    ),
                )
                parsed = RAGResponseSchema.model_validate_json(response.text)
                metrics["llm_time"] = round(time.time() - llm_start, 4)
                
                return {
                    "answer": parsed.answer,
                    "confidence_score": parsed.confidence_score,
                    "reasoning": parsed.reasoning,
                    "is_grounded": parsed.is_grounded,
                    "sources": [{"doc_id": doc.doc_id, "metadata": doc.metadata} for doc in final_docs],
                    "metrics": metrics
                }

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {"answer": f"Error: {str(e)}", "sources": [], "metrics": metrics}

# ==========================================
# اختبار المنظومة
# ==========================================
if __name__ == "__main__":
    docs = [
        RawDocument("Alpha company's Q1 2024 net profit reached $50 million, reflecting a robust 10% YoY growth driven by cloud software expansion in North America.", {"dept": "finance"}, "doc_1"),
        RawDocument("The corporate remote work policy mandates 3 days in-office and 2 days flexible per week to maintain agile collaboration across engineering teams.", {"dept": "hr"}, "doc_2")
    ]
    
    rag = UltraProductionHybridRAGv6(raw_documents=docs)
    
    print("\n" + "="*50)
    print("🚀 Testing Real-time Streaming Response:")
    print("="*50)
    res = rag.query("What is the remote work policy?", stream=True)
    for chunk in res["stream_generator"]:
        print(chunk.text, end="", flush=True)
    print("\n" + "="*50)
    print(f"Sources: {res['sources']}")
    print(f"Metrics: {res['metrics']}")
    print("="*50)
