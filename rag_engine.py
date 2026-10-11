"""
rag_engine.py — Advanced Hybrid Retrieval-Augmented Generation Engine.
Features:
  1. Dense Semantic Vector Search (ChromaDB)
  2. Sparse Keyword Search (BM25 Okapi)
  3. Reciprocal Rank Fusion (RRF) Re-ranking
  4. Dual-Synthesis LLM Prompting:
     - Research-Backed Evidence (Citations [1], [2])
     - General AI Knowledge & Practical Advice
     - Side-by-side Comparative Insights
"""
import sys
import re
import pickle
import time
from typing import List, Dict, Any, Optional

# Fix Windows terminal encoding for emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import (
    CHROMA_DIR, BM25_INDEX_PATH,
    OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE,
    TOP_K, RRF_K
)


def tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words."""
    return [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-²³]+\b', text)]


class RAGEngine:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing from your .env file!")

        # 1. Initialize OpenAI client
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

        # 2. Connect to existing persistent ChromaDB
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        self.chroma = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embeddings
        )

        # 3. Load BM25 index from disk
        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}. Run `python ingest.py` first!")

        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25_data = pickle.load(f)

        self.bm25 = self.bm25_data["bm25"]
        self.chunks = self.bm25_data["chunks"]
        self.metadatas = self.bm25_data["metadatas"]
        print("✅ RAGEngine initialized: ChromaDB & BM25 loaded into memory.")

    def dense_search(self, query: str, crop_filter: Optional[str] = None, top_k: int = 10) -> List[Dict]:
        """Performs semantic similarity search on ChromaDB."""
        search_kwargs = {"k": top_k}
        if crop_filter and crop_filter.lower() != "all crops":
            search_kwargs["filter"] = {"crop": crop_filter.lower()}

        try:
            docs_and_scores = self.chroma.similarity_search_with_relevance_scores(query, **search_kwargs)
            results = []
            for doc, score in docs_and_scores:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            return results
        except Exception:
            docs = self.chroma.similarity_search(query, **search_kwargs)
            return [{"content": d.page_content, "metadata": d.metadata, "score": 0.8} for d in docs]

    def bm25_search(self, query: str, crop_filter: Optional[str] = None, top_k: int = 10) -> List[Dict]:
        """Performs sparse keyword search using BM25 Okapi."""
        tokens = tokenize(query)
        if not tokens:
            return []

        raw_scores = self.bm25.get_scores(tokens)
        scored_items = []

        for content, meta, score in zip(self.chunks, self.metadatas, raw_scores):
            if score <= 0:
                continue
            if crop_filter and crop_filter.lower() != "all crops":
                if meta.get("crop") != crop_filter.lower():
                    continue
            scored_items.append({
                "content": content,
                "metadata": meta,
                "score": float(score)
            })

        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items[:top_k]

    def hybrid_search(self, query: str, crop_filter: Optional[str] = None, top_k: int = TOP_K) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF):
        Score = 1 / (60 + rank_dense) + 1 / (60 + rank_bm25)
        """
        dense_results = self.dense_search(query, crop_filter, top_k=top_k * 3)
        sparse_results = self.bm25_search(query, crop_filter, top_k=top_k * 3)

        fusion_scores: Dict[str, Dict] = {}

        # Accumulate dense rankings
        for rank, item in enumerate(dense_results, 1):
            key = item["content"]
            if key not in fusion_scores:
                fusion_scores[key] = {
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "rrf_score": 0.0
                }
            fusion_scores[key]["rrf_score"] += 1.0 / (RRF_K + rank)

        # Accumulate sparse rankings
        for rank, item in enumerate(sparse_results, 1):
            key = item["content"]
            if key not in fusion_scores:
                fusion_scores[key] = {
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "rrf_score": 0.0
                }
            fusion_scores[key]["rrf_score"] += 1.0 / (RRF_K + rank)

        ranked = sorted(fusion_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        return ranked[:top_k]

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generates an augmented answer synthesizing:
        1. Specific research findings with inline citations [1], [2]
        2. Broader agronomic intelligence from LLM's general knowledge
        3. A brief comparative takeaway
        """
        # Format research passages
        passages = []
        for i, item in enumerate(context_chunks, 1):
            meta = item.get("metadata", {})
            source = meta.get("source", "Unknown file")
            page = meta.get("page", 1)
            crop = meta.get("crop", "General")
            passages.append(
                f"[Source {i}: {source} | Crop: {crop} | Page: {page}]\n"
                f"{item['content'].strip()}"
            )

        context_str = "\n\n---\n\n".join(passages) if passages else "No indexed research documents found."

        system_prompt = (
            "You are HydroRAG, a concise and expert hydroponics knowledge assistant. Answer the user's question directly and concisely (aim for 3-5 clear bullet points or 1-2 short paragraphs).\n\n"
            "YOUR TASK:\n"
            "Answer the farmer's/researcher's question by combining the retrieved scientific research papers with your broad agricultural knowledge.\n\n"
            "STRUCTURE YOUR ANSWER INTO THESE 3 SECTIONS:\n\n"
            "### 1. Research-Backed Evidence (Grounded Findings)\n"
            "- Extract facts, specific scientific thresholds, and findings directly from the provided Research Context.\n"
            "- Cite the exact sources inline using [1], [2], etc.\n"
            "- Highlight critical quantitative metrics: Electrical Conductivity (EC in dS/m), pH, dissolved oxygen, temperature, and specific PPM balances.\n\n"
            "### 2. General Agronomic Best Practices\n"
            "- Complement the research with your broader horticultural knowledge.\n"
            "- Provide practical advice, operational tips, or troubleshooting steps that general growers use.\n\n"
            "### 3. Comparative Insight & Takeaway\n"
            "- Briefly compare the specific research data against common general farming assumptions.\n"
            "- Give the grower a clear, actionable summary.\n\n"
            f"=== RESEARCH CONTEXT FROM DATABASE ===\n{context_str}"
        )

        response = self.openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=LLM_TEMPERATURE
        )
        return response.choices[0].message.content

    def query(self, user_question: str, crop_filter: Optional[str] = "All Crops", mode: str = "hybrid") -> Dict[str, Any]:
        """Executes full RAG workflow: Retrieve -> Generate -> Profile Latency -> Format Citations."""
        start_time = time.time()

        # 1. Retrieval
        ret_start = time.time()
        if mode == "dense":
            docs = self.dense_search(user_question, crop_filter=crop_filter, top_k=TOP_K)
        elif mode == "sparse":
            docs = self.bm25_search(user_question, crop_filter=crop_filter, top_k=TOP_K)
        else:
            docs = self.hybrid_search(user_question, crop_filter=crop_filter, top_k=TOP_K)
        retrieval_latency = round(time.time() - ret_start, 3)

        # 2. Generation
        gen_start = time.time()
        answer = self.generate_answer(user_question, docs)
        gen_latency = round(time.time() - gen_start, 2)

        total_latency = round(time.time() - start_time, 2)

        # 3. Format structured citations
        citations = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            citations.append({
                "index": i,
                "source": meta.get("source", "unknown"),
                "crop": meta.get("crop", "general"),
                "page": meta.get("page", 1),
                "score": round(doc.get("rrf_score", doc.get("score", 0.0)), 4),
                "snippet": doc["content"][:260].replace("\n", " ").strip() + "..."
            })

        return {
            "answer": answer,
            "citations": citations,
            "latency": total_latency,
            "retrieval_time": retrieval_latency,
            "generation_time": gen_latency,
            "mode": mode,
            "crop_filter": crop_filter
        }


if __name__ == "__main__":
    print("=" * 65)
    print("🌾 Testing AgriRAG Hybrid Engine...")
    print("=" * 65)

    engine = RAGEngine()

    test_query = "What is the optimal EC and temperature range for hydroponic tomatoes, and how do I prevent Blossom-End Rot?"
    print(f"\n🔍 Query: '{test_query}'")
    print("🌿 Filter: Tomatoes | Mode: Hybrid RRF\n")

    result = engine.query(test_query, crop_filter="tomatoes", mode="hybrid")

    print("\n" + "=" * 65)
    print("🤖 AgriRAG Answer:")
    print("=" * 65)
    print(result["answer"])

    print("\n" + "=" * 65)
    print("📚 Retrieved Research Citations:")
    print("=" * 65)
    for c in result["citations"]:
        print(f"[{c['index']}] {c['source']} (Crop: {c['crop']}, Page: {c['page']}) | Score: {c['score']}")
        print(f"    Excerpt: \"{c['snippet']}\"\n")

    print(f"⚡ Performance: Total: {result['latency']}s (Retrieval: {result['retrieval_time']}s, LLM: {result['generation_time']}s)")
