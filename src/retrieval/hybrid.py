from src.config import settings
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.metadata_store import get_metadata_store


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.metadata_store = get_metadata_store()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        vec_results = self.vector_store.search(query, top_k=20)
        bm25_results = self.bm25_store.search(query, top_k=20)
        
        scores = {}
        k = 60
        for rank, (doc_id, _) in enumerate(vec_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(bm25_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for chunk_id, score in ranked:
            chunk = self.metadata_store.get_chunk(chunk_id)
            if chunk:
                results.append({
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "score": score,
                    "paper_id": chunk["paper_id"],
                    "metadata": chunk.get("metadata", {})
                })
        return results

    def search_formatted(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k)
        if not results:
            return "No papers found."
        
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Result {i}] {r['metadata'].get('title', 'Unknown')}\n"
                f"  Paper ID: {r['paper_id']}\n"
                f"  Content: {r['text'][:300]}...\n"
            )
        return "\n".join(parts)