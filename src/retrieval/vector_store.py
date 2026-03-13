from pathlib import Path
import faiss
import numpy as np
from src.config import settings
from src.ingestion.embedder import get_embedder


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunk_ids: list[str] = []
        self.embedder = get_embedder()

    def build(self, chunks: list) -> None:
        texts = [f"Paper: {c.metadata.get('title', '')}\n\n{c.text}" for c in chunks]
        print(f"Embedding {len(texts)} chunks...")
        vectors = self.embedder.embed_texts(texts)
        
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunk_ids = [c.chunk_id for c in chunks]
        print(f"FAISS index: {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("Index not built")
        
        query_vec = self.embedder.embed_query(query)
        scores, indices = self.index.search(query_vec, top_k)
        
        return [(self.chunk_ids[i], float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0]

    def save(self, path: str | None = None) -> None:
        path = Path(path or settings.faiss_index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        np.save(str(path.with_suffix(".ids.npy")), np.array(self.chunk_ids, dtype=object))

    def load(self, path: str | None = None) -> None:
        path = Path(path or settings.faiss_index_path)
        self.index = faiss.read_index(str(path))
        self.chunk_ids = list(np.load(str(path.with_suffix(".ids.npy")), allow_pickle=True))