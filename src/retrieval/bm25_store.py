import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from src.config import settings


class BM25Store:
    def __init__(self):
        self.index = None
        self.chunk_ids: list[str] = []
        self.corpus: list[list[str]] = []

    def build(self, chunks: list) -> None:
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.corpus = [self._tokenize(c.text) for c in chunks]
        self.index = BM25Okapi(self.corpus)
        print(f"BM25 index: {len(self.corpus)} documents")

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("Index not built")
        
        scores = self.index.get_scores(self._tokenize(query))
        top_idx = scores.argsort()[-top_k:][::-1]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def save(self, path: str | None = None) -> None:
        path = Path(path or settings.bm25_index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"corpus": self.corpus, "chunk_ids": self.chunk_ids}, f)

    def load(self, path: str | None = None) -> None:
        path = Path(path or settings.bm25_index_path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.corpus = data["corpus"]
        self.chunk_ids = data["chunk_ids"]
        self.index = BM25Okapi(self.corpus)

    def _tokenize(self, text: str) -> list[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in text.split() if len(t) > 2]