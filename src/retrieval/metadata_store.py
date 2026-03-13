import json
import sqlite3
from pathlib import Path
from src.config import settings


class MetadataStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.sqlite_path
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY, title TEXT, authors TEXT, abstract TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY, paper_id TEXT, text TEXT, metadata TEXT
                );
            """)
        return self._conn

    def add_paper(self, paper):
        self.conn.execute(
            "INSERT OR REPLACE INTO papers VALUES (?,?,?,?)",
            (paper.paper_id, paper.title, json.dumps(paper.authors), paper.abstract)
        )
        self.conn.commit()

    def add_chunks(self, chunks):
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?)",
            [(c.chunk_id, c.paper_id, c.text, json.dumps(c.metadata)) for c in chunks]
        )
        self.conn.commit()

    def get_paper(self, paper_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        if row:
            r = dict(row)
            r["authors"] = json.loads(r["authors"] or "[]")
            return r
        return None

    def get_chunk(self, chunk_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        if row:
            r = dict(row)
            r["metadata"] = json.loads(r["metadata"] or "{}")
            return r
        return None

    def get_paper_chunks(self, paper_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM chunks WHERE paper_id=?", (paper_id,)).fetchall()
        return [dict(r) for r in rows]


_store = None


def get_metadata_store():
    global _store
    if _store is None:
        _store = MetadataStore()
    return _store