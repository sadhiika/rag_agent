from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    faiss_index_path: str = "data/index/faiss.index"
    bm25_index_path: str = "data/index/bm25.pkl"
    sqlite_path: str = "data/index/metadata.db"
    raw_papers_dir: str = "data/raw"

    class Config:
        env_file = ".env"


settings = Settings()