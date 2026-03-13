import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.parser import parse_directory
from src.ingestion.chunker import chunk_papers
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.metadata_store import MetadataStore
from src.config import settings


def main():
    print("Building indices...")
    
    papers = parse_directory(settings.raw_papers_dir)
    if not papers:
        print("No PDFs found in data/raw/")
        sys.exit(1)
    
    chunks = chunk_papers(papers)
    
    vs = VectorStore()
    vs.build(chunks)
    vs.save()
    
    bm = BM25Store()
    bm.build(chunks)
    bm.save()
    
    ms = MetadataStore()
    for p in papers:
        ms.add_paper(p)
    ms.add_chunks(chunks)
    
    print(f"Done! {len(papers)} papers, {len(chunks)} chunks")


if __name__ == "__main__":
    main()

