from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import settings
from src.ingestion.parser import ParsedPaper


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    text: str
    metadata: dict


def chunk_paper(paper: ParsedPaper) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    
    if not paper.full_text.strip():
        return []

    raw_chunks = splitter.split_text(paper.full_text)
    
    return [
        Chunk(
            chunk_id=f"{paper.paper_id}_chunk_{i:04d}",
            paper_id=paper.paper_id,
            text=text,
            metadata={"title": paper.title, "chunk_index": i}
        )
        for i, text in enumerate(raw_chunks)
    ]


def chunk_papers(papers: list[ParsedPaper]) -> list[Chunk]:
    all_chunks = []
    for paper in papers:
        chunks = chunk_paper(paper)
        all_chunks.extend(chunks)
    print(f"Total: {len(all_chunks)} chunks from {len(papers)} papers")
    return all_chunks