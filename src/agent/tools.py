from langchain.tools import tool
import ollama
from src.config import settings

_retriever = None
_metadata_store = None


def init_tools(retriever, metadata_store):
    global _retriever, _metadata_store
    _retriever = retriever
    _metadata_store = metadata_store


def _llm_call(prompt: str) -> str:
    response = ollama.chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


@tool
def search_papers(query: str) -> str:
    """Search papers by topic. Use when asked about research topics."""
    if _retriever is None:
        return "Retriever not initialized"
    return _retriever.search_formatted(query, top_k=5)


@tool
def summarize_paper(paper_id: str) -> str:
    """Summarize a specific paper. Use after search to get details."""
    if _metadata_store is None:
        return "Metadata store not initialized"
    
    paper = _metadata_store.get_paper(paper_id)
    if not paper:
        return f"Paper {paper_id} not found"
    
    chunks = _metadata_store.get_paper_chunks(paper_id)
    text = "\n".join(c["text"] for c in chunks[:5])[:3000]
    
    prompt = f"""Summarize this paper:
Title: {paper['title']}
Text: {text}

Provide: 1) Main contribution 2) Method 3) Key findings"""
    
    return _llm_call(prompt)


@tool
def compare_papers(paper_id_1: str, paper_id_2: str) -> str:
    """Compare two papers. Use when asked to compare research."""
    if _metadata_store is None:
        return "Metadata store not initialized"
    
    p1 = _metadata_store.get_paper(paper_id_1)
    p2 = _metadata_store.get_paper(paper_id_2)
    
    if not p1:
        return f"Paper {paper_id_1} not found"
    if not p2:
        return f"Paper {paper_id_2} not found"
    
    prompt = f"""Compare these papers:
Paper 1: {p1['title']}
Abstract: {p1['abstract'][:500]}

Paper 2: {p2['title']}
Abstract: {p2['abstract'][:500]}

Compare: methodology, findings, strengths."""
    
    return _llm_call(prompt)