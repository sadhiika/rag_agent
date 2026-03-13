# RAG Research Agent

A retrieval-augmented generation system for searching academic papers using a ReAct agent with hybrid search.

## Features

- **Hybrid Search**: FAISS (semantic) + BM25 (keyword) with Reciprocal Rank Fusion
- **ReAct Agent**: Multi-step reasoning with tools (search, summarize, compare)
- **Local LLM**: Runs on Ollama — no API costs

## Tech Stack

LangChain • FAISS • BM25 • FastAPI • Ollama • PyMuPDF • SQLite

## Setup
```bash
# Install Ollama, then pull model
ollama pull llama3.2

# Install dependencies
pip install -r requirements.txt

# Download papers & build index
python scripts/download_papers.py --max 50
python scripts/build_index.py

# Run (two terminals)
ollama serve
python -m uvicorn src.api.main:app
```

## Usage

Open http://127.0.0.1:8000/docs and query:
```json
{"query": "what is bias in language models"}
```

## Architecture
```
Query → FastAPI → ReAct Agent → Hybrid Search (FAISS+BM25) → LLM → Response
```
