import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent.executor import create_agent_executor

agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    print("Loading agent...")
    agent = create_agent_executor()
    print("Ready!")
    yield


app = FastAPI(title="RAG Research Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if agent is None:
        raise HTTPException(503, "Agent not loaded")
    
    start = time.time()
    try:
        result = agent.invoke({"input": req.query})
        return QueryResponse(
            answer=result["output"],
            latency_ms=round((time.time() - start) * 1000, 1)
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "agent_loaded": agent is not None}