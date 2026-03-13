from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from src.agent.tools import search_papers, summarize_paper, compare_papers, init_tools
from src.config import settings
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.metadata_store import get_metadata_store


PROMPT = """You are a research assistant with access to academic papers.

Tools available:
{tools}

Tool names: {tool_names}

Format:
Thought: what to do
Action: tool name
Action Input: input
Observation: result
... (repeat)
Thought: I have the answer
Final Answer: answer with citations

Question: {input}
{agent_scratchpad}"""


def create_agent_executor() -> AgentExecutor:
    vector_store = VectorStore()
    vector_store.load()
    
    bm25_store = BM25Store()
    bm25_store.load()
    
    metadata_store = get_metadata_store()
    retriever = HybridRetriever(vector_store, bm25_store)
    
    init_tools(retriever, metadata_store)
    
    llm = Ollama(model=settings.ollama_model, base_url=settings.ollama_base_url)
    
    tools = [search_papers, summarize_paper, compare_papers]
    prompt = PromptTemplate.from_template(PROMPT)
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )