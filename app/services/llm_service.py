import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

_api_key = os.getenv("OPENAI_API_KEY")

# Main LLM — used for asking questions and evaluating answers
_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    max_tokens=1024,
    api_key=_api_key,
)

# Fast LLM — used for simple extractions (name, topic) with fewer tokens
_llm_fast = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=256,
    api_key=_api_key,
)

def get_llm() -> ChatOpenAI:
    """Return the main LLM for questions and evaluation."""
    return _llm

def get_fast_llm() -> ChatOpenAI:
    """Return the fast LLM for lightweight extractions."""
    return _llm_fast
