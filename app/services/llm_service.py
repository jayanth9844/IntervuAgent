import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_api_key = os.getenv("GOOGLE_API_KEY")

# Main LLM — used for asking questions and evaluating answers
_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    max_output_tokens=1024,
    google_api_key=_api_key,
)

# Fast LLM — used for simple extractions (name, topic) with fewer tokens
_llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    max_output_tokens=256,
    google_api_key=_api_key,
)


def get_llm() -> ChatGoogleGenerativeAI:
    """Return the main LLM for questions and evaluation."""
    return _llm


def get_fast_llm() -> ChatGoogleGenerativeAI:
    """Return the fast LLM for lightweight extractions."""
    return _llm_fast
