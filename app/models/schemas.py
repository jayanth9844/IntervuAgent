from typing import Literal

from pydantic import BaseModel


class NameExtraction(BaseModel):
    """Structured output for extracting a student's first name."""
    name: str


class TopicExtraction(BaseModel):
    """Structured output for extracting a technical topic."""
    topic: str


class EvalIntentOutput(BaseModel):
    """Combined feedback + intent detection from a student's answer."""
    feedback: str
    intent: Literal["continue", "quit"]
