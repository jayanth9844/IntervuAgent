from typing import Annotated, Literal

from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class InterviewState(BaseModel):
    """Full interview state — tracks conversation, stage, and progress."""

    messages: Annotated[list[BaseMessage], add_messages] = []

    student_name: str | None = None
    selected_topic: str | None = None

    stage: Literal[
        "ask_name",
        "extract_name",
        "ask_topic",
        "extract_topic",
        "ask_question",
        "await_answer",
        "end",
    ] = "ask_name"

    question_count: int = 0
    max_questions: int = 3

    intent: Literal["continue", "quit"] = "continue"
    should_end: bool = False
