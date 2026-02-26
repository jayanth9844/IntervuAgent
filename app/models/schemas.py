from typing import Literal, Optional, List
from pydantic import BaseModel

class IdentityIntent(BaseModel):
    intent: Literal["valid", "not_valid", "repeat", "silence"]

class TopicIntent(BaseModel):
    intent: Literal["topic_valid", "repeat", "quit", "silence"]
    extracted_topic: Optional[str] = None

class DifficultyIntent(BaseModel):
    intent: Literal["difficulty_answer", "repeat", "quit", "unknown"]
    extracted_difficulty: Optional[Literal["beginner", "medium", "hard"]] = None

class QuestionBatch(BaseModel):
    questions: List[str]

class QuestionIntent(BaseModel):
    intent: Literal["answer", "repeat", "quit", "unknown"]

class EvaluationSchema(BaseModel):
    correct: bool
    short_feedback: str
    correction: str | None
