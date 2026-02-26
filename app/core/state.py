from typing import Annotated, Literal, Optional, List
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class InterviewState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    
    student_name: str = ""
    college: str = ""
    course: str = ""
    
    topic: Optional[str] = None
    difficulty: Optional[Literal["beginner", "medium", "hard"]] = None
    
    question_pool: List[str] = Field(default_factory=list)
    asked_questions: List[str] = Field(default_factory=list)
    current_question: Optional[str] = None
    
    question_count: int = 0
    max_questions: int = 3
    
    last_user_input: Optional[str] = None
    intent: Optional[str] = None
    
    short_feedback: Optional[str] = None
    correction: Optional[str] = None
    correct: Optional[bool] = None
