"""All LangGraph node functions for the interview workflow."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.state import InterviewState
from app.core.prompts import (
    EXTRACT_NAME_PROMPT,
    EXTRACT_TOPIC_PROMPT,
    QUESTION_SYSTEM_PROMPT,
    EVAL_SYSTEM_PROMPT,
)
from app.models.schemas import NameExtraction, TopicExtraction, EvalIntentOutput
from app.services.llm_service import get_llm, get_fast_llm


# ─── Greeting & Name ───────────────────────────────────────────────

def ask_name_node(state: InterviewState):
    """Greet the user and ask for their name."""
    return {
        "messages": [
            AIMessage(
                content="Hey there! 👋 I'm your friendly AI interviewer. "
                        "Before we start, what's your name?"
            )
        ],
        "stage": "extract_name",
    }


def extract_name_node(state: InterviewState):
    """Extract the student's first name from their last message."""
    structured_llm = get_fast_llm().with_structured_output(NameExtraction)
    last_msg = state.messages[-1].content
    result = structured_llm.invoke([
        SystemMessage(content=EXTRACT_NAME_PROMPT),
        HumanMessage(content=last_msg),
    ])
    return {
        "student_name": result.name,
        "stage": "ask_topic",
    }


# ─── Topic Selection ───────────────────────────────────────────────

def ask_topic_node(state: InterviewState):
    """Ask the student which topic they want to practice."""
    return {
        "messages": [
            AIMessage(
                content=f"Awesome, {state.student_name}! 🎯 What topic would you "
                        f"like to practice? (e.g. Python, JavaScript, SQL, React, "
                        f"Java, C++, etc.)"
            )
        ],
        "stage": "extract_topic",
    }


def extract_topic_node(state: InterviewState):
    """Extract the technical topic from the student's last message."""
    structured_llm = get_fast_llm().with_structured_output(TopicExtraction)
    last_msg = state.messages[-1].content
    result = structured_llm.invoke([
        SystemMessage(content=EXTRACT_TOPIC_PROMPT),
        HumanMessage(content=last_msg),
    ])
    return {
        "selected_topic": result.topic,
        "stage": "ask_question",
    }


# ─── Questions & Evaluation ───────────────────────────────────────

def ask_question_node(state: InterviewState):
    """Ask a beginner-level interview question on the selected topic."""
    llm = get_llm()

    # Only send the last 4 messages for context (faster, saves tokens)
    recent_msgs = state.messages[-4:] if len(state.messages) > 4 else state.messages

    messages = [
        SystemMessage(
            content=QUESTION_SYSTEM_PROMPT.format(
                student_name=state.student_name,
                selected_topic=state.selected_topic,
            )
        ),
    ]
    messages.extend(recent_msgs)
    messages.append(
        HumanMessage(
            content=f"Ask a beginner-level question about {state.selected_topic}. "
                    f"Question #{state.question_count + 1}"
        )
    )

    response = llm.invoke(messages)
    return {
        "messages": [AIMessage(content=response.content)],
        "stage": "await_answer",
    }


def evaluate_and_check_node(state: InterviewState):
    """Give feedback on the answer and decide whether to continue or end."""
    structured_llm = get_llm().with_structured_output(EvalIntentOutput)
    last_msg = state.messages[-1].content

    # Retry up to 3 times in case of structured-output parse failures
    for attempt in range(3):
        try:
            result = structured_llm.invoke([
                SystemMessage(content=EVAL_SYSTEM_PROMPT),
                HumanMessage(content=last_msg),
            ])
            break
        except Exception:
            if attempt == 2:
                result = EvalIntentOutput(
                    feedback="Nice effort! Let's keep going.",
                    intent="continue",
                )
            continue

    new_count = state.question_count + 1
    should_continue = result.intent == "continue" and new_count < state.max_questions

    return {
        "messages": [AIMessage(content=result.feedback)],
        "intent": result.intent,
        "question_count": new_count,
        "stage": "ask_question" if should_continue else "end",
    }


# ─── End ───────────────────────────────────────────────────────────

def end_node(state: InterviewState):
    """Send a farewell message and mark the interview as finished."""
    return {
        "messages": [
            AIMessage(
                content=f"Great job {state.student_name}! 🎉 That was a solid "
                        f"practice session on {state.selected_topic}. Keep learning "
                        f"and you'll do amazing. Good luck! 🚀"
            )
        ],
        "should_end": True,
    }
