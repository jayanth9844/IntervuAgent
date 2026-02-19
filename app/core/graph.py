"""Build and compile the LangGraph interview workflow."""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.core.state import InterviewState
from app.core.nodes import (
    ask_name_node,
    extract_name_node,
    ask_topic_node,
    extract_topic_node,
    ask_question_node,
    evaluate_and_check_node,
    end_node,
)


def route_by_stage(state: InterviewState) -> str:
    """Route to the next node based on the current stage."""
    return state.stage


def build_graph():
    """Construct and compile the full interview graph.

    Uses `interrupt_before` on extraction/evaluation nodes so
    the caller can inject user input via `graph.update_state()`
    before those nodes execute.
    """
    workflow = StateGraph(InterviewState)

    # --- Register nodes ---
    workflow.add_node("ask_name", ask_name_node)
    workflow.add_node("extract_name", extract_name_node)
    workflow.add_node("ask_topic", ask_topic_node)
    workflow.add_node("extract_topic", extract_topic_node)
    workflow.add_node("ask_question", ask_question_node)
    workflow.add_node("evaluate_answer", evaluate_and_check_node)
    workflow.add_node("end", end_node)

    # --- Entry point ---
    workflow.set_entry_point("ask_name")

    # --- Edges (stage-based routing) ---
    workflow.add_conditional_edges(
        "ask_name", route_by_stage, {"extract_name": "extract_name"}
    )
    workflow.add_conditional_edges(
        "extract_name", route_by_stage, {"ask_topic": "ask_topic"}
    )
    workflow.add_conditional_edges(
        "ask_topic", route_by_stage, {"extract_topic": "extract_topic"}
    )
    workflow.add_conditional_edges(
        "extract_topic", route_by_stage, {"ask_question": "ask_question"}
    )
    workflow.add_conditional_edges(
        "ask_question", route_by_stage, {"await_answer": "evaluate_answer"}
    )
    workflow.add_conditional_edges(
        "evaluate_answer",
        route_by_stage,
        {
            "ask_question": "ask_question",
            "end": "end",
        },
    )
    workflow.add_edge("end", END)

    # --- Compile with checkpointing & interrupt points ---
    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["extract_name", "extract_topic", "evaluate_answer"],
    )
