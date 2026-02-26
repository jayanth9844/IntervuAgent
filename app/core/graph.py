from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.core.state import InterviewState
from app.core.nodes import (
    load_candidate_context,
    intro_hook, quit_call, end_call, goodbye_node,
    identity_router, identity_repeat,
    topic_ask, topic_router, topic_repeat,
    difficulty_ask, difficulty_router, difficulty_repeat,
    prepare_question_pool, ask_question, question_intent_router, question_repeat, evaluate_answer
)

def route_identity(state: InterviewState) -> str:
    if state.intent == "valid":
        return "topic_ask"
    elif state.intent == "not_valid":
        return "quit_call"
    else:
        return "identity_repeat"

def route_topic(state: InterviewState) -> str:
    if state.intent == "topic_valid":
        return "difficulty_ask"
    elif state.intent == "quit":
        return "goodbye_node"
    else:
        return "topic_repeat"

def route_difficulty(state: InterviewState) -> str:
    if state.intent == "difficulty_answer":
        return "prepare_question_pool"
    elif state.intent == "quit":
        return "goodbye_node"
    else:
        return "difficulty_repeat"

def route_question_intent(state: InterviewState) -> str:
    if state.intent == "answer":
        return "evaluate_answer"
    elif state.intent == "quit":
        return "goodbye_node"
    else:
        return "question_repeat"

def check_questions_done(state: InterviewState) -> str:
    if state.question_count >= state.max_questions:
        return "end_call"
    return "ask_question"

def build_graph():
    workflow = StateGraph(InterviewState)

    # Context & Hooks
    workflow.add_node("load_candidate_context", load_candidate_context)
    workflow.add_node("intro_hook", intro_hook)
    
    # Terminal nodes
    workflow.add_node("quit_call", quit_call)
    workflow.add_node("end_call", end_call)
    workflow.add_node("goodbye_node", goodbye_node)

    # Identity
    workflow.add_node("identity_router", identity_router)
    workflow.add_node("identity_repeat", identity_repeat)
    
    # Topic
    workflow.add_node("topic_ask", topic_ask)
    workflow.add_node("topic_router", topic_router)
    workflow.add_node("topic_repeat", topic_repeat)

    # Difficulty
    workflow.add_node("difficulty_ask", difficulty_ask)
    workflow.add_node("difficulty_router", difficulty_router)
    workflow.add_node("difficulty_repeat", difficulty_repeat)

    # Questions
    workflow.add_node("prepare_question_pool", prepare_question_pool)
    workflow.add_node("ask_question", ask_question)
    workflow.add_node("question_intent_router", question_intent_router)
    workflow.add_node("question_repeat", question_repeat)
    workflow.add_node("evaluate_answer", evaluate_answer)

    # Entry
    workflow.set_entry_point("load_candidate_context")
    workflow.add_edge("load_candidate_context", "intro_hook")
    
    # Pre-router edges (where we interrupt) -> router
    # We will interrupt before identity_router, topic_router, difficulty_router, question_intent_router
    workflow.add_edge("intro_hook", "identity_router")
    workflow.add_conditional_edges("identity_router", route_identity)
    workflow.add_edge("identity_repeat", "identity_router")
    
    workflow.add_edge("topic_ask", "topic_router")
    workflow.add_conditional_edges("topic_router", route_topic)
    workflow.add_edge("topic_repeat", "topic_router")
    
    workflow.add_edge("difficulty_ask", "difficulty_router")
    workflow.add_conditional_edges("difficulty_router", route_difficulty)
    workflow.add_edge("difficulty_repeat", "difficulty_router")

    workflow.add_edge("prepare_question_pool", "ask_question")
    workflow.add_edge("ask_question", "question_intent_router")
    workflow.add_conditional_edges("question_intent_router", route_question_intent)
    workflow.add_edge("question_repeat", "question_intent_router")
    
    workflow.add_conditional_edges("evaluate_answer", check_questions_done)

    # Terminal connections
    workflow.add_edge("quit_call", END)
    workflow.add_edge("end_call", END)
    workflow.add_edge("goodbye_node", END)

    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=[
            "identity_router", 
            "topic_router", 
            "difficulty_router", 
            "question_intent_router"
        ]
    )
