from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.core.state import InterviewState
from app.models.schemas import (
    IdentityIntent, TopicIntent, DifficultyIntent, QuestionBatch, QuestionIntent, EvaluationSchema
)
from app.core.prompts import (
    build_system_prompt,
    TOPIC_ROUTER_SYSTEM_PROMPT,
    DIFFICULTY_ROUTER_SYSTEM_PROMPT,
    QUESTION_POOL_SYSTEM_PROMPT,
    QUESTION_ROUTER_SYSTEM_PROMPT,
    EVALUATION_SYSTEM_PROMPT
)
from app.services.llm_service import get_llm, get_fast_llm

def load_candidate_context(state: InterviewState) -> dict:
    return {"messages": [SystemMessage(content=build_system_prompt(
        state.student_name, state.college, state.course
    ))]}

def intro_hook(state: InterviewState) -> dict:
    response = f"Hello! This is the interview agent. Am I speaking to {state.student_name}?"
    return {"messages": [AIMessage(content=response)]}

def quit_call(state: InterviewState) -> dict:
    msg = "Sorry about that. I'll end the call here. Have a good day."
    return {"messages": [AIMessage(content=msg)]}

def end_call(state: InterviewState) -> dict:
    msg = "Thank you! Have a great day, Goodbye!"
    return {"messages": [AIMessage(content=msg)]}

def goodbye_node(state: InterviewState) -> dict:
    msg = "Thanks for your time. You'll receive detailed feedback soon. All the best."
    return {"messages": [AIMessage(content=msg)]}

# Identity
def identity_router(state: InterviewState) -> dict:
    user_text = state.last_user_input or ""
    structured_llm = get_fast_llm().with_structured_output(IdentityIntent)
    
    # Try multiple times in case of structured output failure
    for attempt in range(3):
        try:
            result = structured_llm.invoke(
                f"""
                The assistant asked: "Am I speaking to {state.student_name}?"
                Classify the user's reply.
                Rules:
                - If clearly yes -> valid
                - If clearly no -> not_valid
                - If asking to repeat or unclear -> repeat
                - If empty or meaningless -> silence
                User reply: {user_text}
                """
            )
            return {"intent": result.intent, "messages": [HumanMessage(content=user_text)]}
        except Exception as e:
            print(f"Error in identity_router: {e}")
            pass
    return {"intent": "repeat", "messages": [HumanMessage(content=user_text)]}

def identity_repeat(state: InterviewState) -> dict:
    if state.intent == "silence":
        msg = "Hello? Can you hear me?"
    else:
        msg = f"Sure. I just wanted to confirm — am I speaking to {state.student_name}?"
    return {"messages": [AIMessage(content=msg)]}


# Topic Flow
def topic_ask(state: InterviewState) -> dict:
    msg = "Which topic would you like to be interviewed on today?"
    return {"messages": [AIMessage(content=msg)]}

def topic_router(state: InterviewState) -> dict:
    user_text = (state.last_user_input or "").strip()
    if user_text == "":
        return {"intent": "silence", "messages": [HumanMessage(content=user_text)]}
    
    topic_llm = get_fast_llm().with_structured_output(TopicIntent)
    for attempt in range(3):
        try:
            result = topic_llm.invoke([
                SystemMessage(content=TOPIC_ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=f'The assistant asked:\n"Which topic would you like to be interviewed on today?"\n\nUser reply:\n"{user_text}"')
            ])
            return {
                "intent": result.intent, 
                "topic": result.extracted_topic or state.topic,
                "messages": [HumanMessage(content=user_text)]
            }
        except Exception as e:
            print(f"Error in topic_router: {e}")
            pass
    return {"intent": "repeat", "messages": [HumanMessage(content=user_text)]}

def topic_repeat(state: InterviewState) -> dict:
    if state.intent == "silence":
        msg = "Hello? Can you hear me?"
    else:
        msg = "No problem. Which topic would you like to go with?"
    return {"messages": [AIMessage(content=msg)]}


# Difficulty Flow
def difficulty_ask(state: InterviewState) -> dict:
    msg = "What difficulty level would you prefer — beginner, medium, or hard?"
    return {"messages": [AIMessage(content=msg)]}

def difficulty_router(state: InterviewState) -> dict:
    user_text = (state.last_user_input or "").strip()
    if user_text == "":
        return {"intent": "silence", "messages": [HumanMessage(content=user_text)]}
    
    difficulty_llm = get_fast_llm().with_structured_output(DifficultyIntent)
    for attempt in range(3):
        try:
            result = difficulty_llm.invoke([
                SystemMessage(content=DIFFICULTY_ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=f'The assistant asked:\n"What difficulty level would you prefer — beginner, medium, or hard?"\n\nUser reply:\n"{user_text}"')
            ])
            if result.intent == "unknown":
                return {"intent": "repeat", "messages": [HumanMessage(content=user_text)]}
            return {
                "intent": result.intent, 
                "difficulty": result.extracted_difficulty or state.difficulty,
                "messages": [HumanMessage(content=user_text)]
            }
        except Exception as e:
            print(f"Error in difficulty_router: {e}")
            pass
    return {"intent": "repeat", "messages": [HumanMessage(content=user_text)]}

def difficulty_repeat(state: InterviewState) -> dict:
    if state.intent == "silence":
        msg = "Hello? Can you hear me?"
    else:
        msg = "Sure. Would you like beginner, medium, or hard?"
    return {"messages": [AIMessage(content=msg)]}


# Question Flow
def prepare_question_pool(state: InterviewState) -> dict:
    question_gen_llm = get_llm().with_structured_output(QuestionBatch)
    prompt = QUESTION_POOL_SYSTEM_PROMPT.format(topic=state.topic, difficulty=state.difficulty)
    result = question_gen_llm.invoke(prompt)
    return {"question_pool": result.questions, "asked_questions": [], "question_count": 0}

def ask_question(state: InterviewState) -> dict:
    remaining_questions = [q for q in state.question_pool if q not in state.asked_questions]
    if not remaining_questions:
        msg = "Looks like we're out of questions."
        return {"messages": [AIMessage(content=msg)]}
    
    question = remaining_questions[0]
    return {
        "current_question": question, 
        "asked_questions": state.asked_questions + [question], 
        "messages": [AIMessage(content=question)]
    }

def question_intent_router(state: InterviewState) -> dict:
    user_text = (state.last_user_input or "").strip()
    if user_text == "":
        return {"intent": "silence", "messages": [HumanMessage(content=user_text)]}
        
    question_router_llm = get_fast_llm().with_structured_output(QuestionIntent)
    for attempt in range(3):
        try:
            result = question_router_llm.invoke([
                SystemMessage(content=QUESTION_ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=f'Question: {state.current_question}\n\nUser reply:\n"{user_text}"')
            ])
            if result.intent == "unknown":
                return {"intent": "answer", "messages": [HumanMessage(content=user_text)]}
            return {"intent": result.intent, "messages": [HumanMessage(content=user_text)]}
        except Exception as e:
            print(f"Error in question_intent_router: {e}")
            pass
    return {"intent": "answer", "messages": [HumanMessage(content=user_text)]}

def question_repeat(state: InterviewState) -> dict:
    if state.intent == "silence":
        msg = "Hello? Can you hear me?"
    else:
        msg = f"No problem. {state.current_question}"
    return {"messages": [AIMessage(content=msg)]}

def evaluate_answer(state: InterviewState) -> dict:
    evaluation_llm = get_fast_llm().with_structured_output(EvaluationSchema)
    
    for attempt in range(3):
        try:
            result = evaluation_llm.invoke([
                SystemMessage(content=EVALUATION_SYSTEM_PROMPT),
                HumanMessage(content=f"Question: {state.current_question}\nAnswer: {state.last_user_input}")
            ])
            break
        except Exception as e:
            print(f"Error in evaluate_answer: {e}")
            if attempt == 2:
                # Fallback to a neutral valid response if parsing fails
                result = EvaluationSchema(
                    correct=True,
                    short_feedback="Nice effort! Let's keep going.",
                    correction=None
                )
            continue
    
    if result.correct:
        message = result.short_feedback
    else:
        message = f"{result.short_feedback} A better way is: {result.correction}"
        
    return {
        "correct": result.correct, 
        "short_feedback": result.short_feedback, 
        "correction": result.correction,
        "messages": [AIMessage(content=message)],
        "question_count": state.question_count + 1
    }
