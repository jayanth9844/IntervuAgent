"""Entry point — runs the interview workflow in the terminal."""

import uuid

from langchain_core.messages import AIMessage, HumanMessage

from app.core.graph import build_graph


# Quick-exit keywords (no LLM call needed)
QUIT_KEYWORDS = {"quit", "exit", "stop", "end", "bye", "done"}


def run():
    graph = build_graph()

    # Unique session ID so each run has its own checkpoint history
    config = {"configurable": {"thread_id": f"interview-{uuid.uuid4().hex[:8]}"}}

    # STEP 1 — Start the graph (runs ask_name, then pauses)
    graph.invoke({"messages": []}, config=config)

    # Print any AI messages produced so far (the greeting)
    state = graph.get_state(config)
    all_msgs = state.values.get("messages", [])
    displayed_count = 0

    for msg in all_msgs:
        if isinstance(msg, AIMessage):
            print(f"\n🤖 AI: {msg.content}\n")
    displayed_count = len(all_msgs)

    # STEP 2 — Main conversation loop
    while True:
        # If the graph has no more nodes to run, we're done
        if not graph.get_state(config).next:
            print("\n✅ Interview complete!")
            break

        user_input = input("👤 You: ").strip()
        if not user_input:
            continue

        # Manual quit (instant, no LLM call) — checks if any keyword appears in input
        if any(kw in user_input.lower() for kw in QUIT_KEYWORDS):
            print("\n👋 Interview ended. Thanks for practicing!")
            break

        print("\n⏳ Thinking...\n")

        # Inject user message and resume the graph
        graph.update_state(config, {"messages": [HumanMessage(content=user_input)]})
        graph.invoke(None, config=config)

        # Print all NEW AI messages (feedback + possibly the next question)
        state = graph.get_state(config)
        all_msgs = state.values.get("messages", [])
        new_msgs = all_msgs[displayed_count:]

        for msg in new_msgs:
            if isinstance(msg, AIMessage):
                print(f"🤖 AI: {msg.content}\n")

        displayed_count = len(all_msgs)

        # Natural end (max questions reached or student said quit in answer)
        if state.values.get("should_end", False):
            print("\n✅ Interview complete!")
            break


if __name__ == "__main__":
    run()
