import asyncio
"""Entry point — runs the interview workflow in the terminal."""

import uuid
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

from app.core.graph import build_graph
from langchain_core.messages import AIMessage

QUIT_KEYWORDS = {"quit", "exit", "stop", "end", "bye", "done"}

async def run():
    graph = build_graph()
    config = {"configurable": {"thread_id": f"interview-{uuid.uuid4().hex[:8]}"}}

    # Initialize state with student_name
    initial_state = {
        "student_name": "Jayanth",
        "college": "10000 Coders",
        "course": "Full Stack",
        "messages": []
    }

    # Start the graph
    await graph.ainvoke(initial_state, config=config)
    
    # Helper to print AI messages
    def print_ai_messages(state_snapshot, start_idx):
        all_msgs = state_snapshot.values.get("messages", [])
        displayed = 0
        for i in range(start_idx, len(all_msgs)):
            msg = all_msgs[i]
            if isinstance(msg, AIMessage):
                print(f"\n🤖 AI: {msg.content}")
                displayed += 1
        return len(all_msgs)

    state = await graph.aget_state(config)
    msg_count = print_ai_messages(state, 0)

    while True:
        state = await graph.aget_state(config)
        if not state.next:
            print("\n✅ Interview complete!")
            break

        user_input = (await asyncio.to_thread(input, "\n👤 You: ")).strip()
        
        # Manual quit
        if any(kw in user_input.lower() for kw in QUIT_KEYWORDS):
            print("\n👋 Interview ended manually. Thanks for practicing!")
            break

        # Inject the user input cleanly
        await graph.aupdate_state(config, {"last_user_input": user_input})
        
        print("\n⏳ Thinking...")
        # Continue the graph execution
        await graph.ainvoke(None, config=config)
        
        state = await graph.aget_state(config)
        msg_count = print_ai_messages(state, msg_count)

if __name__ == "__main__":
    asyncio.run(run())
