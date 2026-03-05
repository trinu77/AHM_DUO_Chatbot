


import warnings
import logging

from langchain_core.messages import HumanMessage
from data import run_anomaly_detection  
from query import create_agent            

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# ── Routing logic ──────────────────────────────────────────────────────────────
def check_for_anomalies(user_input: str) -> bool:
    return any(k in user_input.lower() for k in ["anomaly", "plot", "graph"])


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 Device Data and Anomaly Detection Bot is running. Type 'exit' to quit.\n")
    print("Examples:")
    print("  anomaly IEMA-601-2-0012 between 2025-06-27 21:00 and 23:00")
    print("  plot anomalies for devices in Kolkata")
    print("  graph anomalies for application DRI last 7 days")
    print("  status of IEMA-601-2-0012")
    print("  status of devices in Kolkata")
    print("  max temperature of IEMA-601-2-0012")
    print("  min vibration in DRI")
    print("  device present in DRI")
    print("  list assets in workshop DUO_")
    print("  details of workshop in Kolkata\n")

    print("⏳ Loading agent...")
    graph = create_agent()   # a.py: SQL tools + Gemini
    print("✅ Agent ready.\n")

    thread_id = "1"

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        try:
            if check_for_anomalies(user_input):
                # ── b.py path: SQL + matplotlib only, NO Gemini ───────────
                summary, graph_paths = run_anomaly_detection(user_input)
                print(summary)
                if graph_paths:
                    print("📂 Saved graphs:", graph_paths)

            else:
                # ── a.py path: SQL tools → Gemini formats the answer ──────
                state = {"messages": [HumanMessage(content=user_input)]}
                config = {"configurable": {"thread_id": thread_id, "recursion_limit": 50}}
                out = graph.invoke(state, config=config)
                print("Bot:", out["messages"][-1].content)

        except Exception as e:
            print(f"⚠️ Error: {e}")