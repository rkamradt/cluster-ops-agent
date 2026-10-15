#!/usr/bin/env python3
"""CLI for local development.

Usage:
  # Start kubectl proxy in another terminal first:
  #   kubectl proxy --port=8001
  #
  # Then:
  export K8S_PROXY_URL=http://localhost:8001
  export OLLAMA_URL=http://<your-ollama-host>:11434   # or tunnel to cluster
  python ask.py "list pods in home"
  python ask.py "is anything broken in the k8s-dashboard namespace?"
"""
import asyncio
import sys

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import build_graph


async def main(question: str) -> None:
    # No checkpointer for CLI — stateless, single-shot
    graph = build_graph(checkpointer=None)

    config = {"configurable": {"thread_id": "cli"}}
    initial = {
        "messages": [HumanMessage(content=question)],
        "step_count": 0,
        "retry_count": 0,
        "validation_status": "",
    }

    print(f"\nQuestion: {question}\n{'─' * 60}")

    async for event in graph.astream(initial, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            labels = {
                "agent": "thinking",
                "validate": "validating",
                "tools": "querying cluster",
                "diagnose": "diagnosing",
            }
            print(f"[{labels.get(node_name, node_name)}]", end=" ", flush=True)

            for msg in reversed(node_output.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    print()  # newline after label
                    print(msg.content)
                    break
            else:
                print()  # newline after label even if no AI message

    print("─" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ask.py '<question>'")
        sys.exit(1)
    asyncio.run(main(" ".join(sys.argv[1:])))
