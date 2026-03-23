import json
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage


# ──────────────────────────────────────────────
#  History builder  (Gradio dicts → LangChain)
# ──────────────────────────────────────────────

def build_lc_messages(history: list, user_message: str) -> list:
    """Convert Gradio role/content history + new message into LangChain messages."""
    lc = []
    for turn in history:
        if turn["role"] == "user":
            lc.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant" and turn["content"]:
            lc.append(AIMessage(content=turn["content"]))
    lc.append(HumanMessage(content=user_message))
    return lc

def run_query(agent, user_message: str, history: list = None, verbose: bool = False):
    if history is None:
        history = []

    history = history + [{"role": "user", "content": user_message}]
    yield history, "⏳ Thinking..."

    trace_lines = []
    final_answer = ""

    lc_messages = build_lc_messages(history[:-1], user_message)

    for step in agent.stream({"messages": lc_messages}):
        node_name, node_state = next(iter(step.items()))

        if verbose:
            print(f"\n📍  [Node]  →  {node_name}")

        for msg in node_state.get("messages", []):

            # ── Agent calling a tool ──
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_names = [tc["name"] for tc in msg.tool_calls]
                trace_lines.append(f"### 🤖 Agent → calling tools\n`{', '.join(tool_names)}`")

                for tc in msg.tool_calls:
                    args_str = json.dumps(tc["args"], indent=2)
                    trace_lines.append(f"**Tool:** `{tc['name']}`\n```json\n{args_str}\n```")

                    if verbose:
                        print(f"\n💭  [Agent decision]  →  Calling {tool_names}")
                        print(f"    Input: {json.dumps(tc['args'], indent=6)}")

                yield history, "\n\n---\n\n".join(trace_lines)

            # ── Tool result ──
            elif isinstance(msg, ToolMessage):
                preview = str(msg.content)[:400]
                ellipsis = "..." if len(str(msg.content)) > 400 else ""
                trace_lines.append(
                    f"### 📦 Tool result: `{msg.name}`\n```\n{preview}{ellipsis}\n```"
                )

                if verbose:
                    print(f"\n📦  [Tool result]  →  {msg.name}")
                    print(f"    Output: {str(msg.content)[:300]}")

                yield history, "\n\n---\n\n".join(trace_lines)

            # ── Final answer ──
            elif isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                final_answer = msg.content

                if verbose:
                    print(f"\n💬  [Agent decision]  →  Responding directly")

    trace_lines.append("### ✅ Done")
    history = history + [{"role": "assistant", "content": final_answer}]

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  FINAL ANSWER:\n  {final_answer}")
        print(f"{'─'*55}")

    yield history, "\n\n---\n\n".join(trace_lines)