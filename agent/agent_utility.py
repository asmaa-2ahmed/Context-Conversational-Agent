import json
from langchain_core.messages import AIMessage, HumanMessage


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

