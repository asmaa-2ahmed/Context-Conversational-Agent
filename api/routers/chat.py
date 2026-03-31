from fastapi import APIRouter
from pydantic import BaseModel

from langchain_core.messages import AIMessage
from agent import agent, build_lc_messages

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    history: list = []


@router.post("/chat")
def chat(req: ChatRequest):
    lc_messages = build_lc_messages(req.history, req.message)

    final_reply = ""

    for step in agent.stream({"messages": lc_messages}):
        _, node_state = next(iter(step.items()))

        for msg in node_state.get("messages", []):
            # grab only the final text response, skip tool calls
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_reply = msg.content

    return {"reply": final_reply}
