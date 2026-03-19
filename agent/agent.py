from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, START, END , add_messages
from langgraph.prebuilt import ToolNode

from src import llm
from tools import context_presence_tool, get_docs_tool, input_spiltter ,relevance_checker_tool


class AgentState(TypedDict):
  messages : Annotated[Sequence[BaseMessage] , add_messages]


tools = [context_presence_tool , get_docs_tool  , input_spiltter , relevance_checker_tool]

SYSTEM_PROMPT = SystemMessage(content="""
    You are a careful AI assistant that must produce verified answers.

      To ensure correctness, follow this strategy:

      - First, analyze the user input. If it contains both context and a question, use `message_splitter_tool`.
      - Always check if the question has enough context using `context_presence_tool`.
      - If the context is missing or incomplete, use `get_docs_tool` to retrieve information.
      - Always verify any context using `relevance_checker_tool` before answering.

      Rules:
      - Do not answer immediately.
      - Prefer multi-step reasoning using tools.
      - Your final answer MUST be based on validated or retrieved context.
""")

def model_response(state:AgentState) -> AgentState :
  model = llm.bind_tools(tools)
  response = model.invoke([SYSTEM_PROMPT] + state["messages"])
  return {"messages": [response]}

# ----------------- Build the Agent Graph -----------------

def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END

graph = StateGraph(AgentState)

graph.add_node("agent", model_response)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    }
)

graph.add_edge("tools", "agent")

agent = graph.compile()



if __name__ == "__main__":
    import json
    from langchain_core.messages import AIMessage, ToolMessage, SystemMessage

    def log(emoji: str, label: str, detail: str = ""):
        print(f"\n{emoji}  [{label}]" + (f"  →  {detail}" if detail else ""))

    queries = [
        "What is the capital of Japan?",
        "I am using Python 3.10 and the langchain library. How do I create a custom tool?",
        "I love football. How do I build a REST API in FastAPI?",
        "I am learning LangChain. How do I load a PDF document?"
        "What does AI stand for?",
    ]

    for query in queries:
        print(f"\n{'═'*55}")
        print(f"  QUERY: {query}")
        print(f"{'═'*55}")

        # Inject system prompt alongside the user query
        for step in agent.stream({"messages": [query]}):
            node_name, node_state = next(iter(step.items()))
            log("📍", "Node", node_name)

            for msg in node_state.get("messages", []):
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        log("💭", "Agent decision", f"Calling {[tc['name'] for tc in msg.tool_calls]}")
                        for tc in msg.tool_calls:
                            print(f"    Input: {json.dumps(tc['args'], indent=6)}")
                    else:
                        log("💬", "Agent decision", "Responding directly")

                elif isinstance(msg, ToolMessage):
                    log("📦", "Tool result", msg.name)
                    print(f"    Output: {str(msg.content)[:300]}")

        final = node_state["messages"][-1]
        print(f"\n{'─'*55}")
        print(f"  FINAL ANSWER:\n  {final.content}")
        print(f"{'─'*55}")
