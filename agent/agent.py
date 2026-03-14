from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, START, END , add_messages
from langgraph.prebuilt import ToolNode

from src import llm
from tools import context_presence_tool, get_docs_tool  


class AgentState(TypedDict):
  messages : Annotated[Sequence[BaseMessage] , add_messages]


def should_continue (state:AgentState):
  """ this fuction decides even it should search for context or respond to the user query based on the presence of context in messages."""
  last_message = state['messages'][-1]
  if not last_message.tool_calls:
    return "context_provided"
  return "context_missing"

tools = [context_presence_tool , get_docs_tool]

def model_response(state:AgentState) -> AgentState :
  SYSTEM_PROMPT = SystemMessage(content="""You are a helpful assistant.
      You MUST always call `context_presence_tool` first on every user query, no exceptions.
      Never answer directly without calling it first.""")
  model = llm.bind_tools(tools)
  response = model.invoke([SYSTEM_PROMPT] + state["messages"])
  return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("agent", model_response)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "context_missing": "tools",
        "context_provided": END,
    }
)

graph.add_edge("tools", "agent")
agent = graph.compile()
agent

if __name__ == "__main__":
    import json
    from langchain_core.messages import AIMessage, ToolMessage, SystemMessage

    def log(emoji: str, label: str, detail: str = ""):
        print(f"\n{emoji}  [{label}]" + (f"  →  {detail}" if detail else ""))

    queries = [
        "What is LangChain used for?",
        "I am building a LangChain app with tools. How do I add memory to it?"
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
