from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage,SystemMessage

from langgraph.graph import StateGraph, START, END , add_messages
from langgraph.prebuilt import ToolNode

from src import llm
from tools import context_presence_tool, get_docs_tool, message_splitter_tool, relevance_checker_tool


class AgentState(TypedDict):
  messages : Annotated[Sequence[BaseMessage] , add_messages]


tools = [context_presence_tool , get_docs_tool  , message_splitter_tool , relevance_checker_tool]

SYSTEM_PROMPT = SystemMessage(content="""
    You are a careful AI assistant that must produce verified answers.

      To ensure correctness, follow this strategy:

      - First, check if the question has context using `context_presence_tool`.
      - Second, If it contains both context and a question, use `message_splitter_tool`.
      - Always verify any context using `relevance_checker_tool` before answering.
      - If the context is missing or not relevant, use `get_docs_tool` to retrieve information.

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


# Smoke Test

# if __name__ == "__main__":
#     from agent import run_query     # ← shared utility, verbose=True for CLI output
 
#     queries = [
#         "What is the capital of Japan?",
#         "I am using Python 3.10 and the langchain library. How do I create a custom tool?",
#         "I love football. How do I build a REST API in FastAPI?",
#         "I am learning LangChain. How do I load a PDF document?",
#         "What does AI stand for?",
#     ]
 
#     for query in queries:
#         print(f"\n{'═'*55}")
#         print(f"  QUERY: {query}")
#         print(f"{'═'*55}")
 
#         # Consume the generator — verbose=True prints the CLI trace
#         *_, (final_history, _) = run_query(agent, query, history=[], verbose=True)
 