from langchain_tavily import TavilySearch

from tools import tool
from src import TAVILY_API_KEY

tavily_search = TavilySearch(
    max_results=5,
    topic="general",
    api_key=TAVILY_API_KEY,
)
 
@tool
def get_docs_tool(user_query : str) -> str :
  """ Retrieve relevant information using real search if context is missing. """
  result = tavily_search.invoke(user_query)
  return f"url : {result['results'][0]['url']} \n content : {result['results'][0]['content']}"

# # Smoke Test
# if __name__ == "__main__":
#     print ("="*50)
#     print ("Running smoke test for Web Search Tool...")
#     tool = get_docs_tool
#     query = "What is the capital of Egypt?"
#     print(tool.run(query))