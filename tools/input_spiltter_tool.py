from tools import tool , PromptTemplate
from src import llm

splitter_template = """
separate the question and the context.

Input:
{message}


"""

splitter_prompt = PromptTemplate(
    input_variables=["message"],
    template=splitter_template
)

splitter_chain = splitter_prompt | llm

@tool
def message_splitter_tool (message:str) -> str:
  """ Split the user input into context and question if both exist. """
  result = splitter_chain.invoke({"message":message})
  return result.content

# # Smoke Test

# if __name__ == "__main__":
#     print("="*50)
#     print ("Testing Message Splitter Tool...")
    
#     message = """
#         I am working on a LangChain project and want to add tools.How can I do that?
#     """
#     print(f"Message: {message}")
#     print(message_splitter_tool.invoke({"message": message}))
#     print("="*50)