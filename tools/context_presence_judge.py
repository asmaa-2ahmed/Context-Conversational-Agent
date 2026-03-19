import os 
from langchain_core.output_parsers import StrOutputParser

from tools import tool , PromptTemplate
from src import llm , BASE_DIR


context_prompt = PromptTemplate.from_template(
        open(os.path.join(BASE_DIR, "prompts", "context_judge_prompt.txt")).read())
context_chain = context_prompt | llm | StrOutputParser()

@tool
def context_presence_tool(user_input:str) -> str:
    """Check if the user_input already contains context or needs external search."""
    result = context_chain.invoke({"input": user_input})
    return result

# Smoke Test
if __name__ == "__main__":
    print("="*50)
    print ("Testing Context Presence Judge Tool...")

    user_query = """
        What is LangChain used for?
        """
    print(f"User Query: {user_query}")
    print(context_presence_tool.invoke({"user_input": user_query}))
    print("="*50)
