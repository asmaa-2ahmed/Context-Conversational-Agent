import os 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import tool
from src import llm , BASE_DIR


context_prompt = PromptTemplate.from_template(
        open(os.path.join(BASE_DIR, "prompts", "context_judge_prompt.txt")).read())
context_chain = context_prompt | llm | StrOutputParser()

@tool
def context_presence_tool(context:str) -> str:
    """Check if the user query already contains enough context or needs external search."""
    result = context_chain.invoke({"input":context})
    return result

# Smoke Test
if __name__ == "__main__":
    print("="*50)
    print ("Testing Context Presence Judge Tool...")

    user_query = """
        What is LangChain used for?
        """
    print(f"User Query: {user_query}")
    print(context_presence_tool.invoke({"context": user_query}))
    print("="*50)
