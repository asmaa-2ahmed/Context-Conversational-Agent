from tools import tool , PromptTemplate
from src import llm

relevant_template = """

check the following context given from message_splitter_tool and check if it is relevant to the question or not and return "relevant" or "not_relevant".

Context:
{context}

Question:
{question}
"""

relevant_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=relevant_template
)

relevance_chain = relevant_prompt | llm

@tool
def relevance_checker_tool(context:str,question:str) -> str:
  """ Checks if the provided context is relevant """
  result = relevance_chain.invoke({"context":context,"question":question})
  return result.content


# # Smoke Test
# if __name__ == "__main__":
#     print("="*50)
#     print ("Testing Context Relevance Checker Tool...")
    
#     query = [
#        {"context": "LangChain is a framework for developing applications powered by language models. It provides tools and abstractions to help developers build applications that can understand and generate human language.",
#         "question" : "What is LangChain used for?"},
#         {"context": "I am working on a LangChain project and want to add tools." ,
#           "question": "How old is me?"}
#     ]

#     for item in query:
#         context = item["context"]
#         question = item["question"]
#         print(f"Context: {context}")
#         print(f"Question: {question}")
#         print(relevance_checker_tool.invoke({"context": context, "question": question}))
#         print("="*50)