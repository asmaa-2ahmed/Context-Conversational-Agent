# from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
import dotenv
import os

dotenv.load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# llm = OllamaLLM(model="llama3.2")

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:groq",
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
    temperature=0.7
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
