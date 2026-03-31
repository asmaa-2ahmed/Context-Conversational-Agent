# 🧠 Conversational Agent

A smart, verified-answer conversational agent built with **LangGraph**, **LangChain**, **FastAPI**, and **Gradio**. The agent reasons step by step using a set of tools before producing a final answer — making sure every response is grounded and validated.

---

## 📖 Table of Contents

- [What does it do?](#what-does-it-do)
- [Project Structure](#project-structure)
- [How it works](#how-it-works)
- [Setup & Installation](#setup--installation)
- [Running the project](#running-the-project)
- [API Endpoints](#api-endpoints)
- [Test Examples](#test-examples)
- [Switching your LLM](#switching-your-llm)
- [Troubleshooting](#troubleshooting)

---

## 💡 What does it do?

Instead of answering immediately, the agent follows a careful reasoning strategy:

1. **Checks** if the user's message already contains context
2. **Splits** the message into context and question if both exist
3. **Validates** whether the context is actually relevant to the question
4. **Searches the web** if no valid context is found
5. **Answers** only after the information has been verified

This means you get reliable, grounded answers — not hallucinations.

---

## 🗂️ Project Structure

```
Conversational Agent/
│
├── agent/
│   ├── agent_runner.py        # LangGraph agent graph + state definition
│   └── agent_utility.py       # Helper: builds LangChain messages from history
│
├── src/
│   └── config.py              # LLM setup, API keys, base config
│
├── tools/
│   ├── context_presence_judge.py     # Checks if user input has context
│   ├── context_relevance_checker.py  # Validates if context is relevant
│   ├── input_splitter.py             # Splits message into context + question
│   └── web_search.py                 # Tavily-powered web search
│
├── api/
│   ├── __init__.py
│   └── routers/
│       ├── __init__.py
│       ├── health.py          # GET  /health
│       ├── chat.py            # POST /chat
│       └── tools.py           # POST /tools/*
│
├── ui/
│   └── gradio_app.py          # Gradio chat interface
│
├── prompts/
│   └── context_judge_prompt.txt   # Prompt for context presence tool
│
├── main.py                    # FastAPI app entry point
└── requirements.txt
```

---

## ⚙️ How it works

```
User message
     │
     ▼
context_presence_tool        ← Does the message have context?
     │
     ├── YES → message_splitter_tool    ← Split into context + question
     │              │
     │              ▼
     │         relevance_checker_tool   ← Is the context relevant?
     │              │
     │              ├── YES → Answer using context ✅
     │              └── NO  → get_docs_tool (web search) → Answer ✅
     │
     └── NO  → get_docs_tool (web search) → Answer ✅
```

The agent is built as a **LangGraph state machine** — it loops between the LLM and the tool layer until it has enough information to give a final answer.

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/conversational-agent.git
cd conversational-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

> 💡 Get a free Tavily API key at [tavily.com](https://tavily.com)
> 💡 Get a free Groq API key at [console.groq.com](https://console.groq.com) (recommended free alternative to OpenAI)

---

## ▶️ Running the project

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Once running, open these in your browser:

| URL | What you get |
|-----|-------------|
| `http://localhost:8000/ui` | 💬 Gradio chat UI |
| `http://localhost:8000/docs` | 📋 Interactive API docs |
| `http://localhost:8000/health` | ✅ Health check |

> 🔁 `--reload` automatically restarts the server when you change code. Remove it in production.

---

## 📡 API Endpoints

### `GET /health`
Liveness check. Used by deployment platforms (Railway, Render, Docker) to confirm the server is alive.

---

### `POST /chat`
The main endpoint. Send a message and get a verified answer back.

**Request body:**
```json
{
  "message": "your question here",
  "history": []
}
```

**Response:**
```json
{
  "reply": "the agent's answer",
  "history": [...]
}
```

---

### `POST /tools/context-check`
Checks whether a user's message already contains context or needs an external search.

**Request body:**
```json
{
  "user_input": "your message here"
}
```

---

### `POST /tools/relevance`
Checks whether a given context is actually relevant to a question.

**Request body:**
```json
{
  "context": "some context text",
  "question": "the question to check against"
}
```

---

### `POST /tools/split`
Splits a message that contains both context and a question into its two parts.

**Request body:**
```json
{
  "message": "message containing both context and a question"
}
```

---

### `POST /tools/search`
Performs a web search using Tavily and returns the most relevant result.

**Request body:**
```json
{
  "user_query": "what you want to search for"
}
```

---

## 🧪 Test Examples

Copy and paste these directly into `/docs` to test each endpoint.

---

### ✅ Health check
```
GET /health
No body needed
```

---

### 💬 Chat — first message (no history)
```json
{
  "message": "What is LangChain?",
  "history": []
}
```

### 💬 Chat — with conversation history
```json
{
  "message": "Can you give me a code example?",
  "history": [
    {
      "role": "user",
      "content": "What is LangChain?"
    },
    {
      "role": "assistant",
      "content": "LangChain is a framework for building applications powered by language models."
    }
  ]
}
```

### 💬 Chat — message with context included
```json
{
  "message": "I am using Python 3.10 and LangChain 0.3. How do I create a custom tool?",
  "history": []
}
```

---

### 🔍 Context check — message has context
```json
{
  "user_input": "I am using Python 3.10 and LangChain 0.3. How do I create a custom tool?"
}
```

### 🔍 Context check — message has no context
```json
{
  "user_input": "What is the capital of France?"
}
```

---

### ✔️ Relevance check — relevant context
```json
{
  "context": "LangChain is a framework for developing applications powered by language models. It provides tools and abstractions for building LLM-powered apps.",
  "question": "What is LangChain used for?"
}
```

### ✔️ Relevance check — irrelevant context
```json
{
  "context": "I love playing football on weekends with my friends.",
  "question": "How do I install LangChain?"
}
```

---

### ✂️ Message split
```json
{
  "message": "I am working on a LangChain project using Python 3.10. How do I load a PDF document?"
}
```

---

### 🌐 Web search
```json
{
  "user_query": "latest version of LangChain 2025"
}
```

---

## 🔄 Switching your LLM

If you run out of credits or want a free alternative, you can swap the LLM in `src/config.py`.

### Option 1 — Groq (free tier, fast)
```bash
pip install langchain-groq
```
```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="your_groq_api_key"
)
```

### Option 2 — Ollama (fully local, no API key needed)
```bash
pip install langchain-ollama
ollama pull llama3.2
```
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
```

### Option 3 — Keep OpenAI (paid)
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key="your_openai_api_key"
)
```

---

## 🛠️ Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `402 Payment Required` | LLM credits exhausted | Buy credits or switch to Groq/Ollama |
| `422 Unprocessable Content` | Malformed JSON in request | Check your request body format |
| `500 Internal Server Error` | Something crashed server-side | Check the terminal for the full traceback |
| `ModuleNotFoundError` | Missing dependency | Run `pip install -r requirements.txt` |
| Gradio UI not loading | Port conflict or server not running | Make sure uvicorn is running on port 8000 |
| Tavily returns no results | Bad API key or query | Check `TAVILY_API_KEY` in your `.env` file |

---

## 🤝 Built with

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent state machine
- [LangChain](https://github.com/langchain-ai/langchain) — LLM tooling
- [FastAPI](https://fastapi.tiangolo.com) — API layer
- [Gradio](https://gradio.app) — chat UI
- [Tavily](https://tavily.com) — web search

---

> Made with ❤️ — feel free to open an issue or pull request if you find something to improve!