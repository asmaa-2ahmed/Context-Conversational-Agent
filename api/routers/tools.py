from fastapi import APIRouter
from pydantic import BaseModel

from tools import context_presence_tool , relevance_checker_tool, message_splitter_tool, get_docs_tool

router = APIRouter(prefix="/tools")

class ContextCheckRequest(BaseModel):
    user_input: str

class RelevanceRequest(BaseModel):
    context: str
    question: str

class SplitRequest(BaseModel):
    message: str

class SearchRequest(BaseModel):
    user_query: str


@router.post("/context-check")
def context_check(req: ContextCheckRequest):
    result = context_presence_tool.invoke({"user_input": req.user_input})
    return {"result": result}


@router.post("/relevance")
def relevance(req: RelevanceRequest):
    result = relevance_checker_tool.invoke({
        "context": req.context,
        "question": req.question
    })
    return {"result": result}


@router.post("/split")
def split(req: SplitRequest):
    result = message_splitter_tool.invoke({"message": req.message})
    return {"result": result}


@router.post("/search")
def search(req: SearchRequest):
    result = get_docs_tool.invoke({"user_query": req.user_query})
    return {"result": result}