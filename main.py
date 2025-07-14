from fastapi import FastAPI
from rag import *
from pydantic import BaseModel  


app = FastAPI()

class Query(BaseModel):
    question: str
    top_k: int = 3

@app.post("/ask")
def ask(query: Query):
    try:
        return {"answer": answer_questions(query.question, query.top_k)}
    except Exception as e:
        return {"error": str(e)}