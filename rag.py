import os
import pickle
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

load_dotenv()

def load_vectorstore():
    with open("vectorstore/faiss_index.pkl", "rb") as f:
        return pickle.load(f)
    
def answer_questions(query: str, k: int) -> str:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    try:
        qa_chain = RetrievalQA.from_chain_type(
        llm = Ollama(model="llama2", temperature=0.7),
        retriever = retriever,
        return_source_documents=True
    )
    except Exception as e:
        print("Error during query: ", e)
        return "Error occured connecting to OpenAI"

    return qa_chain.invoke(query)