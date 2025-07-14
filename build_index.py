from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loader import load_documents
import pickle

def build_vector_store():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    if len(chunks) == 0:
        print("No chunks to index, exiting")
        return

    embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    with open("vectorstore/faiss_index.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
    print("FAISS index built")

if __name__ == "__main__":
    build_vector_store()