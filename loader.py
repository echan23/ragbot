import fitz
from pathlib import Path
from langchain.schema import Document

def load_documents(folder = "data/docs"):
    docs = []
    for path in Path(folder).glob("*"):
        if path.suffix == ".pdf":
            content = parse_pdf(path)
            docs.append(Document(page_content=content, metadata={"source": str(path)}))
        elif path.suffix == ".txt":
            content = parse_txt(path)
            docs.append(Document(page_content=content, metadata={"Source": str(path)}))

    return docs


def parse_pdf(path):
    text = fitz.open(path)
    return "\n".join([page.get_text() for page in text])

def parse_txt(path):
    with open(path, "r", encoding = "utf-8") as f:
        return f.read()

