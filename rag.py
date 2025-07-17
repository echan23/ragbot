import os
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

import click
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager

from rag import answer_questions, load_vectorstore
from loader import load_documents, parse_pdf, parse_txt
from vector_builder import build_vector_store

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=20)
    include_sources: bool = Field(default=True)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    processing_time: float
    timestamp: str
    top_k: int

class DocumentUpload(BaseModel):
    content: str
    filename: str
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"

class QueryHistory(BaseModel):
    id: str
    question: str
    answer: str
    timestamp: str
    processing_time: float

query_history: List[QueryHistory] = []
uploaded_documents: List[DocumentUpload] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG application...")
    yield
    logger.info("Shutting down RAG application...")

app = FastAPI(
    title="Enhanced RAG API",
    description="A comprehensive RAG system with CLI support and advanced features",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def rate_limit():
    pass

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )

@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: Query, background_tasks: BackgroundTasks, _: None = Depends(rate_limit)):
    start_time = time.time()
    
    try:
        result = answer_questions(query.question, query.top_k)
        
        if query.include_sources:
            answer = result.get('result', '')
            sources = [{"content": doc.page_content, "metadata": doc.metadata} 
                      for doc in result.get('source_documents', [])]
        else:
            answer = result.get('result', '') if isinstance(result, dict) else str(result)
            sources = None
        
        processing_time = time.time() - start_time
        
        response = QueryResponse(
            answer=answer,
            sources=sources,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            top_k=query.top_k
        )
        
        background_tasks.add_task(
            store_query_history,
            query.question,
            answer,
            processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/documents")
async def upload_document(document: DocumentUpload):
    try:
        doc_path = Path(f"data/docs/{document.filename}")
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(document.content)
        
        uploaded_documents.append(document)
        
        return {"message": "Document uploaded successfully", "path": str(doc_path)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )

@app.post("/rebuild-index")
async def rebuild_index(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(build_vector_store)
        return {"message": "Index rebuild started"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error rebuilding index: {str(e)}"
        )

@app.get("/documents")
async def list_documents():
    try:
        docs = load_documents()
        return {"documents": [{"source": doc.metadata.get("source", "unknown"), 
                             "content_length": len(doc.page_content)} for doc in docs]}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )

@app.get("/history")
async def get_query_history(limit: int = 10):
    return {"history": query_history[-limit:]}

@app.delete("/history")
async def clear_history():
    global query_history
    query_history = []
    return {"message": "History cleared"}

async def store_query_history(question: str, answer: str, processing_time: float):
    history_item = QueryHistory(
        id=str(len(query_history) + 1),
        question=question,
        answer=answer,
        timestamp=datetime.now().isoformat(),
        processing_time=processing_time
    )
    query_history.append(history_item)
    
    if len(query_history) > 100:
        query_history.pop(0)

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """RAG CLI Tool"""
    pass

@cli.command()
@click.argument('question')
@click.option('--top-k', default=3, help='Number of top documents to retrieve')
@click.option('--sources', is_flag=True, help='Include source information')
@click.option('--output', type=click.Path(), help='Output file for the answer')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def ask(question, top_k, sources, output, format):
    try:
        start_time = time.time()
        result = answer_questions(question, top_k)
        processing_time = time.time() - start_time
        
        if isinstance(result, dict):
            answer = result.get('result', '')
            source_docs = result.get('source_documents', []) if sources else []
        else:
            answer = str(result)
            source_docs = []
        
        if format == 'json':
            output_data = {
                "question": question,
                "answer": answer,
                "sources": [{"content": doc.page_content, "metadata": doc.metadata} 
                           for doc in source_docs],
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            output_text = json.dumps(output_data, indent=2)
        else:
            output_text = f"Question: {question}\n\nAnswer: {answer}\n\nProcessing time: {processing_time:.2f}s"
            if sources and source_docs:
                output_text += f"\n\nSources:\n"
                for i, doc in enumerate(source_docs, 1):
                    output_text += f"{i}. {doc.metadata.get('source', 'Unknown')}\n"
        
        if output:
            with open(output, 'w') as f:
                f.write(output_text)
            click.echo(f"Answer saved to {output}")
        else:
            click.echo(output_text)
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def upload(file_path):
    try:
        source_path = Path(file_path)
        dest_path = Path("data/docs") / source_path.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if source_path.suffix == ".pdf":
            content = parse_pdf(source_path)
            with open(dest_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write(content)
            click.echo(f"PDF converted and saved to {dest_path.with_suffix('.txt')}")
        else:
            import shutil
            shutil.copy2(source_path, dest_path)
            click.echo(f"Document copied to {dest_path}")
        
        click.echo("Run 'python main.py rebuild-index' to update the search index")
        
    except Exception as e:
        click.echo(f"Error uploading document: {str(e)}", err=True)
        raise click.Abort()

@cli.command()
def rebuild_index():
    try:
        click.echo("Rebuilding vector store index...")
        build_vector_store()
        click.echo("Index rebuilt successfully")
    except Exception as e:
        click.echo(f"Error rebuilding index: {str(e)}", err=True)
        raise click.Abort()

@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to run the server on')
@click.option('--port', default=8000, help='Port to run the server on')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

@cli.command()
@click.option('--limit', default=10, help='Number of recent queries to show')
def history(limit):
    try:
        if not query_history:
            click.echo("No query history available")
            return
            
        click.echo(f"Recent {min(limit, len(query_history))} queries:")
        click.echo("-" * 50)
        
        for i, query in enumerate(query_history[-limit:], 1):
            click.echo(f"{i}. {query.question}")
            click.echo(f"   Answer: {query.answer[:100]}...")
            click.echo(f"   Time: {query.processing_time:.2f}s")
            click.echo()
            
    except Exception as e:
        click.echo(f"Error retrieving history: {str(e)}", err=True)

@cli.command()
def interactive():
    click.echo("Interactive mode- Type 'quit' to exit")
    
    while True:
        try:
            question = click.prompt("Question", type=str)
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            top_k = click.prompt("Top K", default=3, type=int)
            
            start_time = time.time()
            result = answer_questions(question, top_k)
            processing_time = time.time() - start_time
            
            if isinstance(result, dict):
                answer = result.get('result', '')
            else:
                answer = str(result)
            
            click.echo(f"\nAnswer: {answer}")
            click.echo(f"Processing time: {processing_time:.2f}s\n")
            
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
    
    click.echo("Goodbye!")

@cli.command()
def list_docs():
    try:
        docs = load_documents()
        click.echo(f"Found {len(docs)} documents:")
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            content_length = len(doc.page_content)
            click.echo(f"  - {source} ({content_length} characters)")
    except Exception as e:
        click.echo(f"Error listing documents: {str(e)}", err=True)

if __name__ == "__main__":
    cli() 