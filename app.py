"""
============================================
FILE 6 OF 8: APP.PY — FastAPI Backend
============================================

📝 PURPOSE:
    This file exposes your RAG pipeline as a REST API.
    
    Why do we need an API?
    - So a web frontend can call your RAG system
    - So a mobile app can call your RAG system
    - So other services can integrate with your RAG system
    
    Without an API, your RAG pipeline can only run locally in Python.
    With an API, ANYONE can use it over HTTP!

🎓 CONCEPTS COVERED:
    1. REST API — The standard way web services communicate
    2. FastAPI — Modern Python web framework for building APIs
    3. Endpoints — URL paths that handle specific requests
    4. Request/Response Models — Structured data validation
    5. File Upload — Handling document uploads
    6. CORS — Cross-Origin Resource Sharing (security)
    7. Background Tasks — Processing files asynchronously

🔗 HOW IT CONNECTS:
    rag_pipeline.py  →  app.py  →  Frontend (Streamlit / Web / Mobile)
                        ^^^^^^^
                        YOU ARE HERE
    
    app.py wraps rag_pipeline.py in HTTP endpoints

💡 WHAT IS A REST API?
    
    REST (Representational State Transfer) is a way for programs 
    to communicate over HTTP (the protocol websites use).
    
    Your API will have these endpoints:
    
    POST /upload     → Upload a document to the RAG system
    POST /query      → Ask a question and get an answer
    GET  /health     → Check if the system is running
    GET  /stats      → Get vector database statistics
    DELETE /reset    → Clear all documents
    
    Each endpoint is like a different "function" that external 
    programs can call over the internet.

💡 INTERVIEW QUESTION: "How would you deploy a RAG system?"
    ANSWER: "Wrap the RAG pipeline in a REST API using FastAPI, 
    containerize with Docker, deploy to Kubernetes/cloud. Use 
    async processing for document ingestion, caching for frequent 
    queries, and monitoring for quality/latency tracking."

============================================
"""

import os
import shutil
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from rag_pipeline import RAGPipeline
from config import config


# ============================================
# PYDANTIC MODELS
# ============================================
# Pydantic models define the SHAPE of data.
# They validate that incoming requests have the right format
# and generate automatic API documentation.
#
# 💡 INTERVIEW CONCEPT: "Data Validation"
# In production, you NEVER trust incoming data.
# Pydantic ensures:
# - Required fields are present
# - Data types are correct
# - Values are within expected ranges
# ============================================

class QueryRequest(BaseModel):
    """
    Request body for the /query endpoint.
    
    Example JSON:
    {
        "question": "What is machine learning?",
        "n_results": 5,
        "source_filter": "research_paper.pdf"
    }
    """
    question: str = Field(..., min_length=1, description="The question to ask")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    source_filter: Optional[str] = Field(default=None, description="Filter by source file")


class QueryResponse(BaseModel):
    """Response body from the /query endpoint."""
    answer: str
    question: str
    sources: list
    retrieval_time: float
    generation_time: float
    total_time: float
    model_used: str
    chunks_retrieved: int


class UploadResponse(BaseModel):
    """Response body from the /upload endpoint."""
    message: str
    filename: str
    documents_loaded: int
    chunks_created: int
    time_seconds: float


class StatsResponse(BaseModel):
    """Response body from the /stats endpoint."""
    collection_name: str
    document_count: int
    persist_directory: str


# ============================================
# APPLICATION SETUP
# ============================================
# We use a "lifespan" context manager to initialize
# the RAG pipeline when the server starts.
#
# 💡 INTERVIEW CONCEPT: "Singleton Resources"
# The RAG pipeline is expensive to initialize (loads models).
# We create it ONCE at startup and share it across all requests.
# ============================================

# Global pipeline instance
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    This runs:
    - BEFORE the server starts accepting requests (startup)
    - AFTER the server stops (shutdown)
    
    We use it to initialize the RAG pipeline when the server boots up.
    """
    global pipeline
    print("\n🚀 Starting RAG Chatbot Server...")
    
    # Create upload directory
    os.makedirs("uploaded_docs", exist_ok=True)
    
    # Initialize pipeline
    try:
        pipeline = RAGPipeline()
        print("✅ Server ready to accept requests!\n")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        print("   The server will start but /query won't work.")
        print("   Make sure your .env file has valid API keys.")
    
    yield  # Server is running
    
    print("\n👋 Shutting down RAG Chatbot Server...")


# ============================================
# CREATE THE FASTAPI APPLICATION
# ============================================
# FastAPI automatically generates:
# 1. API documentation at /docs (Swagger UI)
# 2. Alternative docs at /redoc
# 3. OpenAPI schema at /openapi.json
#
# 💡 This is one of FastAPI's KILLER features.
# You get professional API documentation for FREE!
# ============================================

app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "A Retrieval-Augmented Generation chatbot that answers questions "
        "based on uploaded documents. Upload PDFs/text files and ask questions!"
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ============================================
# CORS MIDDLEWARE
# ============================================
# CORS = Cross-Origin Resource Sharing
#
# By default, browsers BLOCK requests from one origin to another.
# Example: A website at localhost:3000 can't call localhost:8000
# 
# CORS middleware tells the browser: "It's okay, allow these requests."
#
# 💡 INTERVIEW: "What is CORS?"
# ANSWER: "A browser security mechanism that restricts cross-origin 
# HTTP requests. The server must explicitly allow other origins 
# via CORS headers. This prevents malicious websites from making 
# requests to your API using a user's cookies."
#
# In production, replace "*" with specific allowed origins:
# allow_origins=["https://your-frontend.com"]
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Allow all origins (dev only!)
    allow_credentials=True,
    allow_methods=["*"],         # Allow all HTTP methods
    allow_headers=["*"],         # Allow all headers
)


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint — basic welcome message.
    
    Access at: http://localhost:8000/
    """
    return {
        "message": "🤖 RAG Chatbot API is running!",
        "docs": "Visit /docs for interactive API documentation",
        "endpoints": {
            "POST /upload": "Upload a document",
            "POST /query": "Ask a question",
            "GET /health": "Health check",
            "GET /stats": "Database statistics",
            "DELETE /reset": "Clear all documents"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of all system components.
    Useful for monitoring and load balancers.
    
    💡 WHY HEALTH CHECKS?
    In production, load balancers regularly ping /health
    to check if your server is alive. If it returns an error,
    the load balancer stops sending traffic to this server.
    """
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "timestamp": time.time()
    }


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, or DOCX file")
):
    """
    Upload a document to the RAG system.
    
    The document will be:
    1. Saved to disk
    2. Loaded and parsed
    3. Chunked into smaller pieces
    4. Embedded as vectors
    5. Stored in the vector database
    
    After uploading, you can ask questions about this document
    using the /query endpoint.
    
    Supported formats: .pdf, .txt, .docx
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Check your API keys in .env"
        )
    
    # Validate file extension
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    supported = {".pdf", ".txt", ".docx"}
    
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(supported)}"
        )
    
    # Save uploaded file to disk
    upload_path = os.path.join("uploaded_docs", filename)
    
    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Ingest the file into the RAG pipeline
        stats = pipeline.ingest_file(upload_path)
        
        return UploadResponse(
            message=f"✅ Successfully ingested '{filename}'",
            filename=filename,
            documents_loaded=stats["documents_loaded"],
            chunks_created=stats["chunks_created"],
            time_seconds=stats["time_seconds"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Ask a question about uploaded documents.
    
    The system will:
    1. Convert your question to a vector embedding
    2. Search for the most relevant document chunks
    3. Build a prompt with the retrieved context
    4. Generate an answer using the LLM
    5. Return the answer with source citations
    
    💡 TIP: Upload documents first using /upload before querying!
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Check your API keys in .env"
        )
    
    # Check if there are any documents
    stats = pipeline.get_db_stats()
    if stats["document_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet! Upload documents first using /upload"
        )
    
    # Build metadata filter if source specified
    where_filter = None
    if request.source_filter:
        where_filter = {"source": request.source_filter}
    
    try:
        response = pipeline.query(
            question=request.question,
            n_results=request.n_results,
            where=where_filter
        )
        
        # Format sources for response
        sources = []
        for s in response.sources:
            sources.append({
                "text_preview": s["text"][:200] + "...",
                "source": s["metadata"].get("source", "unknown"),
                "page": s["metadata"].get("page", "N/A"),
                "relevance": round(s["relevance_score"], 4)
            })
        
        return QueryResponse(
            answer=response.answer,
            question=request.question,
            sources=sources,
            retrieval_time=response.retrieval_time,
            generation_time=response.generation_time,
            total_time=response.total_time,
            model_used=response.model_used,
            chunks_retrieved=response.num_chunks_retrieved
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/stats", response_model=StatsResponse, tags=["Database"])
async def get_stats():
    """
    Get statistics about the vector database.
    
    Returns the number of stored documents, collection name,
    and storage location.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    stats = pipeline.get_db_stats()
    return StatsResponse(**stats)


@app.delete("/reset", tags=["Database"])
async def reset_database():
    """
    ⚠️ Delete all documents and reset the database.
    
    This is IRREVERSIBLE! All uploaded documents and their
    embeddings will be deleted.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    pipeline.reset()
    
    # Also clean uploaded files
    if os.path.exists("uploaded_docs"):
        shutil.rmtree("uploaded_docs")
        os.makedirs("uploaded_docs")
    
    return {"message": "🔄 Database reset successfully. All documents removed."}


# ============================================
# RUN THE SERVER
# ============================================
# Run with: python app.py
# Or: uvicorn app:app --reload --port 8000
#
# --reload: Auto-restart on code changes (dev only)
# --port 8000: Run on port 8000
#
# Then visit: http://localhost:8000/docs
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🤖 RAG Chatbot API Server")
    print("=" * 60)
    print("\n📚 API Documentation: http://localhost:8000/docs")
    print("🔗 Health Check: http://localhost:8000/health")
    print("\n" + "=" * 60 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes
    )
