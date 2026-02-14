"""
FinSight AI - FastAPI Server
=============================
This is the main entry point for the FinSight AI backend.

Endpoints:
- GET  /health   → Health check
- POST /retrieve → Semantic retrieval from indexed document
- POST /chat     → RAG-based Q&A with grounded answers

Run with:
    uvicorn main:app --reload

Then open:
    http://localhost:8000/docs  (Swagger UI)

Author: FinSight AI Team
Phase: 2.5 (Corpus Architecture)
"""

import os
from typing import List, Optional
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Our retrieval pipeline
from retriever_pipeline import RetrieverPipeline, RetrievalResult

# Phase 2: Generation layer
from openai_client import OpenAIClient
from prompt_builder import build_context, build_prompt, extract_citations

# Phase 2.5: Corpus architecture
from corpus_manager import CorpusManager

# Configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class RetrieveRequest(BaseModel):
    """
    Request schema for the /retrieve endpoint.
    
    Example:
        {"query": "What are the risk factors?"}
    """
    query: str = Field(
        ...,  # ... means required
        description="The question to search for in the document",
        min_length=3,  # At least 3 characters
        examples=["What are the risk factors?"]
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of results to return (default: 5)",
        ge=1,  # Greater than or equal to 1
        le=20  # Less than or equal to 20
    )


class RetrieveResultItem(BaseModel):
    """
    A single retrieval result.
    """
    chunk_id: str = Field(description="Unique identifier for the chunk")
    score: float = Field(description="Similarity score (0-1, higher is better)")
    snippet: str = Field(description="The text content of the chunk")


class RetrieveResponse(BaseModel):
    """
    Response schema for the /retrieve endpoint.
    
    Example:
        {
            "query": "What are the risk factors?",
            "top_k": 5,
            "results": [
                {"chunk_id": "chunk_12", "score": 0.87, "snippet": "..."},
                ...
            ]
        }
    """
    query: str = Field(description="The original query")
    top_k: int = Field(description="Number of results returned")
    results: List[RetrieveResultItem] = Field(description="Retrieved chunks")


class HealthResponse(BaseModel):
    """
    Response schema for the /health endpoint.
    """
    status: str = Field(description="Service status")
    indexed: bool = Field(description="Whether a document has been indexed")
    num_chunks: int = Field(description="Number of chunks in the index")
    generation_ready: bool = Field(description="Whether OpenAI API is configured")


# --- Phase 2: Chat Models ---

class ChatRequest(BaseModel):
    """
    Request schema for the /chat endpoint.
    
    Example:
        {"question": "What are the risk factors?"}
    """
    question: str = Field(
        ...,
        description="The question to ask about the document",
        min_length=3,
        examples=["What are the risk factors?"]
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of evidence chunks to retrieve (default: 5)",
        ge=1,
        le=20
    )


class EvidenceItem(BaseModel):
    """
    A single piece of evidence used to generate the answer.
    """
    chunk_id: str = Field(description="Chunk identifier")
    snippet: str = Field(description="Text content of the chunk")


class ChatResponse(BaseModel):
    """
    Response schema for the /chat endpoint.
    
    Example:
        {
            "answer": "The key risk factors include...",
            "citations": ["chunk_12", "chunk_47"],
            "evidence": [
                {"chunk_id": "chunk_12", "snippet": "..."},
                {"chunk_id": "chunk_47", "snippet": "..."}
            ]
        }
    """
    answer: str = Field(description="The generated answer grounded in document evidence")
    citations: List[str] = Field(description="Chunk IDs cited in the answer")
    evidence: List[EvidenceItem] = Field(description="Evidence chunks used to generate the answer")


# =============================================================================
# GLOBAL STATE
# =============================================================================

# We use global instances that persist across requests
# These are initialized when the server starts (see lifespan below)
pipeline: Optional[RetrieverPipeline] = None
llm_client: Optional[OpenAIClient] = None

# Phase 2.5: Corpus manager wraps pipeline for metadata-aware retrieval
corpus_manager: Optional[CorpusManager] = None


# =============================================================================
# LIFESPAN (Startup/Shutdown Events)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle.
    
    This runs:
    - At startup: Load the embedding model and index the PDF
    - At shutdown: Clean up resources
    
    Using lifespan is the modern FastAPI way (replaces @app.on_event).
    
    Phase 2.5: Creates CorpusManager wrapping the RetrieverPipeline.
    Documents are now ingested through corpus_manager.add_document()
    instead of pipeline.index_document() directly.
    """
    global pipeline, llm_client, corpus_manager
    
    print("\n" + "="*60)
    print("🚀 FinSight AI - Starting Up")
    print("="*60)
    
    # Initialize the retrieval pipeline (Phase 1)
    pipeline = RetrieverPipeline()
    
    # Initialize the OpenAI client (Phase 2)
    llm_client = OpenAIClient()
    
    # Phase 2.5: Wrap pipeline in CorpusManager
    corpus_manager = CorpusManager(pipeline)
    
    # Get PDF path and metadata from environment
    pdf_path = os.getenv("PDF_PATH", "data/sample.pdf")
    default_company = os.getenv("DEFAULT_COMPANY", "demo_company")
    default_doc_type = os.getenv("DEFAULT_DOC_TYPE", "DRHP")
    default_year = os.getenv("DEFAULT_YEAR", "2024")
    
    # Check if PDF exists
    if os.path.exists(pdf_path):
        try:
            # Phase 2.5: Ingest through CorpusManager with metadata
            num_chunks = corpus_manager.add_document(
                pdf_path=pdf_path,
                company=default_company,
                document_type=default_doc_type,
                year=default_year,
            )
            print(f"\n✅ Server ready! Indexed {num_chunks} chunks from {pdf_path}")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not index PDF: {e}")
            print("   The /retrieve endpoint will not work until a PDF is indexed.")
    else:
        print(f"\n⚠️  Warning: PDF not found at {pdf_path}")
        print("   Please place a PDF file at this location and restart the server.")
        print("   Or update PDF_PATH in your .env file.")
    
    print("\n" + "="*60)
    print("📖 Open http://localhost:8000/docs for Swagger UI")
    print("="*60 + "\n")
    
    # Yield control to the application
    yield
    
    # Shutdown (cleanup if needed)
    print("\n👋 Shutting down FinSight AI...")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="FinSight AI",
    description="""
## 🔍 Indian Financial Document Analyzer

**Phase 1: Retrieval Pipeline** + **Phase 2: RAG Generation**

This API provides semantic search and AI-powered Q&A over Indian financial documents (SEBI DRHP/RHP, annual reports).

### How it works:
1. A PDF document is loaded at startup (configured via `PDF_PATH` in `.env`)
2. The system chunks the document and creates embeddings
3. Query `/retrieve` to find relevant passages (evidence only)
4. Query `/chat` for AI-generated answers grounded in the document

### Endpoints:
- **GET /health** — Service health check
- **POST /retrieve** — Semantic retrieval (returns evidence)
- **POST /chat** — RAG Q&A (grounded answer + citations + evidence)
    """,
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware (allows frontend to call this API)
# For Phase 1 we allow all origins (demo purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Check if the service is running and a document is indexed."
)
def health_check():
    """
    Health check endpoint.
    
    Returns:
        - status: "ok" if service is running
        - indexed: True if a document has been indexed
        - num_chunks: Number of chunks in the index
    
    Use this to verify the service is running before the demo!
    """
    global corpus_manager, llm_client
    
    is_indexed = corpus_manager is not None and corpus_manager.is_indexed
    num_chunks = corpus_manager.num_chunks if is_indexed else 0
    gen_ready = llm_client is not None and llm_client.is_configured
    
    return HealthResponse(
        status="ok",
        indexed=is_indexed,
        num_chunks=num_chunks,
        generation_ready=gen_ready
    )


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    tags=["Retrieval"],
    summary="Semantic Retrieval",
    description="Find the most relevant chunks for a given query."
)
def retrieve(request: RetrieveRequest):
    """
    Main retrieval endpoint.
    
    This is the core of FinSight AI Phase 1:
    1. Takes a natural language query
    2. Embeds it using the same model as the document
    3. Finds the top-K most similar chunks
    4. Returns the chunks as "evidence"
    
    Args:
        request: Contains the query and optional top_k parameter
        
    Returns:
        - query: The original query (for reference)
        - top_k: Number of results
        - results: List of chunks with scores and snippets
        
    Raises:
        503: If no document has been indexed yet
    """
    global corpus_manager
    
    # Check if corpus is ready
    if corpus_manager is None or not corpus_manager.is_indexed:
        raise HTTPException(
            status_code=503,
            detail="No document indexed. Please ensure PDF_PATH is set correctly and restart the server."
        )
    
    # Get top_k (use request value or default from env)
    top_k = request.top_k if request.top_k is not None else int(os.getenv("TOP_K", 5))
    
    try:
        # Phase 2.5: Retrieve through CorpusManager (delegates to pipeline)
        results = corpus_manager.search(request.query, top_k=top_k)
        
        # Convert to response format
        result_items = [
            RetrieveResultItem(
                chunk_id=r.chunk_id,
                score=round(r.score, 4),  # Round to 4 decimal places
                snippet=r.snippet
            )
            for r in results
        ]
        
        return RetrieveResponse(
            query=request.query,
            top_k=len(result_items),
            results=result_items
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval error: {str(e)}"
        )


# =============================================================================
# PHASE 2: CHAT ENDPOINT (RAG Generation)
# =============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Generation"],
    summary="RAG Chat",
    description="Ask a question and get an AI-generated answer grounded in the document."
)
def chat(request: ChatRequest):
    """
    Phase 2: RAG Chat endpoint.
    
    This is the core of FinSight AI Phase 2:
    1. Takes a natural language question
    2. Retrieves top-K relevant chunks (reuses Phase 1 pipeline)
    3. Builds a grounded prompt with context + grounding rules
    4. Sends to OpenAI GPT-4o-mini
    5. Returns answer + citations + evidence
    
    The model is strictly instructed to:
    - ONLY answer from the provided context
    - Refuse if information is not present
    - Cite chunk IDs used in the answer
    
    Args:
        request: Contains the question and optional top_k
        
    Returns:
        - answer: The generated answer
        - citations: List of chunk IDs cited
        - evidence: Full evidence chunks used
        
    Raises:
        503: If no document indexed or OpenAI not configured
        500: If generation fails
    """
    global corpus_manager, llm_client
    
    # --- Guard: Check corpus ---
    if corpus_manager is None or not corpus_manager.is_indexed:
        raise HTTPException(
            status_code=503,
            detail="No document indexed. Please ensure PDF_PATH is set correctly and restart the server."
        )
    
    # --- Guard: Check OpenAI client ---
    if llm_client is None or not llm_client.is_configured:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file and restart the server."
        )
    
    # Get top_k
    top_k = request.top_k if request.top_k is not None else int(os.getenv("TOP_K", 5))
    
    try:
        # STEP 1: Retrieve via CorpusManager (delegates to pipeline)
        print(f"\n💬 Chat request: '{request.question[:60]}...'")
        results = corpus_manager.search(request.question, top_k=top_k)
        
        # STEP 2: Build context from retrieved chunks
        context, chunk_ids = build_context(results)
        
        # STEP 3: Build the prompt (system + user message)
        system_prompt, user_message = build_prompt(context, request.question)
        
        # STEP 4: Generate answer via OpenAI
        print(f"🤖 Generating answer with {llm_client.model}...")
        answer = llm_client.generate(system_prompt, user_message)
        
        # STEP 5: Extract citations from the answer
        citations = extract_citations(answer, chunk_ids)
        
        # STEP 6: Build evidence list
        evidence = [
            EvidenceItem(chunk_id=r.chunk_id, snippet=r.snippet)
            for r in results
        ]
        
        print(f"✅ Answer generated ({len(citations)} citations)")
        
        return ChatResponse(
            answer=answer,
            citations=citations,
            evidence=evidence
        )
        
    except ValueError as e:
        # OpenAI configuration error
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        # OpenAI API error (auth, rate limit, etc.)
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation error: {str(e)}"
        )


@app.get(
    "/",
    tags=["System"],
    summary="Root",
    description="Redirect to documentation."
)
def root():
    """
    Root endpoint - provides a friendly welcome message.
    """
    return {
        "message": "Welcome to FinSight AI!",
        "docs": "Visit /docs for the interactive API documentation",
        "health": "Visit /health to check service status"
    }


# =============================================================================
# RUN DIRECTLY (for development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    # In production, use: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True  # Auto-reload on code changes
    )
