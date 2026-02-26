"""
FinSight AI - FastAPI Server
=============================
This is the main entry point for the FinSight AI backend.

Endpoints:
- GET  /health   → Health check
- POST /retrieve → Semantic retrieval from indexed document
- POST /chat     → RAG-based Q&A with grounded answers
- POST /upload   → Upload a PDF for session-scoped retrieval

Run with:
    uvicorn main:app --reload

Then open:
    http://localhost:8000/docs  (Swagger UI)

Author: FinSight AI Team
Stage: 8B (Multi-Corpus Routing)
"""

import os
import time
import shutil
import tempfile
from uuid import uuid4
from typing import List, Optional
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cache_utils import has_leftover_tmp, clean_cache

# Our retrieval pipeline
from retriever_pipeline import RetrieverPipeline
from metadata_schema import RetrievalResult

# Phase 2: Generation layer
from openai_client import OpenAIClient
from prompt_builder import build_context, build_prompt, extract_citations

# Phase 2.5: Corpus architecture
from corpus_manager import CorpusManager

# Stage 6: Query orchestration (parse → plan → execute)
from query_orchestrator import retrieve_context
from query_understanding import parse_query
from search_plan_builder import build_plan

# Stage 8B: Multi-corpus routing
from corpus_router import CorpusRouter

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
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID from /upload to include uploaded document in search",
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
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID from /upload to include uploaded document in search",
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


# --- Stage 8B: Upload Models ---

class UploadResponse(BaseModel):
    """
    Response schema for the /upload endpoint.
    """
    session_id: str = Field(description="Session ID for subsequent queries")
    chunks: int = Field(description="Number of chunks created from the uploaded document")
    company: str = Field(description="Company name used for the upload")
    year: str = Field(description="Year assigned to the upload")
    document_type: str = Field(description="Document type assigned to the upload")


# =============================================================================
# GLOBAL STATE
# =============================================================================

# We use global instances that persist across requests
# These are initialized when the server starts (see lifespan below)
pipeline: Optional[RetrieverPipeline] = None
llm_client: Optional[OpenAIClient] = None

# Phase 2.5: Corpus manager wraps pipeline for metadata-aware retrieval
corpus_manager: Optional[CorpusManager] = None

# Stage 8B: Corpus router for multi-corpus support
corpus_router: Optional[CorpusRouter] = None


# =============================================================================
# LIFESPAN (Startup/Shutdown Events)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle.

    Read-only startup: loads pre-ingested corpus from index_cache/.
    Never ingests documents — use ingest.py or batch_ingest_annual_reports.py.

    Load order:
        1. load_index()
        2. load_registry()
        3. validate_cache_integrity()
        4. init_lookup_index()
    """
    global pipeline, llm_client, corpus_manager, corpus_router

    print("\n" + "="*60)
    print("🚀 FinSight AI - Starting Up (Read-Only Mode)")
    print("="*60)

    start_time = time.time()

    cache_dir = os.getenv("INDEX_CACHE_DIR", "index_cache")

    # Initialize the retrieval pipeline
    pipeline = RetrieverPipeline()

    # Initialize the OpenAI client
    llm_client = OpenAIClient()

    # Wrap pipeline in CorpusManager
    corpus_manager = CorpusManager(pipeline)

    # --- Read-only cache load ---
    if not os.path.exists(cache_dir):
        raise RuntimeError(
            f"Cache directory '{cache_dir}' does not exist. "
            f"Run batch_ingest_annual_reports.py first."
        )

    if has_leftover_tmp(cache_dir):
        clean_cache(cache_dir)

    # Step 1: Load FAISS index
    if not pipeline.load_index(cache_dir):
        raise RuntimeError(
            f"Failed to load FAISS index from '{cache_dir}'. "
            f"Run batch_ingest_annual_reports.py to rebuild."
        )

    # Step 2: Load registry
    if not corpus_manager.load_registry(cache_dir):
        raise RuntimeError(
            f"Failed to load document registry from '{cache_dir}'. "
            f"Run batch_ingest_annual_reports.py to rebuild."
        )

    # Step 3: Validate integrity
    if not corpus_manager.validate_cache_integrity(pipeline.index.ntotal):
        raise RuntimeError(
            "Cache integrity check failed. "
            "Run batch_ingest_annual_reports.py to rebuild."
        )

    # Step 4: Initialize LookupIndex
    corpus_manager.init_lookup_index(cache_dir, pipeline.index.ntotal)

    # Stage 8B: Initialize corpus router with global corpus
    corpus_router = CorpusRouter(corpus_manager)

    elapsed = time.time() - start_time
    print(f"\n✅ Corpus loaded in {elapsed:.2f}s "
          f"({pipeline.index.ntotal} vectors, "
          f"{len(corpus_manager.documents)} documents)")

    print("\n" + "="*60)
    print("📖 Open http://localhost:8000/docs for Swagger UI")
    print("="*60 + "\n")

    # Yield control to the application
    yield

    # Shutdown
    print("\n👋 Shutting down FinSight AI...")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="FinSight AI",
    description="""
## 🔍 Indian Financial Document Analyzer

**Phase 1: Retrieval Pipeline** + **Phase 2: RAG Generation** + **Stage 8B: Upload Support**

This API provides semantic search and AI-powered Q&A over Indian financial documents (SEBI DRHP/RHP, annual reports).

### How it works:
1. A global corpus of NIFTY annual reports is loaded at startup
2. Users can upload additional PDFs for session-scoped retrieval
3. Query `/retrieve` to find relevant passages (evidence only)
4. Query `/chat` for AI-generated answers grounded in the document

### Endpoints:
- **GET /health** — Service health check
- **POST /upload** — Upload a PDF for session-scoped retrieval
- **POST /retrieve** — Semantic retrieval (returns evidence)
- **POST /chat** — RAG Q&A (grounded answer + citations + evidence)
    """,
    version="3.0.0",
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


# =============================================================================
# STAGE 8B: UPLOAD ENDPOINT
# =============================================================================

@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Upload"],
    summary="Upload PDF",
    description="Upload a PDF document for session-scoped retrieval.",
)
def upload_document(
    file: UploadFile = File(..., description="PDF file to upload"),
    company_name: str = Form(..., description="Company name for the document"),
    year: Optional[str] = Form(default=None, description="Fiscal year (e.g. '2024')"),
    document_type: Optional[str] = Form(
        default=None, description="Document type (default: 'Annual_Report')"
    ),
):
    """
    Upload a PDF for session-scoped retrieval.

    Creates an in-memory corpus that is NOT persisted to disk and does NOT
    modify the global NIFTY corpus. The returned session_id can be passed
    to /retrieve and /chat to include this document in search results.

    Args:
        file: The PDF file to upload
        company_name: Company name (used for scoping and metadata)
        year: Fiscal year (defaults to '2024')
        document_type: Filing type (defaults to 'Annual_Report')

    Returns:
        session_id, chunk count, and metadata used
    """
    global corpus_router

    if corpus_router is None:
        raise HTTPException(
            status_code=503,
            detail="Server not ready. Please wait for startup to complete.",
        )

    # Apply defaults
    year = year or "2024"
    document_type = document_type or "Annual_Report"

    # Save uploaded file to a temp location
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="finsight_upload_")
        tmp_path = os.path.join(tmp_dir, file.filename or "upload.pdf")

        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Create isolated pipeline + corpus (shares embedding model weights)
        session_pipeline = RetrieverPipeline()
        session_corpus = CorpusManager(session_pipeline)

        # Ingest into session corpus (in-memory only)
        num_chunks = session_corpus.add_document(
            pdf_path=tmp_path,
            company=company_name,
            document_type=document_type,
            year=year,
        )

        # Generate session ID and register
        session_id = str(uuid4())
        corpus_router.register_session(session_id, session_corpus)

        return UploadResponse(
            session_id=session_id,
            chunks=num_chunks,
            company=company_name,
            year=year,
            document_type=document_type,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload error: {str(e)}",
        )
    finally:
        # Clean up temp file
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# RETRIEVE ENDPOINT
# =============================================================================

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
    global corpus_manager, corpus_router

    # Check if corpus is ready
    if corpus_manager is None or not corpus_manager.is_indexed:
        raise HTTPException(
            status_code=503,
            detail="No document indexed. Please ensure PDF_PATH is set correctly and restart the server."
        )

    # Get top_k (use request value or default from env)
    top_k = request.top_k if request.top_k is not None else int(os.getenv("TOP_K", 5))

    try:
        if request.session_id is not None:
            # Session path: build plan manually, route through corpus_router
            entities = corpus_manager.list_available_entities()
            companies = entities.get("companies", [])
            parsed = parse_query(request.query, companies)
            plan = build_plan(parsed, top_k)
            results = corpus_router.execute_plan(
                plan,
                embed_query=lambda q: pipeline.embed_texts([q]),
                session_id=request.session_id,
            )
        else:
            # Global-only path: use standard orchestrator
            results = retrieve_context(
                raw_query=request.query,
                corpus_manager=corpus_manager,
                embed_query=lambda q: pipeline.embed_texts([q]),
                default_top_k=top_k,
            )

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

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
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
    global corpus_manager, llm_client, corpus_router

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
        if request.session_id is not None:
            # Session path: build plan manually, route through corpus_router
            print(f"\n💬 Chat request (session {request.session_id[:8]}...): "
                  f"'{request.question[:60]}...'")
            entities = corpus_manager.list_available_entities()
            companies = entities.get("companies", [])
            parsed = parse_query(request.question, companies)
            plan = build_plan(parsed, top_k)
            results = corpus_router.execute_plan(
                plan,
                embed_query=lambda q: pipeline.embed_texts([q]),
                session_id=request.session_id,
            )
        else:
            # Global-only path: use standard orchestrator
            print(f"\n💬 Chat request: '{request.question[:60]}...'")
            results = retrieve_context(
                raw_query=request.question,
                corpus_manager=corpus_manager,
                embed_query=lambda q: pipeline.embed_texts([q]),
                default_top_k=top_k,
            )

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

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
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
