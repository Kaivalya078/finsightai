"""
FinSight AI - Retriever Pipeline
=================================
This module implements the core retrieval logic:
1. Load PDF and extract text
2. Chunk text into smaller pieces
3. Generate embeddings for each chunk
4. Store embeddings in FAISS vector database
5. Retrieve top-K similar chunks for a query

Author: FinSight AI Team
Phase: 1 (Retrieval Only)
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# PDF processing
import fitz  # PyMuPDF

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector database
import faiss
import numpy as np

# Configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Chunk:
    """
    Represents a single chunk of text from the document.
    
    Attributes:
        chunk_id: Unique identifier (e.g., "chunk_0", "chunk_1")
        text: The actual text content
        start_char: Starting character position in original document
        end_char: Ending character position in original document
    """
    chunk_id: str
    text: str
    start_char: int
    end_char: int


@dataclass
class RetrievalResult:
    """
    Represents a single retrieval result.
    
    Attributes:
        chunk_id: Which chunk was retrieved
        score: Similarity score (higher = more similar)
        snippet: The text content of the chunk
    """
    chunk_id: str
    score: float
    snippet: str


# =============================================================================
# STAGE 1: PDF LOADING
# =============================================================================

def load_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    
    How it works:
    1. Open PDF with PyMuPDF (fitz)
    2. Iterate through each page
    3. Extract text from each page
    4. Concatenate all text with page separators
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Full text content of the PDF
        
    Raises:
        FileNotFoundError: If PDF doesn't exist
        Exception: If PDF is corrupted or password-protected
    """
    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Open and extract text
    text_parts = []
    
    # fitz.open() handles the PDF parsing
    with fitz.open(pdf_path) as doc:
        print(f"📄 Loading PDF: {pdf_path}")
        print(f"   Pages: {len(doc)}")
        
        for page_num, page in enumerate(doc):
            # get_text() extracts text from the page
            page_text = page.get_text()
            text_parts.append(page_text)
            
    full_text = "\n\n".join(text_parts)
    print(f"   Total characters: {len(full_text):,}")
    
    return full_text


# =============================================================================
# STAGE 2: CHUNKING
# =============================================================================

def chunk_text(
    text: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50
) -> List[Chunk]:
    """
    Split text into overlapping chunks.
    
    Why chunk?
    - Embedding models have token limits
    - Smaller chunks = more precise retrieval
    - Overlap ensures context isn't lost at boundaries
    
    Algorithm:
    1. Start at position 0
    2. Take chunk_size characters
    3. Move forward by (chunk_size - overlap) characters
    4. Repeat until end of text
    
    Args:
        text: The full document text
        chunk_size: Maximum characters per chunk (default: 500)
        chunk_overlap: Overlapping characters between chunks (default: 50)
        
    Returns:
        List of Chunk objects
    """
    chunks = []
    start = 0
    chunk_index = 0
    
    # Clean the text (remove excessive whitespace)
    text = " ".join(text.split())
    
    while start < len(text):
        # Calculate end position
        end = min(start + chunk_size, len(text))
        
        # Extract the chunk text
        chunk_text = text[start:end]
        
        # Create Chunk object with metadata
        chunk = Chunk(
            chunk_id=f"chunk_{chunk_index}",
            text=chunk_text,
            start_char=start,
            end_char=end
        )
        chunks.append(chunk)
        
        # Move to next chunk (with overlap)
        # If we're at the end, break to avoid infinite loop
        if end >= len(text):
            break
            
        start = end - chunk_overlap
        chunk_index += 1
    
    print(f"📦 Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    return chunks


# =============================================================================
# STAGE 3, 4, 5: RETRIEVER PIPELINE CLASS
# =============================================================================

class RetrieverPipeline:
    """
    Main retrieval pipeline that manages embeddings and vector search.
    
    This class handles:
    - Loading the embedding model
    - Generating embeddings for document chunks
    - Building and querying the FAISS index
    
    Usage:
        pipeline = RetrieverPipeline()
        pipeline.index_document("path/to/document.pdf")
        results = pipeline.retrieve("What are the risk factors?", top_k=5)
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the retriever pipeline.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default: Uses EMBEDDING_MODEL from .env or 'all-MiniLM-L6-v2'
        """
        # Get model name from environment or use default
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        print(f"🤖 Loading embedding model: {model_name}")
        print("   (This may take a minute on first run as the model downloads...)")
        
        # Load the embedding model
        # SentenceTransformer automatically downloads and caches the model
        self.model = SentenceTransformer(model_name)
        
        # Get embedding dimension (needed for FAISS)
        # For 'all-MiniLM-L6-v2', this is 384
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"   Embedding dimension: {self.embedding_dim}")
        
        # Initialize storage
        self.chunks: List[Chunk] = []  # Store chunks for retrieval
        self.index: Optional[faiss.IndexFlatIP] = None  # FAISS index
        
        # Track if we've indexed a document
        self.is_indexed = False
        
    def index_document(self, pdf_path: str) -> int:
        """
        Process a PDF document and build the search index.
        
        Steps:
        1. Load PDF → extract text
        2. Chunk text → create smaller pieces
        3. Embed chunks → convert to vectors
        4. Build FAISS index → enable fast search
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of chunks indexed
        """
        # Get chunking parameters from environment
        chunk_size = int(os.getenv("CHUNK_SIZE", 500))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
        
        # STAGE 1: Load PDF
        print("\n" + "="*50)
        print("STAGE 1: Loading PDF")
        print("="*50)
        text = load_pdf(pdf_path)
        
        if not text.strip():
            raise ValueError("PDF appears to be empty or contains no extractable text")
        
        # STAGE 2: Chunk text
        print("\n" + "="*50)
        print("STAGE 2: Chunking Text")
        print("="*50)
        self.chunks = chunk_text(text, chunk_size, chunk_overlap)
        
        if len(self.chunks) == 0:
            raise ValueError("No chunks created from document")
        
        # STAGE 3: Generate embeddings
        print("\n" + "="*50)
        print("STAGE 3: Generating Embeddings")
        print("="*50)
        print(f"🔢 Embedding {len(self.chunks)} chunks...")
        
        # Extract just the text from each chunk for embedding
        chunk_texts = [chunk.text for chunk in self.chunks]
        
        # Generate embeddings for all chunks at once (batch processing = faster)
        # This returns a numpy array of shape (num_chunks, embedding_dim)
        embeddings = self.model.encode(
            chunk_texts,
            show_progress_bar=True,  # Nice progress bar
            convert_to_numpy=True     # Return numpy array for FAISS
        )
        
        print(f"   Embeddings shape: {embeddings.shape}")
        
        # STAGE 4: Build FAISS index
        print("\n" + "="*50)
        print("STAGE 4: Building FAISS Index")
        print("="*50)
        
        # Normalize embeddings for cosine similarity
        # FAISS IndexFlatIP uses inner product, which equals cosine similarity
        # when vectors are normalized
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        # IndexFlatIP = Flat index with Inner Product (cosine similarity when normalized)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        print(f"✅ Indexed {self.index.ntotal} chunks successfully!")
        
        self.is_indexed = True
        return len(self.chunks)
    
    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        Retrieve the most relevant chunks for a query.
        
        How it works:
        1. Embed the query using the same model
        2. Search FAISS index for top-K similar vectors
        3. Return chunks with similarity scores
        
        Args:
            query: The user's question
            top_k: Number of results to return (default: from .env or 5)
            
        Returns:
            List of RetrievalResult objects, sorted by score (highest first)
        """
        if not self.is_indexed:
            raise RuntimeError("No document indexed. Call index_document() first.")
        
        # Get top_k from environment if not specified
        if top_k is None:
            top_k = int(os.getenv("TOP_K", 5))
        
        # Ensure we don't request more results than we have chunks
        top_k = min(top_k, len(self.chunks))
        
        print(f"\n🔍 Retrieving top {top_k} chunks for: '{query[:50]}...'")
        
        # STAGE 5: Embed the query
        query_embedding = self.model.encode(
            [query],  # Model expects a list
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        # Returns:
        #   - distances: similarity scores (shape: 1 x top_k)
        #   - indices: chunk indices (shape: 1 x top_k)
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
            # Get the chunk
            chunk = self.chunks[idx]
            
            # Create result object
            result = RetrievalResult(
                chunk_id=chunk.chunk_id,
                score=float(score),  # Convert numpy float to Python float
                snippet=chunk.text
            )
            results.append(result)
            
            print(f"   [{i+1}] {chunk.chunk_id}: score={score:.4f}")
        
        return results

    def get_chunks(self) -> List['Chunk']:
        """
        Expose the chunk list for external use.
        
        Used by CorpusManager to map vector positions to metadata.
        
        Returns:
            List of Chunk objects (same as self.chunks)
        
        Added in Phase 2.5 — does not modify any existing behavior.
        """
        return self.chunks


# =============================================================================
# TESTING (Only runs when script is executed directly)
# =============================================================================

if __name__ == "__main__":
    """
    Quick test of the retrieval pipeline.
    Run with: python retriever_pipeline.py
    """
    print("\n" + "="*60)
    print("FinSight AI - Retriever Pipeline Test")
    print("="*60)
    
    # Get PDF path from environment
    pdf_path = os.getenv("PDF_PATH", "data/sample.pdf")
    
    # Check if sample PDF exists
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  Sample PDF not found at: {pdf_path}")
        print("   Please place a PDF file at this location and try again.")
        print("   Or update PDF_PATH in your .env file.")
        exit(1)
    
    # Initialize pipeline
    pipeline = RetrieverPipeline()
    
    # Index the document
    num_chunks = pipeline.index_document(pdf_path)
    print(f"\n✅ Successfully indexed {num_chunks} chunks!")
    
    # Test retrieval
    test_queries = [
        "What are the risk factors?",
        "What is the company's revenue?",
        "Who are the promoters?"
    ]
    
    print("\n" + "="*60)
    print("Testing Retrieval")
    print("="*60)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        results = pipeline.retrieve(query, top_k=3)
        
        print(f"   Found {len(results)} results:")
        for r in results:
            # Show first 100 chars of snippet
            snippet_preview = r.snippet[:100].replace("\n", " ") + "..."
            print(f"   - {r.chunk_id} (score: {r.score:.3f}): {snippet_preview}")
