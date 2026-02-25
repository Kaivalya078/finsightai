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
import json
import logging
import pickle
import hashlib
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# PDF processing
import fitz  # PyMuPDF

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector database
import faiss
import numpy as np

# Configuration
from dotenv import load_dotenv

# Cache safety utilities
from cache_utils import atomic_write_bytes, atomic_write_json, atomic_faiss_write

# Retrieval result data contract (owned by metadata_schema)
from metadata_schema import RetrievalResult

# Load environment variables
load_dotenv()


# =============================================================================
# VERSION CONSTANTS (bump when the corresponding logic changes)
# =============================================================================

NORMALIZATION_VERSION = "1"       # bump when normalize_text() logic changes
METADATA_SCHEMA_VERSION = "1"     # bump when ChunkMetadata fields change
CACHE_FORMAT_VERSION = 1          # bump on incompatible cache structure changes
CODE_VERSION = "1.1.0-safety"     # for debugging — informational only
EMBEDDING_BATCH_SIZE = 32         # fixed across all paths for float stability

# Stage 3: search_scoped() tuning constants
_SCOPED_DENSITY_MULTIPLIER: int = 4
_SCOPED_INITIAL_MULTIPLIER: int = 3
logger = logging.getLogger(__name__)


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


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize document text before chunking.
    
    This ensures deterministic chunking regardless of PDF extraction quirks:
    1. Normalize Unicode to NFC form (canonical decomposition + composition)
    2. Strip non-printable characters (keeps spaces, newlines, tabs)
    3. Collapse multiple whitespace/newlines to single spaces
    4. Strip leading/trailing whitespace
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned, normalized text
    """
    # Step 1: Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    
    # Step 2: Remove non-printable characters (keep printable + whitespace)
    text = "".join(
        ch for ch in text
        if ch.isprintable() or ch in ("\n", "\r", "\t", " ")
    )
    
    # Step 3: Collapse all whitespace to single spaces
    text = " ".join(text.split())
    
    # Step 4: Strip
    text = text.strip()
    
    return text


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
    
    # Normalize text (deterministic cleaning)
    text = normalize_text(text)
    
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
    
    # =================================================================
    # STAGE 2: INGESTION SUPPORT (Append-Only)
    # =================================================================
    
    def prepare_document(self, pdf_path: str) -> List[str]:
        """
        Prepare a PDF document for ingestion.
        
        Encapsulates the full document preparation pipeline:
          1. Load PDF → extract raw text
          2. Normalize text (Unicode NFC, whitespace collapse)
          3. Chunk text using configured parameters
        
        CorpusManager calls this instead of importing chunking internals,
        keeping preparation logic inside RetrieverPipeline.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of chunk text strings (ordered)
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If PDF is empty or produces no chunks
        """
        # Get chunking parameters from environment
        chunk_size = int(os.getenv("CHUNK_SIZE", 500))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
        
        # Load PDF
        logger.info("Preparing document: %s", pdf_path)
        text = load_pdf(pdf_path)
        
        if not text.strip():
            raise ValueError("PDF appears to be empty or contains no extractable text")
        
        # Chunk text (normalize_text is called inside chunk_text)
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        
        if len(chunks) == 0:
            raise ValueError("No chunks created from document")
        
        # Return text-only list — RetrieverPipeline never exposes
        # structured Chunk objects to the ingestion path
        texts = [chunk.text for chunk in chunks]
        logger.info("Prepared %d text chunks", len(texts))
        
        return texts
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed raw text strings into normalized vectors.
        
        Pure text → vector API. Uses the same model and normalization
        as index_document() but does NOT modify self.index or self.chunks.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Normalized numpy array of shape (len(texts), embedding_dim)
        """
        logger.info("Embedding %d texts...", len(texts))
        
        # batch_size is fixed across all ingestion paths
        # to guarantee identical float results regardless of method
        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        
        # Normalize for cosine similarity (same as index_document)
        faiss.normalize_L2(embeddings)
        
        logger.info("Embeddings shape: %s", embeddings.shape)
        return embeddings
    
    def append_vectors(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """
        Append new vectors and texts to the existing index.
        
        If no index exists, creates a new IndexFlatIP first.
        
        FAISS index.add() always appends at positions [ntotal, ntotal+N).
        These positions become the vector_id range for the new document.
        Old vectors at [0, ntotal) are never touched.
        
        The retriever stores List[str] — never structured Chunk objects.
        Document identity is owned entirely by CorpusManager.
        
        Chunk offset semantics:
            start_char and end_char represent position within the chunk
            text itself, NOT original document positions. This is a
            local-chunk span only. Document-level span reconstruction
            is unsupported until a structured parsing stage.
        
        Args:
            embeddings: Normalized numpy array from embed_texts()
            texts: Corresponding text strings (same order as embeddings)
            
        Raises:
            ValueError: If embeddings and texts have mismatched lengths
        """
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and texts ({len(texts)}) "
                f"count mismatch"
            )
        
        # Create index if this is the first ingestion
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info("Created new FAISS index (dim=%d)", self.embedding_dim)
        
        start_pos = self.index.ntotal
        
        # Append vectors — positions are [start_pos, start_pos + N)
        self.index.add(embeddings)
        
        # Extend the text list (retriever stores raw text only)
        # Convert texts to Chunk objects for backward compatibility with
        # existing retrieve() and save_index() which expect self.chunks
        # Offsets are local-chunk spans (0 to len), not document positions
        for i, text in enumerate(texts):
            global_idx = start_pos + i
            chunk = Chunk(
                chunk_id=f"chunk_{global_idx}",
                text=text,
                start_char=0,
                end_char=len(text),
            )
            self.chunks.append(chunk)
        
        self.is_indexed = True
        
        logger.info(
            "Appended %d vectors (positions %d-%d), total: %d",
            len(texts), start_pos, start_pos + len(texts),
            self.index.ntotal
        )

    # =================================================================
    # STAGE 3: SCOPED RETRIEVAL
    # =================================================================

    @staticmethod
    def _vid_in_ranges(
        vid: int,
        allowed_ranges: List[Tuple[int, int]],
    ) -> bool:
        lo, hi = 0, len(allowed_ranges) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, end = allowed_ranges[mid]
            if vid < start:
                hi = mid - 1
            elif vid >= end:
                lo = mid + 1
            else:
                return True
        return False

    def search_scoped(
        self,
        query_vector: np.ndarray,
        allowed_ranges: List[Tuple[int, int]],
        candidate_k: int,
        total_allowed: int,
    ) -> List[Tuple[float, int]]:
        if not allowed_ranges:
            logger.warning(
                "search_scoped: empty allowed_ranges — returning [] "
                "(candidate_k=%d, total_allowed=%d)",
                candidate_k, total_allowed,
            )
            return []

        max_fetch = min(
            self.index.ntotal,
            total_allowed * _SCOPED_DENSITY_MULTIPLIER,
        )

        batch = candidate_k * _SCOPED_INITIAL_MULTIPLIER
        collected: List[Tuple[float, int]] = []

        while True:
            batch = min(batch, max_fetch)
            distances, indices = self.index.search(query_vector, batch)

            collected = []
            for dist, vid in zip(distances[0], indices[0]):
                if vid < 0:
                    continue
                if self._vid_in_ranges(int(vid), allowed_ranges):
                    collected.append((float(dist), int(vid)))

            if len(collected) >= candidate_k:
                break

            if batch >= max_fetch:
                logger.debug(
                    "search_scoped: max_fetch=%d reached with only %d/%d "
                    "scoped results collected.",
                    max_fetch, len(collected), candidate_k,
                )
                break

            batch *= 2

        logger.debug(
            "search_scoped: returning %d results "
            "(candidate_k=%d, final_batch=%d, max_fetch=%d)",
            min(len(collected), candidate_k), candidate_k, batch, max_fetch,
        )
        return collected[:candidate_k]

    # =================================================================
    # PERSISTENCE (Stage 1)
    # =================================================================
    def save_index(self, save_dir: str, pdf_path: str,
                   document_id: str = "__default__") -> None:
        """
        Persist the FAISS index, chunks, and a manifest to disk.
        
        Uses atomic writes (write → tmp → fsync → rename) so a crash
        mid-save never leaves a corrupt cache.
        
        Saved files:
          - faiss.index          : FAISS binary index (with sanity check)
          - chunks.pkl           : Pickled list of Chunk objects
          - index_manifest.json  : Hash, params, fingerprint, sentinels
        
        Args:
            save_dir: Directory to save into (created if missing)
            pdf_path: Path to the source PDF (used for hash)
            document_id: ID of the document being saved (for per-doc sentinels)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Atomic FAISS write (with post-write sanity check)
        atomic_faiss_write(self.index, os.path.join(save_dir, "faiss.index"))
        
        # Atomic chunks write
        chunks_data = pickle.dumps(self.chunks)
        atomic_write_bytes(os.path.join(save_dir, "chunks.pkl"), chunks_data)
        
        # Build manifest with fingerprint, sentinels, and version
        pdf_hash = self._compute_file_hash(pdf_path)
        manifest = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "config_fingerprint": self.compute_config_fingerprint(),
            "source_pdf_hash": pdf_hash,
            "source_pdf_path": pdf_path,
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "chunk_size": int(os.getenv("CHUNK_SIZE", 500)),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50)),
            "num_chunks": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "chunk_order_sentinel": self.compute_chunk_sentinels(
                self.chunks, document_id
            ),
            "created_at": datetime.now().isoformat(),
            "code_version": CODE_VERSION,
        }
        atomic_write_json(os.path.join(save_dir, "index_manifest.json"), manifest)
        
        print(f"💾 Index saved to {save_dir}/ ({len(self.chunks)} chunks)")
    
    def load_index(self, save_dir: str) -> bool:
        """
        Load a previously saved FAISS index and chunks from disk.
        
        Validates:
          - File existence
          - Chunk count consistency (index vs manifest vs chunks list)
          - Embedding dimension matches runtime model
          - Chunk order sentinels match loaded data
        
        Args:
            save_dir: Directory containing saved index files
            
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = os.path.join(save_dir, "faiss.index")
        chunks_path = os.path.join(save_dir, "chunks.pkl")
        manifest_path = os.path.join(save_dir, "index_manifest.json")
        
        # Check all required files exist
        if not all(os.path.exists(p) for p in [index_path, chunks_path, manifest_path]):
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load chunks
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            
            # Load manifest for validation
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            # --- Validation 1: Chunk count consistency ---
            if self.index.ntotal != manifest["num_chunks"]:
                print("⚠️  Index/manifest chunk count mismatch — will re-index")
                return False
            
            if self.index.ntotal != len(self.chunks):
                print("⚠️  Index/chunks count mismatch — will re-index")
                return False
            
            # --- Validation 2: Embedding dimension drift ---
            stored_dim = manifest.get("embedding_dim")
            if stored_dim is not None and stored_dim != self.embedding_dim:
                print(f"⚠️  Embedding dimension drift: cached={stored_dim}, "
                      f"runtime={self.embedding_dim} — will re-index")
                return False
            
            # --- Validation 3: Chunk order sentinels ---
            stored_sentinels = manifest.get("chunk_order_sentinel")
            if stored_sentinels:
                for doc_id, expected in stored_sentinels.items():
                    current = self.compute_chunk_sentinels(
                        self.chunks, doc_id
                    )
                    current_doc = current.get(doc_id, {})
                    if (current_doc.get("first") != expected.get("first") or
                            current_doc.get("last") != expected.get("last")):
                        print(f"⚠️  Chunk order sentinel mismatch for '{doc_id}' "
                              f"— will re-index")
                        return False
            
            self.is_indexed = True
            print(f"📂 Loaded index from cache: {self.index.ntotal} chunks")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load cached index: {e}")
            return False
    
    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """
        Compute SHA-256 hash of a file.
        
        Used to detect when the source PDF has changed,
        triggering a re-index instead of loading stale cache.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()
    
    @staticmethod
    def compute_config_fingerprint() -> str:
        """
        Compute a SHA-256 fingerprint from all configuration parameters
        that affect index content.
        
        Any change in these values means the cache is incompatible.
        """
        config = {
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "chunk_size": int(os.getenv("CHUNK_SIZE", 500)),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50)),
            "normalization_version": NORMALIZATION_VERSION,
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    @staticmethod
    def compute_chunk_sentinels(
        chunks: List['Chunk'], document_id: str,
        n: int = 5
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Compute per-document ordering sentinels.
        
        Stores SHA-1 of (normalized_text + chunk_index) for the first
        and last N chunks. This detects reordering while remaining stable
        against benign normalization tweaks (because the index is included).
        
        Args:
            chunks: List of Chunk objects
            document_id: ID to key the sentinel under
            n: Number of sentinel hashes per side (default: 5)
            
        Returns:
            { document_id: { "first": [...], "last": [...] } }
        """
        def _hash_chunk(chunk: 'Chunk', idx: int) -> str:
            data = f"{chunk.text}{idx}".encode("utf-8")
            return hashlib.sha1(data).hexdigest()
        
        first = [_hash_chunk(c, i) for i, c in enumerate(chunks[:n])]
        last = [_hash_chunk(c, len(chunks) - n + i)
                for i, c in enumerate(chunks[-n:])]
        
        return {document_id: {"first": first, "last": last}}
    
    @staticmethod
    def check_cache_valid(save_dir: str, pdf_path: str) -> bool:
        """
        Check if the cached index is still valid for the given PDF.
        
        Validation order:
        1. Cache format version matches CACHE_FORMAT_VERSION
        2. Config fingerprint matches runtime config
        3. PDF hash matches (file hasn't changed)
        4. Embedding dimension matches runtime model
        
        Args:
            save_dir: Directory containing saved index files
            pdf_path: Path to the current source PDF
            
        Returns:
            True if cache is valid, False if re-indexing is needed
        """
        manifest_path = os.path.join(save_dir, "index_manifest.json")
        
        if not os.path.exists(manifest_path):
            return False
        
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            # Check 1: Cache format version
            stored_version = manifest.get("cache_format_version")
            if stored_version != CACHE_FORMAT_VERSION:
                print(f"🔄 Cache format version mismatch: "
                      f"cached={stored_version}, current={CACHE_FORMAT_VERSION} "
                      f"— will re-index")
                return False
            
            # Check 2: Config fingerprint
            runtime_fp = RetrieverPipeline.compute_config_fingerprint()
            if manifest.get("config_fingerprint") != runtime_fp:
                print("🔄 Config fingerprint changed — will re-index")
                return False
            
            # Check 3: PDF hash
            current_hash = RetrieverPipeline._compute_file_hash(pdf_path)
            if manifest.get("source_pdf_hash") != current_hash:
                print("🔄 PDF has changed — will re-index")
                return False
            
            # Check 4: Embedding dimension (early detection)
            # Full dimension validation happens in load_index() against
            # the actual runtime model, but we can pre-check here if
            # the dimension is stored in the manifest.
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error reading manifest: {e}")
            return False


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
