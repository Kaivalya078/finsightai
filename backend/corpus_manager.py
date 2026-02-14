"""
FinSight AI - Corpus Manager
================================
Orchestration layer between the API and the retrieval engine.

Responsibilities:
- Document registry: track ingested documents with full metadata
- Identity mapping: vector_id → ChunkMetadata
- Search orchestration: delegate FAISS search, post-filter, return results
- Entity resolution: call resolver, use result as filter
- Corpus introspection: list companies, doc types, years
- Document lifecycle: version management, active/superseded marking

Design principle:
    CorpusManager is a THIN WRAPPER. It delegates all heavy work
    (embedding, FAISS search) to RetrieverPipeline. It only adds
    metadata tracking and post-filtering on top.

Phase 2.5: Works with a single document. All filters default to no-op.
Phase 3:   add_document() called multiple times → multi-company corpus.

Author: FinSight AI Team
Phase: 2.5 (Corpus Architecture)
"""

from typing import List, Dict, Optional
from retriever_pipeline import RetrieverPipeline, RetrievalResult
from metadata_schema import (
    ChunkMetadata,
    SearchFilters,
    DocumentRecord,
    create_default_metadata,
)
from company_resolver import resolve_company


# =============================================================================
# CORPUS MANAGER
# =============================================================================

class CorpusManager:
    """
    Central orchestrator for the corpus-based RAG architecture.
    
    This class sits between main.py and RetrieverPipeline:
    
        main.py → CorpusManager → RetrieverPipeline → FAISS
    
    It adds three capabilities that RetrieverPipeline doesn't have:
    1. Metadata tracking (which company, doc type, authority, etc.)
    2. Post-filter search results by metadata
    3. Document registry (what's been ingested, versions, lifecycle)
    
    Usage:
        corpus = CorpusManager(retriever_pipeline)
        corpus.add_document("data/sample.pdf", company="TCS", ...)
        results = corpus.search("What are the risk factors?")
    """
    
    def __init__(self, retriever: RetrieverPipeline):
        """
        Initialize the Corpus Manager.
        
        Args:
            retriever: An already-initialized RetrieverPipeline instance.
                      CorpusManager does NOT create its own — it wraps the existing one.
        """
        self.retriever = retriever
        
        # Document registry: document_id → DocumentRecord
        self.documents: Dict[str, DocumentRecord] = {}
        
        # Metadata mapping: vector_id (int) → ChunkMetadata
        # This is the core identity resolution table.
        # FAISS returns vector positions → we look up metadata here.
        self.chunk_metadata: Dict[int, ChunkMetadata] = {}
        
        # Track total vectors indexed (for vector_id_start calculation)
        self._total_vectors: int = 0
        
        print("📚 Corpus Manager initialized")
    
    # =========================================================================
    # DOCUMENT INGESTION
    # =========================================================================
    
    def add_document(
        self,
        pdf_path: str,
        company: str = "demo_company",
        document_type: str = "DRHP",
        year: str = "2024",
        source_type: str = "pdf",
        authority: int = 100,
        source_class: str = "official",
    ) -> int:
        """
        Ingest a document into the corpus with metadata.
        
        Steps:
        1. Delegate PDF loading, chunking, embedding to RetrieverPipeline
        2. Create metadata for each chunk
        3. Register the document in the corpus
        
        Phase 2.5: Called once at startup with defaults.
        Phase 3:   Called for each new document/company.
        
        Args:
            pdf_path: Path to the PDF file
            company: Company identifier
            document_type: Filing type (DRHP, Annual_Report, etc.)
            year: Fiscal year/period
            source_type: How document was obtained (pdf, html, user_upload)
            authority: Trust level (100=official, 50=user, 30=derived)
            source_class: official, user, or derived
        
        Returns:
            Number of chunks indexed from this document
        """
        # Build the document label and ID
        version = self._get_next_version(company, document_type, year)
        document_label = f"{company}_{document_type}_{year}_v{version}"
        document_id = document_label
        
        print(f"\n📄 Adding document to corpus: {document_id}")
        
        # Record where this document's vectors start in the FAISS index
        vector_id_start = self._total_vectors
        
        # --- Delegate to RetrieverPipeline (unchanged Phase 1 logic) ---
        num_chunks = self.retriever.index_document(pdf_path)
        
        # Record where this document's vectors end
        vector_id_end = vector_id_start + num_chunks
        
        # --- Create metadata for each chunk ---
        chunks = self.retriever.get_chunks()
        
        for i in range(vector_id_start, vector_id_end):
            # Chunk index within this document
            local_index = i - vector_id_start
            display_chunk_id = f"chunk_{local_index}"
            
            metadata = create_default_metadata(
                vector_id=i,
                display_chunk_id=display_chunk_id,
                company=company,
                document_type=document_type,
                year=year,
            )
            # Override non-default fields
            metadata.document_label = document_label
            metadata.source_type = source_type
            metadata.authority = authority
            metadata.source_class = source_class
            metadata.version = version
            
            self.chunk_metadata[i] = metadata
        
        # --- Register the document ---
        # Use the first chunk's metadata as the template
        template_meta = self.chunk_metadata[vector_id_start]
        
        record = DocumentRecord(
            document_id=document_id,
            pdf_path=pdf_path,
            chunk_count=num_chunks,
            vector_id_start=vector_id_start,
            vector_id_end=vector_id_end,
            metadata=template_meta,
        )
        self.documents[document_id] = record
        
        # Update total vector count
        self._total_vectors = vector_id_end
        
        print(f"✅ Corpus updated: {document_id} ({num_chunks} chunks, vectors {vector_id_start}-{vector_id_end})")
        
        return num_chunks
    
    # =========================================================================
    # SEARCH
    # =========================================================================
    
    def search(
        self,
        query: str,
        top_k: int = None,
        filters: SearchFilters = None,
    ) -> List[RetrievalResult]:
        """
        Search the corpus with optional metadata filtering.
        
        Flow:
        1. Determine retrieval mode (semantic vs constrained)
        2. Delegate FAISS search to RetrieverPipeline
        3. Post-filter results by metadata (if filters provided)
        4. Return results in same format as Phase 2
        
        Phase 2.5: filters=None → no filtering → identical to Phase 2.
        
        Args:
            query: The user's natural language question
            top_k: Number of results to return
            filters: Optional metadata filters
        
        Returns:
            List of RetrievalResult (same format as RetrieverPipeline.retrieve)
        """
        if filters is None:
            filters = SearchFilters()
        
        # --- Determine retrieval mode ---
        is_constrained = not filters.is_empty()
        
        if is_constrained:
            # Constrained mode: oversample to compensate for post-filter loss
            # Phase 2.5: this branch won't execute (no filters active)
            oversample_factor = 3  # Phase 3: make adaptive
            search_k = top_k * oversample_factor if top_k else None
        else:
            # Semantic mode: no oversampling needed
            search_k = top_k
        
        # --- Delegate to RetrieverPipeline ---
        results = self.retriever.retrieve(query, top_k=search_k)
        
        # --- Post-filter by metadata ---
        if is_constrained:
            results = self._apply_filters(results, filters)
            # Trim to requested top_k after filtering
            if top_k is not None:
                results = results[:top_k]
        
        return results
    
    # =========================================================================
    # CORPUS INTROSPECTION
    # =========================================================================
    
    def list_available_entities(self) -> dict:
        """
        List what's in the corpus.
        
        Returns a summary of registered documents:
        - companies: list of unique company names
        - document_types: list of unique filing types
        - years: list of unique fiscal periods
        - total_chunks: total chunks across all documents
        - documents: list of document_ids
        
        Returns:
            Dictionary with corpus summary
        """
        companies = set()
        doc_types = set()
        years = set()
        
        for doc in self.documents.values():
            companies.add(doc.metadata.company)
            doc_types.add(doc.metadata.document_type)
            years.add(doc.metadata.year)
        
        return {
            "companies": sorted(companies),
            "document_types": sorted(doc_types),
            "years": sorted(years),
            "total_chunks": self._total_vectors,
            "documents": list(self.documents.keys()),
        }
    
    @property
    def is_indexed(self) -> bool:
        """Whether any document has been successfully indexed."""
        return self.retriever.is_indexed
    
    @property
    def num_chunks(self) -> int:
        """Total number of chunks in the corpus."""
        return self._total_vectors
    
    # =========================================================================
    # INTERNAL: Filtering
    # =========================================================================
    
    def _apply_filters(
        self,
        results: List[RetrievalResult],
        filters: SearchFilters,
    ) -> List[RetrievalResult]:
        """
        Post-filter retrieval results by metadata.
        
        Phase 2.5: This method exists but is never called (no filters active).
        Phase 3:   Filters company, doc_type, year, authority, etc.
        
        Args:
            results: Raw retrieval results from FAISS
            filters: Active filter criteria
        
        Returns:
            Filtered results (subset of input)
        """
        filtered = []
        
        for result in results:
            # Look up metadata for this chunk
            # Extract vector_id from the chunk index in our metadata map
            # Note: chunk_id from retriever is "chunk_N" — we need the FAISS index
            vector_id = self._resolve_vector_id(result.chunk_id)
            
            if vector_id is None:
                # If we can't find metadata, include the result (safe default)
                filtered.append(result)
                continue
            
            meta = self.chunk_metadata.get(vector_id)
            if meta is None:
                filtered.append(result)
                continue
            
            # Apply each filter
            if not self._matches_filter(meta, filters):
                continue
            
            filtered.append(result)
        
        return filtered
    
    def _matches_filter(self, meta: ChunkMetadata, filters: SearchFilters) -> bool:
        """
        Check if a chunk's metadata passes all active filters.
        
        Returns True if the chunk should be included in results.
        """
        # Company filter
        if filters.company is not None:
            companies = filters.company if isinstance(filters.company, list) else [filters.company]
            if meta.company not in companies:
                return False
        
        # Document type filter
        if filters.document_type is not None:
            doc_types = filters.document_type if isinstance(filters.document_type, list) else [filters.document_type]
            if meta.document_type not in doc_types:
                return False
        
        # Year filter
        if filters.year is not None:
            years = filters.year if isinstance(filters.year, list) else [filters.year]
            if meta.year not in years:
                return False
        
        # Source type filter
        if filters.source_type is not None:
            if meta.source_type != filters.source_type:
                return False
        
        # Active only filter
        if filters.active_only and not meta.is_active:
            return False
        
        # Temporal scope filter
        if filters.temporal_scope is not None:
            if meta.temporal_scope != filters.temporal_scope:
                return False
        
        # Authority floor
        if filters.min_authority is not None:
            if meta.authority < filters.min_authority:
                return False
        
        # Source class filter
        if filters.source_class is not None:
            if meta.source_class != filters.source_class:
                return False
        
        # Numeric content filter
        if filters.require_numeric:
            if not meta.contains_numeric:
                return False
        
        return True
    
    def _resolve_vector_id(self, chunk_id: str) -> Optional[int]:
        """
        Map a display chunk_id back to a FAISS vector_id.
        
        Phase 2.5: Simple extraction since chunk_id = "chunk_N"
        and vector_id = N for a single document.
        
        Phase 3: Will need a reverse lookup table for multi-document.
        """
        try:
            # Extract the numeric index from "chunk_N"
            index = int(chunk_id.replace("chunk_", ""))
            return index
        except (ValueError, AttributeError):
            return None
    
    # =========================================================================
    # INTERNAL: Versioning
    # =========================================================================
    
    def _get_next_version(self, company: str, document_type: str, year: str) -> int:
        """
        Determine the next version number for a document.
        
        If TCS_DRHP_2024_v1 exists, returns 2.
        If no previous version exists, returns 1.
        """
        version = 1
        while True:
            doc_id = f"{company}_{document_type}_{year}_v{version}"
            if doc_id not in self.documents:
                return version
            version += 1
