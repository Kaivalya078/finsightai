"""
FinSight AI - Prompt Builder
==============================
Builds the prompt for the OpenAI API from retrieved evidence.

This module handles:
- Converting retrieval results into a formatted context block
- Assembling the system prompt with grounding rules
- Extracting citations from the model's response

Author: FinSight AI Team
Phase: 2 (Generation Layer)
"""

import re
from typing import List, Tuple


# =============================================================================
# SYSTEM PROMPT TEMPLATE
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are FinSight AI, a financial document analysis assistant specialized in Indian financial documents (DRHP, RHP, Annual Reports).

RULES — YOU MUST FOLLOW THESE:
1. ONLY answer using the CONTEXT provided below. Do NOT use any external knowledge.
2. Answer the question based on ALL relevant information available in the context, even if the context does not use the exact same phrasing as the question.
3. If the context contains ANY information related to the question, use it to form your answer.
4. ONLY say "The information is not present in the provided document." if the context is entirely unrelated to the question.
5. If no context chunks are provided below (empty CONTEXT section), respond ONLY with: "The information is not present in the provided documents."
6. When answering, cite the relevant chunk IDs (e.g., chunk_12) that support your answer.
7. Keep your answer concise, accurate, and professional.
8. If only partial information is available, answer with what is present and note the limitation.
9. Do NOT fabricate facts. Every claim must be traceable to a specific chunk.

CONTEXT:
{context}
"""


# =============================================================================
# CONTEXT BUILDER
# =============================================================================

def build_context(results: list) -> Tuple[str, List[str]]:
    """
    Convert retrieval results into a formatted context block.
    
    Takes the Top-K retrieval results and formats them as:
        [chunk_0]: Text of chunk 0...
        [chunk_5]: Text of chunk 5...
        [chunk_12]: Text of chunk 12...
    
    This labeled format helps the LLM:
    - Know which chunk each piece of information comes from
    - Cite specific chunks in its answer
    - Stay grounded to the provided context
    
    Args:
        results: List of retrieval results (each has chunk_id, score, snippet)
        
    Returns:
        Tuple of:
        - context_string: Formatted context block
        - chunk_ids: List of chunk IDs included in the context
    """
    context_parts = []
    chunk_ids = []
    
    for result in results:
        # Format each chunk with its ID as a label
        # The label helps the LLM cite sources
        context_parts.append(f"[{result.chunk_id}]: {result.snippet}")
        chunk_ids.append(result.chunk_id)
    
    # Join all chunks with double newlines for readability
    context_string = "\n\n".join(context_parts)
    
    return context_string, chunk_ids


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt(context: str, question: str) -> Tuple[str, str]:
    """
    Assemble the full prompt for the OpenAI API.
    
    The prompt has two parts:
    1. System prompt: Contains grounding rules + context
    2. User message: The actual question
    
    Why separate system and user messages?
    - OpenAI's API treats them differently
    - System messages set the "persona" and rules
    - User messages are the actual input
    - This separation helps the model follow instructions better
    
    Args:
        context: The formatted context block from build_context()
        question: The user's original question
        
    Returns:
        Tuple of (system_prompt, user_message)
    """
    # Insert the context into the system prompt template
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    
    # The user message is just the question
    user_message = question
    
    return system_prompt, user_message


# =============================================================================
# CITATION EXTRACTOR
# =============================================================================

def extract_citations(answer: str, available_chunk_ids: List[str]) -> List[str]:
    """
    Extract cited chunk IDs from the model's answer.
    
    The model is instructed to cite chunks like: chunk_12, chunk_47
    This function finds all such references in the answer text.
    
    We only return chunk IDs that were actually in the context
    (to prevent hallucinated citations).
    
    Algorithm:
    1. Use regex to find all "chunk_N" patterns in the answer
    2. Filter to only those that were in the provided context
    3. Return unique citations in order of appearance
    
    Args:
        answer: The model's generated answer text
        available_chunk_ids: List of chunk IDs that were in the context
        
    Returns:
        List of cited chunk IDs (only those that exist in context)
    """
    # Find all chunk_N patterns in the answer
    # Pattern: "chunk_" followed by one or more digits
    found = re.findall(r"chunk_\d+", answer)
    
    # Filter to only chunks that were actually in the context
    # This prevents hallucinated citations
    cited = []
    seen = set()
    
    for chunk_id in found:
        if chunk_id in available_chunk_ids and chunk_id not in seen:
            cited.append(chunk_id)
            seen.add(chunk_id)
    
    return cited
