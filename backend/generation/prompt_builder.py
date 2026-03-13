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

SYSTEM_PROMPT_TEMPLATE = """You are FinSight AI, an expert financial analyst specializing in Indian companies, annual reports, DRHP, RHP, and financial markets.

Your job is to give the user a clear, helpful, and accurate answer every time.

HOW TO ANSWER:
1. First, look at the CONTEXT below — these are excerpts retrieved from the relevant financial documents.
2. If the context contains the answer or related information, use it to form your answer and cite the chunk IDs (e.g. chunk_42).
3. If the context does not contain enough specific information to answer, still give the best expert answer you can based on your knowledge of the company and sector.
   - In this case, end your response with this exact line on its own: *(Based on combined data review — no specific passage found in the retrieved sections.)*
4. Never refuse to answer or say "the information is not present" — always provide a useful response.
5. Do not fabricate specific numbers (like exact revenue figures) unless they appear in the context.
6. Be concise, professional, and analytical in tone.

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
