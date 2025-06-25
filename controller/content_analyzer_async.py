"""
content_analyzer_async.py

Asynchronous Content Analysis for Podcast Debate Pipeline

- Concurrent GPT-4.1-mini processing for document analysis
- Rate limiting and retry logic with tenacity
- Optimized for production speed and reliability
"""

import re
import json
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from controller.config import get_openai_api_key, get_openai_model, MAX_TOKENS

# Rate limiting for OpenAI API calls
MAX_CONCURRENT_API_CALLS = 3
API_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

def chunk_text_for_gpt(text: str, max_tokens: int = 4096, overlap: int = 200) -> List[str]:
    """
    Splits text into chunks suitable for GPT-4.1-mini API, preserving paragraph boundaries and context.
    """
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4

    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if chunks and overlap_chars > 0:
                overlap_text = current_chunk[-overlap_chars:]
                current_chunk = overlap_text + para + "\n\n"
            else:
                current_chunk = para + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def make_openai_request(
    session: aiohttp.ClientSession,
    prompt: str,
    response_schema: Dict[str, Any],
    model: Optional[str] = None,
    max_tokens: int = 512
) -> Dict[str, Any]:
    """
    Make async OpenAI API request with retry logic and rate limiting.
    """
    async with API_SEMAPHORE:
        if model is None:
            model = get_openai_model()
        
        headers = {
            "Authorization": f"Bearer {get_openai_api_key()}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "response_format": {
                "type": "json_schema",
                "json_schema": response_schema
            }
        }
        
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                if not content:
                    raise ValueError("No content returned from OpenAI API")
                return json.loads(content)
            else:
                error_text = await response.text()
                raise aiohttp.ClientError(f"OpenAI API failed: {response.status} {error_text}")

async def analyze_content_structure_async(text: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes document structure using GPT-4.1-mini with async processing.
    """
    # Extract potential headings using regex
    headings = []
    for line in text.splitlines():
        if re.match(r"^\s*([A-Z][A-Z\s\d\.\-:]{3,}|[0-9]+\.\s+.+)$", line.strip()):
            headings.append(line.strip())
    
    prompt = (
        "Analyze the following document content and identify its structure. "
        "Identify sections with headings and subheadings if present. "
        "Infer the hierarchy and organization of the document. "
        "Document content:\n"
        "-----\n"
        f"{text[:6000]}\n"
        "-----"
    )
    
    schema = {
        "name": "document_structure",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "subheadings": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["heading", "subheadings"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["sections"],
            "additionalProperties": False
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            result = await make_openai_request(session, prompt, schema, model)
            return result
    except Exception:
        # Fallback: return headings found by regex
        return {"sections": [{"heading": h, "subheadings": []} for h in headings]}

async def extract_key_points_async(text: str, model: Optional[str] = None) -> List[str]:
    """
    Extracts key points from text using GPT-4.1-mini with async processing.
    """
    prompt = (
        "Read the following document content and extract the most important key points, arguments, and findings. "
        "Focus on main arguments, conclusions, and significant facts. "
        "Document content:\n"
        "-----\n"
        f"{text[:6000]}\n"
        "-----"
    )
    
    schema = {
        "name": "key_points",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["key_points"],
            "additionalProperties": False
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            result = await make_openai_request(session, prompt, schema, model)
            return result.get("key_points", [])
    except Exception:
        return []

async def generate_summary_async(text: str, summary_type: str = "medium", model: Optional[str] = None) -> str:
    """
    Generates document summary using GPT-4.1-mini with async processing.
    """
    length_guidance = {
        "short": "2-3 sentences",
        "medium": "1-2 paragraphs", 
        "long": "3-4 paragraphs"
    }
    
    prompt = (
        f"Summarize the following document in {length_guidance.get(summary_type, 'medium length')}. "
        "Focus on the main themes, arguments, and conclusions. "
        "Document content:\n"
        "-----\n"
        f"{text[:6000]}\n"
        "-----"
    )
    
    schema = {
        "name": "document_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"}
            },
            "required": ["summary"],
            "additionalProperties": False
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            result = await make_openai_request(session, prompt, schema, model)
            return result.get("summary", "")
    except Exception:
        return ""

async def identify_discussion_topics_async(text: str, model: Optional[str] = None) -> List[str]:
    """
    Identifies discussion topics using GPT-4.1-mini with async processing.
    """
    prompt = (
        "Identify the most interesting and debatable topics from this document that would make for engaging podcast discussion. "
        "Focus on controversial points, different perspectives, implications, and thought-provoking questions. "
        "Document content:\n"
        "-----\n"
        f"{text[:6000]}\n"
        "-----"
    )
    
    schema = {
        "name": "discussion_topics",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["topics"],
            "additionalProperties": False
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            result = await make_openai_request(session, prompt, schema, model)
            return result.get("topics", [])
    except Exception:
        return []

# Main async processing functions for multiple chunks

async def analyze_document_structure_async(text_chunks: List[str]) -> Dict[str, Any]:
    """
    Analyze document structure from multiple text chunks concurrently.
    """
    if not text_chunks:
        return {"sections": []}
    
    # Process chunks concurrently
    tasks = [analyze_content_structure_async(chunk) for chunk in text_chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Merge results
    all_sections = []
    for result in results:
        if isinstance(result, dict) and "sections" in result:
            all_sections.extend(result["sections"])
    
    return {"sections": all_sections}

async def extract_document_key_points_async(text_chunks: List[str]) -> List[str]:
    """
    Extract key points from multiple text chunks concurrently.
    """
    if not text_chunks:
        return []
    
    # Process chunks concurrently
    tasks = [extract_key_points_async(chunk) for chunk in text_chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Merge and deduplicate results
    all_key_points = []
    for result in results:
        if isinstance(result, list):
            all_key_points.extend(result)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_points = []
    for point in all_key_points:
        if point not in seen:
            seen.add(point)
            unique_points.append(point)
    
    return unique_points

async def generate_document_summary_async(text_chunks: List[str], summary_type: str = "medium") -> str:
    """
    Generate document summary from multiple text chunks.
    Uses first chunk for summary to maintain coherence.
    """
    if not text_chunks:
        return ""
    
    # Use the first (typically largest) chunk for summary
    return await generate_summary_async(text_chunks[0], summary_type)

async def identify_document_discussion_topics_async(text_chunks: List[str]) -> List[str]:
    """
    Identify discussion topics from multiple text chunks concurrently.
    """
    if not text_chunks:
        return []
    
    # Process chunks concurrently
    tasks = [identify_discussion_topics_async(chunk) for chunk in text_chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Merge and deduplicate results
    all_topics = []
    for result in results:
        if isinstance(result, list):
            all_topics.extend(result)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_topics = []
    for topic in all_topics:
        if topic not in seen:
            seen.add(topic)
            unique_topics.append(topic)
    
    return unique_topics

async def analyze_content_completely_async(text_chunks: List[str]) -> Dict[str, Any]:
    """
    Run all content analysis tasks concurrently for maximum speed.
    """
    if not text_chunks:
        return {
            "structure": {"sections": []},
            "key_points": [],
            "summary": "",
            "topics": []
        }
    
    # Run all analysis tasks concurrently
    structure_task = analyze_document_structure_async(text_chunks)
    key_points_task = extract_document_key_points_async(text_chunks)
    summary_task = generate_document_summary_async(text_chunks, "medium")
    topics_task = identify_document_discussion_topics_async(text_chunks)
    
    # Wait for all tasks to complete
    structure, key_points, summary, topics = await asyncio.gather(
        structure_task,
        key_points_task,
        summary_task,
        topics_task,
        return_exceptions=True
    )
    
    # Handle exceptions gracefully
    if isinstance(structure, Exception):
        structure = {"sections": []}
    if isinstance(key_points, Exception):
        key_points = []
    if isinstance(summary, Exception):
        summary = ""
    if isinstance(topics, Exception):
        topics = []
    
    return {
        "structure": structure,
        "key_points": key_points,
        "summary": summary,
        "topics": topics
    }

# Synchronous wrappers for backwards compatibility
def analyze_document_structure(text_chunks: List[str]) -> Dict[str, Any]:
    """Sync wrapper for analyze_document_structure_async"""
    return asyncio.run(analyze_document_structure_async(text_chunks))

def extract_document_key_points(text_chunks: List[str]) -> List[str]:
    """Sync wrapper for extract_document_key_points_async"""
    return asyncio.run(extract_document_key_points_async(text_chunks))

def generate_document_summary(text_chunks: List[str], summary_type: str = "medium") -> str:
    """Sync wrapper for generate_document_summary_async"""
    return asyncio.run(generate_document_summary_async(text_chunks, summary_type))

def identify_document_discussion_topics(text_chunks: List[str]) -> List[str]:
    """Sync wrapper for identify_document_discussion_topics_async"""
    return asyncio.run(identify_document_discussion_topics_async(text_chunks))