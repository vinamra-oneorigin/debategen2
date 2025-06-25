"""
script_generator_async.py

Asynchronous Script Generation for Podcast Debate Pipeline

- Concurrent host persona and dialogue script generation
- Rate limiting and retry logic with tenacity
- Optimized prompts for faster GPT responses
"""

import json
import random
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from controller.config import get_openai_api_key, get_openai_model

# Rate limiting for OpenAI API calls
MAX_CONCURRENT_API_CALLS = 3
API_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

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
    max_tokens: int = 512,
    temperature: float = 0.7
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
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": response_schema
            }
        }
        
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60  # Longer timeout for script generation
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

async def generate_host_personas_async(podcast_topic: str, tone: str = "neutral") -> Dict[str, Any]:
    """
    Generates two distinct and complementary podcast host personas using GPT-4.1-mini async.
    """
    seed = random.randint(1000, 9999)
    prompt = (
        f"You are designing two podcast hosts for a show about '{podcast_topic}'. "
        f"The desired tone is '{tone}'. "
        f"Create two distinct and complementary host personas. "
        f"For each host, provide: name, background, personality traits, expertise level, and speaking style. "
        f"Ensure the hosts have different but compatible personalities and conversational styles. "
        f"Include some randomization for variety (seed: {seed})."
    )
    
    schema = {
        "name": "host_personas",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "host1": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "background": {"type": "string"},
                        "personality_traits": {"type": "string"},
                        "expertise_level": {"type": "string"},
                        "speaking_style": {"type": "string"}
                    },
                    "required": ["name", "background", "personality_traits", "expertise_level", "speaking_style"],
                    "additionalProperties": False
                },
                "host2": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "background": {"type": "string"},
                        "personality_traits": {"type": "string"},
                        "expertise_level": {"type": "string"},
                        "speaking_style": {"type": "string"}
                    },
                    "required": ["name", "background", "personality_traits", "expertise_level", "speaking_style"],
                    "additionalProperties": False
                }
            },
            "required": ["host1", "host2"],
            "additionalProperties": False
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            result = await make_openai_request(session, prompt, schema, temperature=0.7)
            if not ("host1" in result and "host2" in result):
                raise ValueError("Personas response missing host1 or host2")
            return result
    except Exception:
        return {}

async def create_dialogue_script_async(
    host_personas: Dict[str, Any],
    podcast_topic: str,
    target_length_minutes: int = 10,
    tone: str = "neutral",
    content_focus: str = "educational",
    technical_level: str = "intermediate",
    inclusion_of_humor: int = 0
) -> List[Dict[str, str]]:
    """
    Generates a conversational script between two hosts using GPT-4.1-mini async.
    """
    # Validate and set defaults
    valid_focus = {"educational", "entertaining", "debate", "balanced"}
    if content_focus not in valid_focus:
        content_focus = "educational"
    
    valid_levels = {"beginner", "intermediate", "expert", "general"}
    if technical_level not in valid_levels:
        technical_level = "intermediate"
    
    if not isinstance(inclusion_of_humor, (int, float)):
        inclusion_of_humor = 0
    
    if not (host_personas and "host1" in host_personas and "host2" in host_personas):
        return []
    
    host1 = host_personas["host1"]
    host2 = host_personas["host2"]
    
    # Estimate target word count (approx 130 words per minute for spoken dialogue)
    target_words = int(target_length_minutes * 130)
    
    prompt = (
        f"You are to generate a podcast script for a show about '{podcast_topic}'.\n"
        f"The desired tone is '{tone}'.\n"
        f"Content focus: {content_focus}.\n"
        f"Technical level: {technical_level}.\n"
        f"Humor level: {inclusion_of_humor}/10.\n"
        f"Target length: approximately {target_words} words total.\n\n"
        f"Host 1: {host1['name']}\n"
        f"Background: {host1['background']}\n"
        f"Personality: {host1['personality_traits']}\n"
        f"Speaking style: {host1['speaking_style']}\n\n"
        f"Host 2: {host2['name']}\n"
        f"Background: {host2['background']}\n"
        f"Personality: {host2['personality_traits']}\n"
        f"Speaking style: {host2['speaking_style']}\n\n"
        f"Generate a natural, engaging dialogue between these two hosts. "
        f"Include an introduction, main discussion points, and a conclusion. "
        f"Make the conversation flow naturally with back-and-forth exchanges. "
        f"Each turn should be 1-3 sentences for natural pacing."
    )
    
    schema = {
        "name": "dialogue_script",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string"},
                            "text": {"type": "string"}
                        },
                        "required": ["speaker", "text"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["script"],
            "additionalProperties": False
        }
    }
    
    try:
        # Use higher max_tokens for script generation
        async with aiohttp.ClientSession() as session:
            result = await make_openai_request(
                session, prompt, schema, 
                max_tokens=2048,  # More tokens for longer scripts
                temperature=0.7
            )
            return result.get("script", [])
    except Exception:
        return []

def format_script_for_tts(script: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Format script for TTS processing (clean up text, normalize speaker names).
    """
    if not script:
        return []
    
    formatted_script = []
    for segment in script:
        speaker = segment.get("speaker", "Unknown").strip()
        text = segment.get("text", "").strip()
        
        if text:  # Only include non-empty segments
            # Clean up text for TTS
            # Remove excessive punctuation, normalize spacing
            import re
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = re.sub(r'[^\w\s\.,!?;:\-\'"]', '', text)  # Remove special chars
            
            formatted_script.append({
                "speaker": speaker,
                "text": text
            })
    
    return formatted_script

async def generate_podcast_script_async(
    podcast_topic: str,
    target_length_minutes: int = 10,
    tone: str = "engaging",
    content_focus: str = "balanced",
    technical_level: str = "general",
    inclusion_of_humor: int = 2
) -> Optional[Dict[str, Any]]:
    """
    Complete async podcast script generation pipeline.
    Generates personas and script concurrently when possible.
    """
    try:
        # Step 1: Generate host personas
        personas = await generate_host_personas_async(podcast_topic, tone)
        
        if not personas:
            return None
        
        # Step 2: Generate dialogue script
        script = await create_dialogue_script_async(
            personas,
            podcast_topic,
            target_length_minutes,
            tone,
            content_focus,
            technical_level,
            inclusion_of_humor
        )
        
        if not script:
            return None
        
        # Step 3: Format for TTS
        tts_script = format_script_for_tts(script)
        
        return {
            "personas": personas,
            "script": script,
            "tts_script": tts_script
        }
    
    except Exception:
        return None

# Synchronous wrappers for backwards compatibility
def generate_host_personas(podcast_topic: str, tone: str = "neutral") -> Dict[str, Any]:
    """Sync wrapper for generate_host_personas_async"""
    return asyncio.run(generate_host_personas_async(podcast_topic, tone))

def create_dialogue_script(
    host_personas: Dict[str, Any],
    podcast_topic: str,
    target_length_minutes: int = 10,
    tone: str = "neutral",
    content_focus: str = "educational",
    technical_level: str = "intermediate",
    inclusion_of_humor: int = 0
) -> List[Dict[str, str]]:
    """Sync wrapper for create_dialogue_script_async"""
    return asyncio.run(create_dialogue_script_async(
        host_personas, podcast_topic, target_length_minutes,
        tone, content_focus, technical_level, inclusion_of_humor
    ))

def generate_podcast_script(
    podcast_topic: str,
    target_length_minutes: int = 10,
    tone: str = "engaging",
    content_focus: str = "balanced",
    technical_level: str = "general",
    inclusion_of_humor: int = 2
) -> Optional[Dict[str, Any]]:
    """Sync wrapper for generate_podcast_script_async"""
    return asyncio.run(generate_podcast_script_async(
        podcast_topic, target_length_minutes, tone,
        content_focus, technical_level, inclusion_of_humor
    ))