"""
voice_generator_async.py

Asynchronous Text-to-Speech Voice Generation for Podcast Debate Pipeline

- Async/concurrent processing with rate limiting and retry logic
- Integrates ElevenLabs API for high-quality TTS with distinct male/female voices
- Fallback to OpenAI TTS if ElevenLabs fails
- Handles voice selection, speech rate, emotional tone, and audio format optimization
- Robust error handling, exponential backoff retries, and logging
"""

import os
import time
import logging
import asyncio
import aiohttp
from typing import List, Dict, Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Import ElevenLabs client
try:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_CLIENT_AVAILABLE = True
except ImportError:
    ELEVENLABS_CLIENT_AVAILABLE = False
    logging.warning("ElevenLabs client library not available, falling back to requests")

# Load config and environment variables
from controller.config import get_elevenlabs_api_key, get_openai_api_key, get_elevenlabs_model

# Constants for API keys and endpoints
ELEVENLABS_API_KEY = get_elevenlabs_api_key()
OPENAI_API_KEY = get_openai_api_key()
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"

# Model configuration
ELEVENLABS_MODEL = get_elevenlabs_model()

# Initialize ElevenLabs client if available
if ELEVENLABS_CLIENT_AVAILABLE and ELEVENLABS_API_KEY:
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
else:
    elevenlabs_client = None

# Simplified voice configuration - only male and female (using .env values)
DEFAULT_VOICES = {
    "male": os.environ.get("ELEVENLABS_MALE_VOICE_ID", "8Pr9Gn4RGUaz7ZQg1YbH"),
    "female": os.environ.get("ELEVENLABS_FEMALE_VOICE_ID", "gDnGxUcsitTxRiGHr904")
}

# OpenAI TTS voice mapping for fallback - simplified
OPENAI_VOICES = {
    "male": "alloy",
    "female": "nova"
}

# Output directory for audio files
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Rate limiting - max concurrent requests
MAX_CONCURRENT_REQUESTS = 5
RATE_LIMIT_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def initialize_voices() -> Dict[str, str]:
    """
    Returns a dictionary of available voice IDs for male and female.
    """
    # In production, fetch available voices from ElevenLabs API
    return DEFAULT_VOICES

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def generate_speech_segment_async(
    text: str,
    voice: str = "male",
    speed: float = 1.0,
    emotion: Optional[str] = None,
    output_format: str = "mp3",
    segment_id: Optional[str] = None,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]:
    """
    Generate speech audio for a text segment using ElevenLabs, fallback to OpenAI TTS.
    Returns the path to the generated audio file, or None on failure.
    Async version with rate limiting and retry logic.
    """
    async with RATE_LIMIT_SEMAPHORE:
        voices = initialize_voices()
        voice_id = voices.get(voice, voices["male"])
        filename = f"segment_{segment_id or int(time.time())}_{voice}.{output_format}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Create session if not provided
        if session is None:
            async with aiohttp.ClientSession() as session:
                return await _generate_with_session(session, text, voice_id, voice, output_path, speed, emotion, segment_id)
        else:
            return await _generate_with_session(session, text, voice_id, voice, output_path, speed, emotion, segment_id)


async def _generate_with_session(
    session: aiohttp.ClientSession,
    text: str,
    voice_id: str,
    voice: str,
    output_path: str,
    speed: float,
    emotion: Optional[str],
    segment_id: Optional[str]
) -> Optional[str]:
    """Helper function to generate speech with a given aiohttp session."""
    
    # Try ElevenLabs client first (recommended for Flash v2.5) - Note: synchronous fallback
    if elevenlabs_client and ELEVENLABS_API_KEY:
        try:
            logging.info(f"Using ElevenLabs client with model {ELEVENLABS_MODEL} and voice {voice_id}")
            
            # Generate audio using the modern client (synchronous)
            audio = elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=ELEVENLABS_MODEL,
                # Optional voice settings can be added here
                # voice_settings=VoiceSettings(
                #     stability=0.5,
                #     similarity_boost=0.7,
                #     style=0.0,
                #     use_speaker_boost=True
                # )
            )
            
            # Save the audio file
            with open(output_path, "wb") as f:
                for chunk in audio:
                    f.write(chunk)
            
            logging.info(f"ElevenLabs client generation successful: {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"ElevenLabs client error: {e}")
            logging.info("Falling back to async ElevenLabs API")
    
    # Async ElevenLabs API fallback
    if ELEVENLABS_API_KEY:
        try:
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "text": text,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "use_speaker_boost": True
                }
            }
            # Remove problematic style parameter for v2 compatibility
            if emotion and emotion != "neutral":
                logging.warning(f"Emotion '{emotion}' not supported in async fallback")
            
            url = f"{ELEVENLABS_URL}/{voice_id}/stream"
            async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(output_path, "wb") as f:
                        f.write(content)
                    logging.info(f"ElevenLabs async fallback successful: {output_path}")
                    return output_path
                else:
                    error_text = await response.text()
                    logging.warning(f"ElevenLabs TTS failed: {response.status} {error_text}")
                    raise aiohttp.ClientError(f"ElevenLabs API failed: {response.status}")
        except Exception as e:
            logging.error(f"ElevenLabs async TTS error: {e}")
            # Don't raise here, try OpenAI fallback
    
    # Final fallback: OpenAI TTS async
    if OPENAI_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            openai_voice = OPENAI_VOICES.get(voice, "alloy")
            payload = {
                "model": "tts-1",
                "input": text,
                "voice": openai_voice,
                "response_format": output_format,
                "speed": speed
            }
            logging.info(f"Using OpenAI TTS async fallback with voice {openai_voice}")
            async with session.post(OPENAI_TTS_URL, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(output_path, "wb") as f:
                        f.write(content)
                    logging.info(f"OpenAI TTS async fallback successful: {output_path}")
                    return output_path
                else:
                    error_text = await response.text()
                    logging.warning(f"OpenAI TTS failed: {response.status} {error_text}")
                    raise aiohttp.ClientError(f"OpenAI TTS API failed: {response.status}")
        except Exception as e:
            logging.error(f"OpenAI async TTS error: {e}")
            raise
    
    logging.error(f"All async TTS methods failed for segment: {segment_id}")
    return None


async def process_dialogue_markers_async(
    script: List[Dict],
    voice_map: Optional[Dict[str, str]] = None,
    speed: float = 1.0,
    emotion_map: Optional[Dict[str, str]] = None,
    output_format: str = "mp3"
) -> List[str]:
    """
    Processes a dialogue script (list of dicts with 'speaker' and 'text') and generates audio files.
    Returns a list of audio file paths.
    Async version with concurrent processing and rate limiting.
    """
    if voice_map is None:
        voice_map = {"Host 1": "male", "Host 2": "female"}
    if emotion_map is None:
        emotion_map = {}

    # Create shared aiohttp session for all requests
    async with aiohttp.ClientSession() as session:
        # Create tasks for all segments
        tasks = []
        for idx, segment in enumerate(script):
            speaker = segment.get("speaker", "Host 1")
            text = segment.get("text", "")
            voice = voice_map.get(speaker, "male")
            emotion = emotion_map.get(speaker, None)
            
            task = generate_speech_segment_async(
                text=text,
                voice=voice,
                speed=speed,
                emotion=emotion,
                output_format=output_format,
                segment_id=f"{idx}_{speaker.replace(' ', '_')}",
                session=session
            )
            tasks.append(task)
        
        # Log progress
        logging.info(f"Starting async generation of {len(tasks)} audio segments with max {MAX_CONCURRENT_REQUESTS} concurrent requests")
        
        # Process all segments concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        audio_paths = []
        failed_count = 0
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Failed to generate audio for segment {idx}: {result}")
                failed_count += 1
            elif result is not None:
                audio_paths.append(result)
            else:
                logging.error(f"Failed to generate audio for segment {idx}: returned None")
                failed_count += 1
        
        logging.info(f"Async voice generation complete: {len(audio_paths)} successful, {failed_count} failed")
        return audio_paths


def process_dialogue_markers_sync(
    script: List[Dict],
    voice_map: Optional[Dict[str, str]] = None,
    speed: float = 1.0,
    emotion_map: Optional[Dict[str, str]] = None,
    output_format: str = "mp3"
) -> List[str]:
    """
    Synchronous wrapper for process_dialogue_markers_async.
    Maintained for backwards compatibility.
    """
    return asyncio.run(process_dialogue_markers_async(
        script=script,
        voice_map=voice_map,
        speed=speed,
        emotion_map=emotion_map,
        output_format=output_format
    ))


def generate_speech_segment(
    text: str,
    voice: str = "male",
    speed: float = 1.0,
    emotion: Optional[str] = None,
    output_format: str = "mp3",
    segment_id: Optional[str] = None
) -> Optional[str]:
    """
    Synchronous wrapper for generate_speech_segment_async.
    Maintained for backwards compatibility.
    """
    return asyncio.run(generate_speech_segment_async(
        text=text,
        voice=voice,
        speed=speed,
        emotion=emotion,
        output_format=output_format,
        segment_id=segment_id
    ))


# CLI for testing
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Generate podcast audio from dialogue script JSON (async version).")
    parser.add_argument("--script", type=str, required=True, help="Path to dialogue script JSON file")
    parser.add_argument("--output_format", type=str, default="mp3", help="Audio format (mp3, wav, etc.)")
    args = parser.parse_args()

    with open(args.script, "r") as f:
        script = json.load(f)

    audio_files = process_dialogue_markers_sync(script, output_format=args.output_format)
    print("Generated audio files:", audio_files)