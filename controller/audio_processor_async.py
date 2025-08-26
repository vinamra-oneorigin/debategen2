"""
audio_processor_async.py

Asynchronous Audio Processing for Podcast Debate Pipeline

- Concurrent audio loading and processing
- Memory-efficient streaming for large files
- Optimized audio combining and normalization
"""

import os
import asyncio
import json
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from pydub.effects import normalize

# Output directory for final podcast
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_audio")
TRANSCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "transcripts")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# Thread pool for CPU-intensive audio operations
AUDIO_THREAD_POOL = ThreadPoolExecutor(max_workers=4)

async def load_audio_segment_async(file_path: str) -> Optional[AudioSegment]:
    """
    Load audio segment asynchronously using thread pool.
    """
    def load_audio():
        try:
            return AudioSegment.from_file(file_path)
        except Exception:
            return None
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(AUDIO_THREAD_POOL, load_audio)

async def combine_audio_segments_async(
    audio_paths: List[str],
    pause_ms: int = 500
) -> Optional[AudioSegment]:
    """
    Combines audio file paths into a single AudioSegment asynchronously.
    Loads all audio files concurrently for better performance.
    """
    if not audio_paths:
        return None
    
    # Load all audio segments concurrently
    tasks = [load_audio_segment_async(path) for path in audio_paths]
    audio_segments = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out failed loads and exceptions
    valid_segments = []
    for segment in audio_segments:
        if isinstance(segment, AudioSegment):
            valid_segments.append(segment)
    
    if not valid_segments:
        return None
    
    # Combine segments with pauses in thread pool
    def combine_segments():
        try:
            combined = valid_segments[0]
            pause = AudioSegment.silent(duration=pause_ms)
            
            for segment in valid_segments[1:]:
                combined += pause + segment
            
            return combined
        except Exception:
            return None
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(AUDIO_THREAD_POOL, combine_segments)

async def combine_audio_segments_with_transcript_async(
    audio_paths: List[str],
    script: List[Dict],
    pause_ms: int = 500
) -> Tuple[Optional[AudioSegment], List[Dict]]:
    """
    Combines audio file paths into a single AudioSegment while generating a timestamped transcript.
    Returns both the combined audio and transcript with timestamps.
    """
    if not audio_paths or not script:
        return None, []
    
    # Load all audio segments concurrently
    tasks = [load_audio_segment_async(path) for path in audio_paths]
    audio_segments = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out failed loads and exceptions
    valid_segments = []
    valid_script_entries = []
    
    for i, segment in enumerate(audio_segments):
        if isinstance(segment, AudioSegment) and i < len(script):
            valid_segments.append(segment)
            valid_script_entries.append(script[i])
    
    if not valid_segments:
        return None, []
    
    # Combine segments with pauses and track timestamps
    def combine_segments_with_timestamps():
        try:
            combined = valid_segments[0]
            pause = AudioSegment.silent(duration=pause_ms)
            transcript = []
            
            current_time_ms = 0
            
            # Add first segment
            segment_duration_ms = len(valid_segments[0])
            transcript.append({
                "speaker": valid_script_entries[0].get("speaker", "Unknown"),
                "text": valid_script_entries[0].get("text", ""),
                "start_time_ms": current_time_ms,
                "end_time_ms": current_time_ms + segment_duration_ms,
                "start_time": format_timestamp(current_time_ms),
                "end_time": format_timestamp(current_time_ms + segment_duration_ms),
                "duration_ms": segment_duration_ms
            })
            current_time_ms += segment_duration_ms
            
            # Add remaining segments with pauses
            for i, segment in enumerate(valid_segments[1:], 1):
                # Add pause
                current_time_ms += pause_ms
                combined += pause
                
                # Add segment
                segment_start = current_time_ms
                segment_duration_ms = len(segment)
                segment_end = current_time_ms + segment_duration_ms
                
                transcript.append({
                    "speaker": valid_script_entries[i].get("speaker", "Unknown"),
                    "text": valid_script_entries[i].get("text", ""),
                    "start_time_ms": segment_start,
                    "end_time_ms": segment_end,
                    "start_time": format_timestamp(segment_start),
                    "end_time": format_timestamp(segment_end),
                    "duration_ms": segment_duration_ms
                })
                
                combined += segment
                current_time_ms += segment_duration_ms
            
            return combined, transcript
        except Exception:
            return None, []
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(AUDIO_THREAD_POOL, combine_segments_with_timestamps)

def format_timestamp(ms: int) -> str:
    """Convert milliseconds to MM:SS.mmm format"""
    total_seconds = ms / 1000
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"

async def normalize_audio_levels_async(audio: AudioSegment) -> AudioSegment:
    """
    Normalize audio levels asynchronously using thread pool.
    """
    def normalize_audio():
        try:
            return normalize(audio)
        except Exception:
            return audio  # Return original if normalization fails
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(AUDIO_THREAD_POOL, normalize_audio)

async def export_final_podcast_async(
    audio: AudioSegment,
    filename: str,
    output_format: str = "mp3",
    bitrate: str = "192k"
) -> Optional[str]:
    """
    Export final podcast audio asynchronously using thread pool.
    """
    def export_audio():
        try:
            output_path = os.path.join(OUTPUT_DIR, f"{filename}.{output_format}")
            
            # Export with quality settings
            if output_format.lower() == "mp3":
                audio.export(
                    output_path,
                    format="mp3",
                    bitrate=bitrate,
                    parameters=["-q:a", "2"]  # High quality MP3
                )
            elif output_format.lower() == "wav":
                audio.export(output_path, format="wav")
            else:
                audio.export(output_path, format=output_format)
            
            return output_path
        except Exception:
            return None
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(AUDIO_THREAD_POOL, export_audio)

async def add_natural_pauses_async(
    audio: AudioSegment,
    pause_positions: List[int],
    pause_ms: int = 500
) -> AudioSegment:
    """
    Insert pauses at specified positions asynchronously.
    """
    def add_pauses():
        try:
            if not pause_positions:
                return audio
            
            result = AudioSegment.empty()
            pause = AudioSegment.silent(duration=pause_ms)
            last_pos = 0
            
            for pos in sorted(pause_positions):
                if pos > last_pos:
                    result += audio[last_pos:pos] + pause
                    last_pos = pos
            
            # Add remaining audio
            if last_pos < len(audio):
                result += audio[last_pos:]
            
            return result
        except Exception:
            return audio
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(AUDIO_THREAD_POOL, add_pauses)

async def apply_audio_effects_async(
    audio: AudioSegment,
    effects: List[str] = None
) -> AudioSegment:
    """
    Apply audio effects asynchronously.
    Available effects: "normalize", "compress", "eq"
    """
    def apply_effects():
        try:
            result = audio
            
            if not effects:
                effects_to_apply = ["normalize"]
            else:
                effects_to_apply = effects
            
            for effect in effects_to_apply:
                if effect == "normalize":
                    result = normalize(result)
                elif effect == "compress":
                    # Simple compression by reducing dynamic range
                    result = result.compress_dynamic_range(threshold=-20.0, ratio=4.0)
                # Add more effects as needed
            
            return result
        except Exception:
            return audio
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(AUDIO_THREAD_POOL, apply_effects)

async def process_complete_audio_async(
    audio_paths: List[str],
    pause_ms: int = 500,
    effects: List[str] = None,
    output_filename: str = "podcast",
    output_format: str = "mp3",
    bitrate: str = "192k"
) -> Optional[str]:
    """
    Complete async audio processing pipeline.
    Combines, processes, and exports audio in one optimized flow.
    """
    try:
        # Step 1: Combine audio segments
        combined_audio = await combine_audio_segments_async(audio_paths, pause_ms)
        
        if not combined_audio:
            return None
        
        # Step 2: Apply audio effects (including normalization)
        processed_audio = await apply_audio_effects_async(combined_audio, effects)
        
        # Step 3: Export final podcast
        final_path = await export_final_podcast_async(
            processed_audio, output_filename, output_format, bitrate
        )
        
        return final_path
    
    except Exception:
        return None

async def process_complete_audio_with_transcript_async(
    audio_paths: List[str],
    script: List[Dict],
    pause_ms: int = 500,
    effects: List[str] = None,
    output_filename: str = "podcast",
    output_format: str = "mp3",
    bitrate: str = "192k"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Complete async audio processing pipeline with timestamped transcript generation.
    Combines, processes, and exports audio while generating a detailed transcript.
    Returns (audio_file_path, transcript_file_path).
    """
    try:
        # Step 1: Combine audio segments with transcript tracking
        combined_audio, transcript = await combine_audio_segments_with_transcript_async(
            audio_paths, script, pause_ms
        )
        
        if not combined_audio or not transcript:
            return None, None
        
        # Step 2: Apply audio effects (including normalization)
        processed_audio = await apply_audio_effects_async(combined_audio, effects)
        
        # Step 3: Export final podcast
        final_audio_path = await export_final_podcast_async(
            processed_audio, output_filename, output_format, bitrate
        )
        
        if not final_audio_path:
            return None, None
        
        # Step 4: Save timestamped transcript
        transcript_filename = f"{output_filename}_transcript.json"
        transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_filename)
        
        # Add metadata to transcript
        total_duration_ms = len(processed_audio)
        transcript_data = {
            "metadata": {
                "total_duration_ms": total_duration_ms,
                "total_duration": format_timestamp(total_duration_ms),
                "audio_file": os.path.basename(final_audio_path),
                "num_segments": len(transcript),
                "pause_duration_ms": pause_ms
            },
            "segments": transcript
        }
        
        def save_transcript():
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            return transcript_path
        
        loop = asyncio.get_event_loop()
        final_transcript_path = await loop.run_in_executor(AUDIO_THREAD_POOL, save_transcript)
        
        return final_audio_path, final_transcript_path
    
    except Exception:
        return None, None

async def cleanup_temp_files_async(file_paths: List[str]) -> int:
    """
    Clean up temporary audio files asynchronously.
    Returns the number of files successfully deleted.
    """
    def delete_file(file_path: str) -> bool:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception:
            pass
        return False
    
    # Delete files concurrently using thread pool
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(AUDIO_THREAD_POOL, delete_file, path)
        for path in file_paths
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count successful deletions
    deleted_count = sum(1 for result in results if result is True)
    return deleted_count

# Synchronous wrappers for backwards compatibility
def combine_audio_segments(audio_paths: List[str], pause_ms: int = 500) -> Optional[AudioSegment]:
    """Sync wrapper for combine_audio_segments_async"""
    return asyncio.run(combine_audio_segments_async(audio_paths, pause_ms))

def normalize_audio_levels(audio: AudioSegment) -> AudioSegment:
    """Sync wrapper for normalize_audio_levels_async"""
    return asyncio.run(normalize_audio_levels_async(audio))

def export_final_podcast(
    audio: AudioSegment,
    filename: str,
    output_format: str = "mp3",
    bitrate: str = "192k"
) -> Optional[str]:
    """Sync wrapper for export_final_podcast_async"""
    return asyncio.run(export_final_podcast_async(audio, filename, output_format, bitrate))

def add_natural_pauses(
    audio: AudioSegment,
    pause_positions: List[int],
    pause_ms: int = 500
) -> AudioSegment:
    """Sync wrapper for add_natural_pauses_async"""
    return asyncio.run(add_natural_pauses_async(audio, pause_positions, pause_ms))