"""
filler_generator_async.py

Asynchronous Filler Audio Generation for Podcast Pipeline

- Generates fresh filler audio files for podcast hosts
- Uses predefined filler text lists for David and Emma
- Replaces existing filler files with new ones
- Runs independently from main podcast generation
"""

import os
import logging
import asyncio
from typing import List, Dict, Optional
from controller.voice_generator_async import generate_speech_segment_async

# Predefined filler text lists for each speaker (4 fillers each)
DAVID_FILLERS = [
    "Hey there! Glad to have you here. Emma do you want to take this one?",  # david_filler_1.mp3
    "Good Point! You've raised a very interesting point there. Emma what do you think about this ?",  # david_filler_2.mp3
    "Hmm.. Interesting Question. Let's hear Emma's take on that.",  # david_filler_3.mp3
    "Good Question! Emma you're really good with this topic."   # david_filler_4.mp3 (if needed)
]

EMMA_FILLERS = [
    "Hello There! That's a fascinating question. I'm sure david would like to take this one",  # emma_filler_1.mp3
    "Interesting Point! David I think you'd have good insights on this.",  # emma_filler_2.mp3
    "Hmm.. Interesting Question. David whats your take on this?",  # emma_filler_3.mp3
    "Good Question! David I know you're a specialist on this topic. Why dont you take this one?"   # emma_filler_4.mp3 (if needed)
]

# Filler audio directory
FILLER_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "filler_audio")
os.makedirs(FILLER_AUDIO_DIR, exist_ok=True)

async def generate_speaker_filler_async(speaker: str) -> List[Optional[str]]:
    """
    Generate all filler audio files for a speaker.
    
    Args:
        speaker (str): Either "David" or "Emma"
        
    Returns:
        List[Optional[str]]: List of paths to generated filler audio files
    """
    try:
        # Get the filler list for the speaker
        if speaker.lower() == "david":
            filler_list = DAVID_FILLERS
        elif speaker.lower() == "emma":
            filler_list = EMMA_FILLERS
        else:
            logging.warning(f"Unknown speaker: {speaker}, defaulting to David")
            filler_list = DAVID_FILLERS
        
        # Determine voice type
        voice_type = "male" if speaker.lower() == "david" else "female"
        speaker_prefix = speaker.lower()
        
        # Generate audio files for each filler text
        generated_paths = []
        
        for i, filler_text in enumerate(filler_list, 1):
            if not filler_text.strip():  # Skip empty strings
                logging.info(f"Skipping empty filler {i} for {speaker}")
                generated_paths.append(None)
                continue
                
            try:
                # Generate filename
                filename = f"{speaker_prefix}_filler_{i}"
                
                # Generate audio using the existing TTS system
                audio_path = await generate_speech_segment_async(
                    text=filler_text,
                    voice=voice_type,
                    speed=1.0,
                    emotion=None,
                    output_format="mp3",
                    segment_id=filename
                )
                
                if audio_path:
                    # Move to filler audio directory with correct naming
                    target_path = os.path.join(FILLER_AUDIO_DIR, f"{filename}.mp3")
                    
                    # Remove existing file if it exists
                    if os.path.exists(target_path):
                        os.remove(target_path)
                        logging.info(f"Removed existing filler: {target_path}")
                    
                    # Move the generated file to the correct location
                    if os.path.exists(audio_path):
                        os.rename(audio_path, target_path)
                        logging.info(f"Generated filler {i} for {speaker}: {target_path}")
                        generated_paths.append(target_path)
                    else:
                        logging.error(f"Generated audio file not found: {audio_path}")
                        generated_paths.append(None)
                else:
                    logging.error(f"Failed to generate filler {i} audio for {speaker}")
                    generated_paths.append(None)
                    
            except Exception as e:
                logging.error(f"Error generating filler {i} for {speaker}: {e}")
                generated_paths.append(None)
        
        return generated_paths
            
    except Exception as e:
        logging.error(f"Error generating fillers for {speaker}: {e}")
        return [None] * len(DAVID_FILLERS if speaker.lower() == "david" else EMMA_FILLERS)

async def cleanup_old_fillers():
    """
    Remove all existing filler files to make room for new ones.
    This ensures we don't accumulate old files over time.
    """
    try:
        for filename in os.listdir(FILLER_AUDIO_DIR):
            if filename.endswith('.mp3') and ('david_filler_' in filename or 'emma_filler_' in filename):
                file_path = os.path.join(FILLER_AUDIO_DIR, filename)
                os.remove(file_path)
                logging.info(f"Cleaned up old filler: {filename}")
    except Exception as e:
        logging.error(f"Error cleaning up old fillers: {e}")

async def generate_fresh_fillers_async() -> Dict[str, List[Optional[str]]]:
    """
    Generate fresh filler audio files for both David and Emma.
    Replaces existing filler files with new ones.
    
    Returns:
        Dict[str, List[Optional[str]]]: Dictionary with speaker names and their lists of filler file paths
    """
    logging.info("Starting fresh filler generation for podcast hosts")
    
    # First, clean up old filler files
    await cleanup_old_fillers()
    
    # Generate fillers for both speakers concurrently
    david_task = generate_speaker_filler_async("David")
    emma_task = generate_speaker_filler_async("Emma")
    
    # Wait for both to complete
    david_result, emma_result = await asyncio.gather(david_task, emma_task, return_exceptions=True)
    
    # Process results
    results = {}
    
    if isinstance(david_result, Exception):
        logging.error(f"David filler generation failed: {david_result}")
        results["David"] = []
    else:
        results["David"] = david_result
    
    if isinstance(emma_result, Exception):
        logging.error(f"Emma filler generation failed: {emma_result}")
        results["Emma"] = []
    else:
        results["Emma"] = emma_result
    
    # Log results
    david_successful = sum(1 for path in results["David"] if path is not None)
    emma_successful = sum(1 for path in results["Emma"] if path is not None)
    total_successful = david_successful + emma_successful
    
    logging.info(f"Filler generation complete: David {david_successful}/{len(DAVID_FILLERS)}, Emma {emma_successful}/{len(EMMA_FILLERS)}, Total {total_successful}")
    
    return results