#!/usr/bin/env python3
"""
Filler Manager System

This module manages pre-generated filler audio files and provides random selection
for natural conversation flow during Q&A sessions.
"""

import random
import base64
from pathlib import Path
from typing import Optional, Tuple
import os

class FillerManager:
    """Manages filler audio files for both David and Emma personas."""
    
    def __init__(self, filler_audio_dir: str = "filler_audio"):
        """Initialize the filler manager.
        
        Args:
            filler_audio_dir: Directory containing filler audio files
        """
        self.filler_dir = Path(filler_audio_dir)
        self.david_fillers = []
        self.emma_fillers = []
        self._load_filler_files()
    
    def _load_filler_files(self):
        """Load all available filler audio files."""
        if not self.filler_dir.exists():
            print(f"⚠️  Filler directory not found: {self.filler_dir}")
            return
        
        # Load David's fillers
        david_files = sorted(self.filler_dir.glob("david_filler_*.mp3"))
        self.david_fillers = [str(f) for f in david_files]
        
        # Load Emma's fillers
        emma_files = sorted(self.filler_dir.glob("emma_filler_*.mp3"))
        self.emma_fillers = [str(f) for f in emma_files]
        
        print(f"📁 Loaded {len(self.david_fillers)} David fillers, {len(self.emma_fillers)} Emma fillers")
    
    def get_random_filler(self, speaker: str) -> Optional[Tuple[str, bytes]]:
        """Get a random filler audio for the specified speaker.
        
        Args:
            speaker: Either "David" or "Emma"
            
        Returns:
            Tuple of (base64_audio, raw_audio_bytes) or None if no fillers available
        """
        try:
            # Select appropriate filler list
            if speaker == "David":
                filler_files = self.david_fillers
            elif speaker == "Emma":
                filler_files = self.emma_fillers
            else:
                print(f"❌ Unknown speaker: {speaker}")
                return None
            
            if not filler_files:
                print(f"❌ No filler files available for {speaker}")
                return None
            
            # Randomly select a filler
            selected_file = random.choice(filler_files)
            
            # Load and encode the audio
            with open(selected_file, "rb") as f:
                raw_audio = f.read()
            
            base64_audio = base64.b64encode(raw_audio).decode('utf-8')
            
            print(f"🎲 Selected random filler for {speaker}: {Path(selected_file).name}")
            return base64_audio, raw_audio
            
        except Exception as e:
            print(f"❌ Error loading filler for {speaker}: {e}")
            return None
    
    def get_filler_for_handoff(self, current_speaker: str, target_speaker: str) -> Optional[Tuple[str, bytes]]:
        """Get a filler that hands off from current speaker to target speaker.
        
        Args:
            current_speaker: The speaker who will give the filler
            target_speaker: The speaker who should respond next
            
        Returns:
            Tuple of (base64_audio, raw_audio_bytes) or None if no suitable filler
        """
        # The filler should come from the current speaker, handing off to target
        return self.get_random_filler(current_speaker)
    
    def is_available(self) -> bool:
        """Check if filler system is available (has audio files)."""
        return len(self.david_fillers) > 0 and len(self.emma_fillers) > 0
    
    def get_stats(self) -> dict:
        """Get statistics about available fillers."""
        return {
            "david_count": len(self.david_fillers),
            "emma_count": len(self.emma_fillers),
            "total_count": len(self.david_fillers) + len(self.emma_fillers),
            "is_available": self.is_available()
        }

# Global filler manager instance
_filler_manager = None

def get_filler_manager() -> FillerManager:
    """Get the global filler manager instance (singleton pattern)."""
    global _filler_manager
    if _filler_manager is None:
        _filler_manager = FillerManager()
    return _filler_manager

def get_random_filler_audio(speaker: str) -> Optional[Tuple[str, bytes]]:
    """Convenience function to get random filler audio.
    
    Args:
        speaker: Either "David" or "Emma"
        
    Returns:
        Tuple of (base64_audio, raw_audio_bytes) or None
    """
    return get_filler_manager().get_random_filler(speaker)

def calculate_filler_duration(audio_data: bytes) -> float:
    """Calculate duration of filler audio in seconds.
    
    This is a simplified version - you might want to use the same
    calculate_audio_duration function from voice_chat_websocket.py
    
    Args:
        audio_data: Raw audio bytes
        
    Returns:
        Duration in seconds (estimated)
    """
    try:
        # For now, use a simple estimate based on file size
        # Typical MP3 bitrate is around 128kbps
        # This gives us roughly: file_size_bytes / (128 * 1024 / 8) seconds
        estimated_duration = len(audio_data) / (128 * 1024 / 8)
        
        # Clamp to reasonable bounds for fillers (1-4 seconds)
        return max(1.0, min(4.0, estimated_duration))
        
    except Exception:
        # Fallback to conservative estimate
        return 2.0

if __name__ == "__main__":
    # Test the filler manager
    print("🧪 Testing Filler Manager")
    print("=" * 40)
    
    manager = get_filler_manager()
    print(f"Stats: {manager.get_stats()}")
    
    if manager.is_available():
        print("\n🎲 Testing random filler selection:")
        
        # Test David filler
        david_result = manager.get_random_filler("David")
        if david_result:
            base64_audio, raw_audio = david_result
            duration = calculate_filler_duration(raw_audio)
            print(f"  David filler: {len(base64_audio)} chars, ~{duration:.1f}s")
        
        # Test Emma filler
        emma_result = manager.get_random_filler("Emma")
        if emma_result:
            base64_audio, raw_audio = emma_result
            duration = calculate_filler_duration(raw_audio)
            print(f"  Emma filler: {len(base64_audio)} chars, ~{duration:.1f}s")
            
        print("\n✅ Filler manager test complete!")
    else:
        print("❌ Filler manager not available - no audio files found")
