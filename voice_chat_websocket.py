
from fastapi import APIRouter, WebSocket
import os
from contextlib import suppress
import json
import numpy as np
import base64
import aiohttp
from vad_utils import VAD_Amp
from statistics import mean
from controller.config import get_openai_api_key, get_elevenlabs_api_key, get_elevenlabs_model
import time
import asyncio
import wave
import tempfile
from pathlib import Path
from openai import OpenAI
from groq import AsyncGroq
import io
from pydub import AudioSegment
import traceback
import glob
import random
router = APIRouter()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = AsyncGroq(api_key=GROQ_API_KEY)

class ConnectionManager:
    """Tracks active WebSocket connections and handles safe connect/close."""
    def __init__(self) -> None:
        self.active_connections = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"🔗 NEW WEBSOCKET CONNECTION ESTABLISHED (active={len(self.active_connections)})")
        print(f"   Connection ID: {id(websocket)}")
        print(f"   Client: {websocket.client}")
        print(f"   Headers: {dict(websocket.headers)}")

    async def safe_close(self, websocket: WebSocket, code: int = 1000) -> None:
        with suppress(Exception):
            await websocket.close(code=code)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.discard(websocket)
            print(f"❌ WEBSOCKET DISCONNECTED (active={len(self.active_connections)})")
            print(f"   Connection ID: {id(websocket)}")
        else:
            print(f"⚠️  ATTEMPTED TO DISCONNECT UNKNOWN WEBSOCKET (ID: {id(websocket)})")

manager = ConnectionManager()

def is_question(text):
    """Check if the input text is a question by looking for question words.
    
    Args:
        text (str): The user's input text
        
    Returns:
        bool: True if the text appears to be a question, False otherwise
    """
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Question words to look for
    question_words = [
        'what', 'why', 'how', 'when', 'who', 'where', 'which', 'whose',
        'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did',
        'is', 'was', 'were', 'have', 'has', 'had'
    ]
    
    # Check if text ends with question mark
    if text_lower.endswith('?'):
        return True
    
    # Check if any question word appears in the text
    for word in question_words:
        if word in text_lower:
            return True
    
    return False

def get_filler_audio_path(speaker):
    """Get a random filler audio file path for the specified speaker.
    
    Args:
        speaker (str): Either "David" or "Emma"
        
    Returns:
        str: Path to a random filler audio file for the speaker, or None if not found
    """
    try:
        speaker_prefix = speaker.lower()
        filler_files = glob.glob(f"filler_audio/{speaker_prefix}_filler_*.mp3")
        
        if filler_files:
            return random.choice(filler_files)
        else:
            print(f"No filler audio files found for speaker: {speaker}")
            return None
    except Exception as e:
        print(f"Error getting filler audio path for {speaker}: {e}")
        return None

def load_filler_audio(speaker):
    """Load and encode filler audio for the specified speaker.
    
    Args:
        speaker (str): Either "David" or "Emma"
        
    Returns:
        tuple: (base64_encoded_audio, raw_audio_data) or (None, None) if failed
    """
    try:
        filler_path = get_filler_audio_path(speaker)
        if not filler_path:
            return None, None
        
        with open(filler_path, 'rb') as f:
            audio_data = f.read()
        
        base64_audio = base64.b64encode(audio_data).decode('utf-8')
        print(f"Loaded filler audio for {speaker}: {filler_path}")
        return base64_audio, audio_data
    except Exception as e:
        print(f"Error loading filler audio for {speaker}: {e}")
        return None, None

def parse_timestamp_to_ms(timestamp):
    """Parse timestamp in MM:SS.mmm format to milliseconds.
    
    Args:
        timestamp: Can be int (already in ms), str in MM:SS.mmm format, or other
        
    Returns:
        int: Timestamp in milliseconds, or None if invalid
    """
    if timestamp is None:
        return None
        
    # If it's already an integer, assume it's in milliseconds
    if isinstance(timestamp, int):
        return timestamp
        
    # If it's a string, try to parse MM:SS.mmm format
    if isinstance(timestamp, str):
        try:
            # Try direct integer conversion first (for cases like "16586")
            return int(timestamp)
        except ValueError:
            pass
            
        try:
            # Parse MM:SS.mmm format
            if ':' in timestamp:
                parts = timestamp.split(':')
                if len(parts) == 2:
                    minutes = int(parts[0])
                    seconds_parts = parts[1].split('.')
                    seconds = int(seconds_parts[0])
                    milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                    
                    # Convert to total milliseconds
                    total_ms = (minutes * 60 + seconds) * 1000 + milliseconds
                    return total_ms
        except (ValueError, IndexError):
            pass
    
    # Try direct conversion as fallback
    try:
        return int(float(timestamp))
    except (ValueError, TypeError):
        return None

def calculate_audio_duration(audio_data):
    """Calculate the duration of audio data in seconds.
    
    Args:
        audio_data (bytes): Audio data in MP3 or other formats
        
    Returns:
        float: Duration in seconds
    """
    try:
        # Load audio data into pydub AudioSegment
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
        # Return duration in seconds
        return len(audio_segment) / 1000.0
    except Exception as e:
        print(f"Error calculating audio duration: {e}")
        # Fallback: estimate based on typical speech rate (150 words per minute)
        # This is a rough estimate if audio parsing fails
        return 3.0  # Default to 3 seconds

def load_transcript_by_audio_id(audio_id):
    """Load transcript JSON file based on audio ID (hash).
    
    Args:
        audio_id (str): Hash ID of the PDF/audio to search for
        
    Returns:
        dict: Transcript data or None if not found
    """
    try:
        # Search for transcript files matching the hash pattern
        # New pattern: podcast_{hash}_transcript.json
        transcript_files = glob.glob(f"transcripts/podcast_{audio_id}_transcript.json")
        
        # Fallback to old pattern for backward compatibility
        if not transcript_files:
            transcript_files = glob.glob(f"transcripts/*{audio_id}*transcript.json")
        
        if not transcript_files:
            print(f"No transcript found for audio_id: {audio_id}")
            return None
            
        # With new naming, there should only be one file, but sort just in case
        transcript_path = transcript_files[0]
        print(f"Loading transcript from: {transcript_path}")
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
            
        return transcript_data
    except Exception as e:
        print(f"Error loading transcript for audio_id {audio_id}: {e}")
        return None

def find_closest_segment_and_extract_interactions(transcript_data, timestamp_ms):
    """Find closest segment to timestamp and extract 8 surrounding interactions.
    
    Args:
        transcript_data (dict): The loaded transcript data
        timestamp_ms (int): Target timestamp in milliseconds
        
    Returns:
        tuple: (target_segment_with_id, surrounding_interactions, resume_timestamp_ms) or (None, [], None)
    """
    try:
        segments = transcript_data.get("segments", [])
        if not segments:
            return None, [], None
            
        # Parse timestamp_ms to ensure it's in milliseconds
        timestamp_ms = parse_timestamp_to_ms(timestamp_ms)
        if timestamp_ms is None:
            print(f"Invalid timestamp_ms: {timestamp_ms}, defaulting to 0")
            timestamp_ms = 0
            
        # Find the closest segment with start_time_ms <= timestamp_ms
        closest_segment = None
        closest_index = -1
        
        for i, segment in enumerate(segments):
            # Ensure start_time_ms is an integer for comparison
            segment_start_time = segment.get("start_time_ms", 0)
            try:
                segment_start_time = int(segment_start_time)
            except (ValueError, TypeError):
                segment_start_time = 0
                
            if segment_start_time <= timestamp_ms:
                closest_segment = segment
                closest_index = i
            else:
                break
                
        if closest_segment is None:
            print(f"No segment found before timestamp {timestamp_ms}ms")
            return None, [], None
            
        print(f"Found closest segment at index {closest_index}: {closest_segment.get('text', '')[:50]}...")
        
        # Extract 2 interactions before and 2 after (4 total)
        # Each interaction is a pair of consecutive segments (question-answer pattern)
        start_index = max(0, closest_index - 4)  # 2 interactions = 4 segments before
        end_index = min(len(segments), closest_index + 4)  # 2 interactions = 4 segments after
        
        surrounding_segments = segments[start_index:end_index]
        
        # Add unique IDs to each segment for reference
        interactions_with_ids = []
        for i, segment in enumerate(surrounding_segments):
            segment_with_id = segment.copy()
            segment_with_id["interaction_id"] = f"seg_{start_index + i}"
            interactions_with_ids.append(segment_with_id)
            
        # Find the paused segment (where user stopped listening)
        paused_segment_with_id = None
        for interaction in interactions_with_ids:
            if interaction.get("start_time_ms") == closest_segment.get("start_time_ms"):
                paused_segment_with_id = interaction
                break
                
        # Resume timestamp will be determined by AI's choice of which segment to transition to
        # For now, return the paused segment's timestamp as default
        closest_message_timestamp_ms = closest_segment.get("start_time_ms", timestamp_ms)
        
        print(f"Extracted {len(interactions_with_ids)} surrounding segments")
        print(paused_segment_with_id, interactions_with_ids, closest_message_timestamp_ms)
        return paused_segment_with_id, interactions_with_ids, closest_message_timestamp_ms
        
    except Exception as e:
        print(f"Error extracting interactions: {e}")
        return None, [], None

async def generate_transcript_transition(summary, conversation_history, surrounding_interactions, paused_segment, user_input, next_speaker):
    """Generate AI conversation that smoothly transitions back to transcript content.
    
    Args:
        summary (str): The conversation summary
        conversation_history (list): Previous conversation messages
        surrounding_interactions (list): The 8 surrounding segments from transcript
        paused_segment (dict): The segment where user paused (context only)
        
    Returns:
        tuple: (response_data, elevenlabs_audio, resume_timestamp_ms)
    """
    try:
        # Create context from surrounding interactions
        interactions_text = "\n".join([
            f"{segment['speaker']}: {segment['text']}" 
            for segment in surrounding_interactions
        ])
        
        # Previous conversation context
        conversation_text = "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in conversation_history])
        
        # Paused segment info (where user stopped listening)
        paused_speaker = paused_segment.get("speaker", "")
        paused_text = paused_segment.get("text", "")
        paused_timestamp = paused_segment.get("start_time_ms", 0)
        
        # Create a list of available transition segments with their IDs
        available_segments = []
        for seg in surrounding_interactions:
            available_segments.append({
                "id": seg.get("interaction_id", ""),
                "speaker": seg.get("speaker", ""),
                "text": seg.get("text", "")[:100] + "..." if len(seg.get("text", "")) > 100 else seg.get("text", ""),
                "timestamp": seg.get("start_time_ms", 0)
            })
        
        # Check if timestamp is near the end (less than 4 segments remaining)
        # Get total duration from transcript metadata, fallback to default
        total_duration_ms = 200874  # Default fallback
        if surrounding_interactions:
            # Try to get from the last segment in surrounding interactions
            last_segment = max(surrounding_interactions, key=lambda x: x.get("start_time_ms", 0))
            if last_segment.get("end_time_ms"):
                total_duration_ms = last_segment.get("end_time_ms", total_duration_ms)
        
        time_remaining_ms = total_duration_ms - paused_timestamp
        
        # If less than 30 seconds remaining, just end normally
        if time_remaining_ms < 10000:
            system_prompt = f"""You are David (male) and Emma (female), AI podcast hosts ending a conversation with a user.

The user input is: {user_input} and the podcast is near the end. 

Respond naturally to the message the user sent and wrap up the conversation and thank the user for joining.

You MUST respond in this exact JSON format:
{{
    "responses": [
        {{
            "speaker": "Emma",
            "text": "Great! Question. That's because the length is measured in inches not centimeters. I hope that answers your question. Well Anyway Thanks for joining the conversation!"
        }},
        {{
            "speaker": "David", 
            "text": "Yes, it was wonderful having you! Enjoy the rest of your day. Goodbye."
        }}
    ],
    "disconnect_trigger": true,
    "resume_timestamp_ms": null,
    "target_segment_id": null
}}"""
        #TODO: here resume_timestamp  should be complete uration of the podcast

        else:
            second_speaker = "Emma" if next_speaker == "David" else "David"
            available_segments_text = "\n".join([
                f"ID: {seg['id']} | {seg['speaker']}: {seg['text']} (timestamp: {seg['timestamp']}ms)"
                for seg in available_segments
            ])
            
            system_prompt = f"""You are David (male) and Emma (female), AI podcast hosts who were joined by the user. First, answer their question/comment naturally, then smoothly transition back to the original podcast conversation.

CONTEXT:
- You were having a conversation about: {summary}
- A user joined with a question/comment
- User paused the podcast at: "{paused_text}" (spoken by {paused_speaker})
- You need to answer their message . The user input is: {user_input} 
- Answer this question and then transition back to the podcast

USER'S MESSAGE/QUESTION:
{user_input}

SURROUNDING CONVERSATION CONTEXT:
{interactions_text}

AVAILABLE TRANSITION POINTS:
{available_segments_text}

TASK:
1. First, properly address and answer the user's question/comment from their message
2. Then naturally transition the conversation to ONE of the available segments above
3. Choose which segment ID makes the most sense to transition to
4. Make sure both the answer and transition feel natural and conversational
5. CRITICAL SPEAKER ALTERNATION RULE: After generating your responses, choose a transition segment from the speaker who did NOT speak last in your generated responses. If Emma speaks last in your responses, choose a segment where David is speaking. If David speaks last, choose a segment where Emma is speaking. This ensures proper speaker alternation.
6. Note that the first speaker in the interaction you generate should be {next_speaker}
7. You are allowed more to one interactions if the second speaker is also the speaker you want to transition to. in which case it is allowed for there to be a 3rd message where the first speaker REDIRECTS TO THE TRANSITION POINT of the 2nd speaker speaking in the original podcast.
You MUST respond in this exact JSON format:
if first_speaker is {next_speaker} and second_speaker is {second_speaker}
{{
    "responses": [
        {{
            "speaker": "{next_speaker}",
            "text": "response here"
        }},
        {{
            "speaker": "{second_speaker}", 
            "text": "response here"
        }},
        {{
            "speaker": "{next_speaker}",
            "text": "response here"
        }} //optional
    ],
    "chosen_segment_id": "seg_X"
}}

EXAMPLE SCENARIO:

CONTEXT: User paused at "Einstein's photoelectric effect was revolutionary..." and asked questions about light waves.

SURROUNDING CONVERSATION CONTEXT:
David: Welcome to our physics discussion today
Emma: We're exploring quantum mechanics and its foundations
David: Yes. Einstein's photoelectric effect was revolutionary... [User Interrupted HERE]
Emma: This discovery showed that light behaves like particles
David: Right. The energy of these photons depends on frequency, not intensity
Emma: It's fascinating how this challenged classical wave theory

PREVIOUS Q&A SESSION:
David: Whoa, looks like somebody wants to join the conversation!
Emma: Amazing! Let's let them in!

User: Can you explain how light can be both a wave and particle?

first_speaker is David and second_speaker is Emma

AVAILABLE TRANSITION POINTS:
ID: seg_3 | Emma: This discovery showed that light behaves like particles (timestamp: 45000ms)
ID: seg_4 | David: Right. The energy of these photons depends on frequency, not intensity (timestamp: 52000ms)  
ID: seg_5 | Emma: It's fascinating how this challenged classical wave theory (timestamp: 61000ms)

GOOD RESPONSE:
{{
    "responses": [
        {{
            "speaker": "David",
            "text": "Think of it like light having two different personalities... Its called wave-particle duality."
        }},
        {{
            "speaker": "Emma", 
            "text": "Absolutely! I hope that answers your question. Now, coming back to what we were discussing - this discovery about photons really was groundbreaking. David, you were explaining how the energy depends on frequency?"
        }}
    ],
    "chosen_segment_id": "seg_4"
}}


Notice how the generated message "Absolutely! Now, coming back to what we were discussing - this discovery about photons really was groundbreaking. David, you were explaining how the energy depends on frequency?"
smoothly transitions to the segue point "Right. The energy of these photons depends on frequency, not intensity".
This is the kind of transition you need to make. Framing the last message a question helps to lead to the segue point.

Choose the segment that creates the smoothest, most natural transition from your conversation."""

        headers = {
            "Authorization": f"Bearer {get_openai_api_key()}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ],
            "max_tokens": 400,
            "temperature": 0.8,
            "response_format": {"type": "json_object"}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    response_content = result["choices"][0]["message"]["content"].strip()
                    
                    try:
                        response_data = json.loads(response_content)
                        
                        # Extract speaker messages and add to conversation history
                        speaker_messages = []
                        for resp in response_data.get("responses", []):
                            speaker = resp.get("speaker", "")
                            text = resp.get("text", "")
                            if speaker and text:
                                speaker_messages.append({"speaker": speaker, "text": text})
                                conversation_history.append({"speaker": speaker, "text": text})
                        
                        # Convert to audio
                        audio_messages = []
                        for resp in speaker_messages:
                            voice = "male" if resp["speaker"] == "David" else "female"
                            audio_messages.append({"text": resp["text"], "voice": voice})
                        
                        elevenlabs_audio, _ = await generate_combined_ai_speech_with_elevenlabs(audio_messages)
                        
                        # Get the AI's chosen segment for transition
                        chosen_segment_id = response_data.get("chosen_segment_id")
                        resume_timestamp_ms = None
                        
                        # Find the timestamp for the chosen segment ID
                        if chosen_segment_id:
                            for seg in surrounding_interactions:
                                if seg.get("interaction_id") == chosen_segment_id:
                                    resume_timestamp_ms = seg.get("start_time_ms")
                                    print(f"AI chose segment {chosen_segment_id} at timestamp {resume_timestamp_ms}ms")
                                    break
                            
                            if not resume_timestamp_ms:
                                print(f"Warning: Could not find timestamp for chosen segment {chosen_segment_id}")
                        
                        return response_data, elevenlabs_audio, resume_timestamp_ms
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error in transition generation: {e}")
                        return {}, "", None
                else:
                    print(f"API error in transition generation: {response.status}")
                    return {}, "", None
    except Exception as e:
        print(f"Error generating transcript transition: {e}")
        return {}, "", None

class InterractionSession:
    def __init__(self, websocket):        
        self.summary = ""
        self.websocket = websocket
        self.data_array = []
        self.voiced_confidences = []
        self.big_chunk = b""
        self.start_looking = False
        self.sd_duration = 20 # TODO: SAME HERE int(os.environ.get("SD_DURATION"))  # the duration of audio on which the process should decide if silence long enough to be triggered
        self.sd_threshold = 300 # TODO: REMOVE THIS HARDCODING#int(os.environ.get("SD_THRESHOLD"))  # the probabilty score of voice in the audio" on which the process should decide if silence long enough to be triggered
        self.conversation = []
        self.introduced = False #flag to check if the assistant has introduced itself. once done, it is set to true
        self.start_looking = False #flag used to start silence detection
        self.sd_duration = int(os.environ.get('SD_DURATION')) #the duration of audio on which the process should decide if silence long enough to be triggered
        self.sd_threshold = int(os.environ.get('SD_THRESHOLD')) #the "probabilty score of voice in the audio" on which the process should decide if silence long enough to be triggered
        self.psd_duration = int(os.environ.get('PSD_DURATION')) #if the average VAD.AMP score over the last n (Phrase start Detection Duration) confidences is more than the Phrase Start Detection Threshold, User has started speaking 
        self.psd_threshold = int(os.environ.get('PSD_THRESHOLD')) # if the average VAD.AMP score over the last n (Phrase start Detection Duration) confidences is more than the Phrase Start Detection Threshold, User has started speaking 
        self.processing = False #boolean storing if a response has already been generated by the llm and is inthe process of being converted to audio and sent to user
        self.last_message = False
        self.audio_blocking = False  # Flag to indicate if user audio should be blocked
        self.audio_block_end_time = 0  # Timestamp when audio blocking should end
        # New fields for transcript transition functionality
        self.initial_timestamp_ms = None  # Timestamp where user paused the audio
        self.audio_id = None  # ID/name of the audio file being played
        self.resume_timestamp_ms = None  # Timestamp to resume audio playback
        # Speaker alternation tracking
        self.who_started_speaking_last = "David"  # Track who spoke first in the last interaction, initialized to David since intro starts with David

    def start_audio_blocking(self, duration_seconds):
        """Start blocking user audio for the specified duration.
        
        Args:
            duration_seconds (float): Duration to block audio in seconds
        """
        self.audio_blocking = True
        self.audio_block_end_time = time.time() + duration_seconds
        print(f"Audio blocking started for {duration_seconds:.2f} seconds")

    def check_audio_blocking(self):
        """Check if audio blocking should still be active.
        
        Returns:
            bool: True if audio should still be blocked, False otherwise
        """
        if self.audio_blocking and time.time() >= self.audio_block_end_time:
            self.audio_blocking = False
            self.audio_block_end_time = 0
            print("Audio blocking ended")
        return self.audio_blocking

    async def push_audio(self, audio_bytes):
        if not self.summary :
            print("no summary")
            return

        # Check if audio should be blocked (AI is speaking)
        if self.check_audio_blocking() is True:
            # print("Audio blocked - AI is speaking")
            return

        if not self.introduced:
            
            self.processing = True
            greeting_messages = [
                {"text": "Whoa, looks like somebody wants to join the conversation!", "voice": "male"},
                {"text": "Amazing! Let's let them in!", "voice": "female"}
            ]
            
            # Save hardcoded introduction message to conversation history
            self.conversation.extend([
                {"speaker": "David", "text": "Whoa, looks like somebody wants to join the conversation!"},
                {"speaker": "Emma", "text": "Amazing! Let's let them in!"}
            ])
            
            # Send combined audio of both AI introductions to the user
            greeting_audio_base64, greeting_audio_data = await generate_combined_ai_speech_with_elevenlabs(greeting_messages)

            # Calculate audio duration and start blocking
            if greeting_audio_data:
                audio_duration = calculate_audio_duration(greeting_audio_data)
                self.start_audio_blocking(audio_duration)

            response_message = {
                "type": "response",
                "audio": {
                    "base64Wav": greeting_audio_base64
                }
            }
            print(f"📤 SENDING INTRO RESPONSE (Connection ID: {id(self.websocket)})")
            print(f"   Response type: {response_message['type']}")
            print(f"   Audio length: {len(greeting_audio_base64) if greeting_audio_base64 else 0} chars")
            await self.websocket.send_text(json.dumps(response_message))
            self.introduced = True
            self.processing = False

        if self.introduced and not self.processing: 
            self.big_chunk += audio_bytes
        
        if len(self.big_chunk) > 3072 : #chunks sent by twilio are much smaller than what is needed for Voice Activity Detection to perform operations, thus a big chunk is used to keep a cache of all chunks being recieved and once this cache is big enough (more than 3072), the chunks are proccessed and the cache is cleared.
            self.data_array.append(self.big_chunk)
            amplitude, VAD_confidence = VAD_Amp(self.big_chunk) #This function returns amplitude and voice activity confidence
            self.big_chunk = b''
            new_confidence = VAD_confidence*amplitude #custom confidence score (V.AMP) is created as a product of amplitude and voice activity confidence
            self.voiced_confidences.append(new_confidence)
    
        if len(self.voiced_confidences[-self.psd_duration:])!= 0 and mean(self.voiced_confidences[-self.psd_duration:]) > self.psd_threshold: #if the average V.AMP score over the last n (Phrase start Detection Duration) confidences is more than the Phrase Start Detection Threshold, User has started speaking 
            print("started speaking")
            clear_message = {"type": "clear"}
            print(f"📤 SENDING CLEAR MESSAGE - USER STARTED SPEAKING (Connection ID: {id(self.websocket)})")
            await self.websocket.send_text(json.dumps(clear_message))  #if interrupt is turned on and the candidate said something clear or stop any response being played over the socket
            self.start_looking = True #start looking for silence
    
        if self.start_looking is True and len(self.voiced_confidences) > self.sd_duration and mean(self.voiced_confidences[-self.sd_duration:]) < self.sd_threshold : #if silence detection has been triggered and length of the voiced confidences array is more than the silence detection duration set and average of the last n (Silence Detection Duration) elements of the voiced confidences array is less than the Silence Detection Threshold, it means sufficient duration of silence has been detected.
            print("silence detected")
            self.start_looking = False
            self.processing = True
            processing_message = {"type": "processing"}
            print(f"📤 SENDING PROCESSING MESSAGE (Connection ID: {id(self.websocket)})")
            print(f"   Silence detected - starting to process user speech")
            await self.websocket.send_text(json.dumps(processing_message))

            transcribed_text = await transcribe_audio_groq(b''.join(self.data_array))
            print("transcribing")
            llm_response, elevenlabs_respons = await generate_podcast_response(self.summary, self.conversation, transcribed_text, session=self) 
            
            if llm_response.get("disconnect_trigger") :
                self.last_message = True
                # Store resume timestamp if provided
                self.resume_timestamp_ms = llm_response.get("resume_timestamp_ms")
                print(f"🔴 DISCONNECT TRIGGER DETECTED! (Connection ID: {id(self.websocket)})")
                print(f"   Resume timestamp: {self.resume_timestamp_ms}")

            clear_message = {"type": "clear"}
            print(f"📤 SENDING CLEAR MESSAGE - BEFORE RESPONSE (Connection ID: {id(self.websocket)})")
            await self.websocket.send_text(json.dumps(clear_message))
            
            # Calculate audio duration and start blocking before sending response
            if elevenlabs_respons:
                # Get the raw audio data for duration calculation
                try:
                    raw_audio_data = base64.b64decode(elevenlabs_respons)
                    audio_duration = calculate_audio_duration(raw_audio_data)
                    self.start_audio_blocking(audio_duration)
                except Exception as e:
                    print(f"Error calculating audio duration for response: {e}")
                    # Fallback: block for a default duration
                    self.start_audio_blocking(3.0)
            
            main_response = {
                "type": "response",
                "audio": {
                    "base64Wav": elevenlabs_respons
                }
            }
            print(f"📤 SENDING MAIN RESPONSE (Connection ID: {id(self.websocket)})")
            print(f"   Response type: {main_response['type']}")
            print(f"   Audio length: {len(elevenlabs_respons) if elevenlabs_respons else 0} chars")
            print(f"   Is last message: {self.last_message}")
            await self.websocket.send_text(json.dumps(main_response))
            
            if self.last_message :
                print(f"⏳ LAST MESSAGE SENT - WILL CLOSE AFTER AUDIO FINISHES (Connection ID: {id(self.websocket)})")
                # Schedule connection close after audio finishes playing
                asyncio.create_task(self.delayed_close_after_audio())
            
            self.processing = False
            self.data_array = []

    async def delayed_close_after_audio(self):
        """Wait for audio to finish playing, then send resume timestamp and close connection."""
        print(f"⏰ STARTING DELAYED CLOSE TIMER (Connection ID: {id(self.websocket)})")
        
        # Wait for audio blocking to end (audio finishes playing)
        while self.check_audio_blocking():
            await asyncio.sleep(0.1)  # Check every 100ms
        
        print(f"🎵 AUDIO FINISHED PLAYING (Connection ID: {id(self.websocket)})")
        
        # Add a small buffer to ensure audio is fully processed on frontend
        await asyncio.sleep(0.5)
        
        print(f"🔚 NOW CLOSING CONNECTION AFTER AUDIO COMPLETION (Connection ID: {id(self.websocket)})")
        await self.close_connection()

    async def close_connection(self):
        print(f"🔚 CLOSING CONNECTION (Connection ID: {id(self.websocket)})")
        # Send resume timestamp if available before closing
        if hasattr(self, 'resume_timestamp_ms') and self.resume_timestamp_ms:
            resume_message = {
                "type": "resume_timestamp",
                "timestamp_ms": self.resume_timestamp_ms
            }
            resume_payload = json.dumps(resume_message)
            print(f"📤 SENDING RESUME TIMESTAMP BEFORE CLOSE (Connection ID: {id(self.websocket)})")
            print(f"   Resume timestamp: {self.resume_timestamp_ms}ms")
            print(f"   Resume message: {resume_message}")
            print(f"   PRE-DISCONNECT MESSAGE: {resume_message}")
            print(f"   PRE-DISCONNECT PAYLOAD: {resume_payload}")
            await self.websocket.send_text(resume_payload)
        else:
            print(f"   NO RESUME TIMESTAMP TO SEND - resume_timestamp_ms: {getattr(self, 'resume_timestamp_ms', 'NOT_SET')}")
        
        print(f"❌ WEBSOCKET CLOSING NOW (Connection ID: {id(self.websocket)})")
        await self.websocket.close()


async def transcribe_audio(audio_bytes):
    """Transcribe raw PCM16 audio bytes using OpenAI SDK.

    Steps:
    - Wrap PCM16 bytes into a mono 48 kHz WAV file
    - Save to a temporary file under data/temp
    - Use OpenAI Python SDK to transcribe (gpt-4o-transcribe)
    - Clean up temp file
    """
    if not audio_bytes:
        return ""
    SAMPLE_RATE = 48000
    NUM_CHANNELS = 1
    SAMPLE_WIDTH_BYTES = 2
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = None
    try:
        cur_time = time.time()
        # Create a unique temp WAV file path
        wav_fd, wav_path_str = tempfile.mkstemp(prefix="ws_audio_", suffix=".wav", dir=str(temp_dir))
        os.close(wav_fd)
        wav_path = Path(wav_path_str)
        # Write WAV header + frames
        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(NUM_CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH_BYTES)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)

        # # Save a permanent copy to the current working directory for manual listening/debugging
        # try:
        #     ts = time.strftime('%Y%m%d_%H%M%S')
        #     rand = str(uuid.uuid4())[:8]
        #     cwd_save_path = Path.cwd() / f"ws_audio_{ts}_{rand}.wav"
        #     shutil.copyfile(str(wav_path), str(cwd_save_path))
        #     print(f"Saved recorded WAV to: {cwd_save_path}")
        # except Exception as save_err:
        #     print(f"Failed to save a local WAV copy: {save_err}")

        # Use OpenAI SDK in a worker thread to avoid blocking the event loop
        def _do_transcribe(path: Path) -> str:
            client = OpenAI(api_key=get_openai_api_key())
            with open(str(path), "rb") as f:
                result = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=f
                )
            # The SDK returns an object with .text
            return getattr(result, "text", "") or ""

        text = await asyncio.to_thread(_do_transcribe, wav_path)
        text = (text or "").strip()
        print(f"Transcription took {time.time() - cur_time:.2f} seconds; length={len(audio_bytes)} bytes")
        print(f"Transcription result: {text}")
        return text
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""
    finally:
        with suppress(Exception):
            if wav_path and wav_path.exists():
                os.remove(str(wav_path))

async def transcribe_audio_groq(audio_bytes):
    """Transcribe raw PCM16 audio bytes using Groq Async SDK.

    Mirrors transcribe_audio: wraps bytes in a mono 48 kHz WAV temp file,
    saves a permanent copy for debugging, calls Groq, returns text.
    """
    if not audio_bytes:
        return ""
    SAMPLE_RATE = 48000
    NUM_CHANNELS = 1
    SAMPLE_WIDTH_BYTES = 2
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = None
    try:
        cur_time = time.time()
        wav_fd, wav_path_str = tempfile.mkstemp(prefix="ws_audio_", suffix=".wav", dir=str(temp_dir))
        os.close(wav_fd)
        wav_path = Path(wav_path_str)
        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(NUM_CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH_BYTES)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)

        # # Save a permanent copy locally for listening/debugging
        # try:
        #     ts = time.strftime('%Y%m%d_%H%M%S')
        #     rand = str(uuid.uuid4())[:8]
        #     cwd_save_path = Path.cwd() / f"ws_audio_groq_{ts}_{rand}.wav"
        #     shutil.copyfile(str(wav_path), str(cwd_save_path))
        #     print(f"Saved recorded WAV (Groq) to: {cwd_save_path}")
        # except Exception as save_err:
        #     print(f"Failed to save a local WAV copy (Groq): {save_err}")

        text = ""
        with open(str(wav_path), "rb") as f:
            transcription = await client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language="en"
            )
        text = getattr(transcription, "text", "") or ""

        text = text.strip()
        print(f"Groq transcription took {time.time() - cur_time:.2f} seconds; length={len(audio_bytes)} bytes")
        print(f"Groq transcription result: {text}")
        return text
    except Exception as e:
        print(f"Groq transcription error: {e}")
        print(f"Groq transcription error traceback:")
        traceback.print_exc()
        return ""
    finally:
        with suppress(Exception):
            if wav_path and wav_path.exists():
                os.remove(str(wav_path))

async def generate_podcast_response(summary, conversation_history, user_question, session):
    """Generate response from both AI personas using GPT."""
    try:
        # Play filler audio for any user message
        # Use the opposite speaker for filler to ensure proper alternation
        filler_speaker = random.choice(["David", "Emma"])
        print(f"🎵 User message received: '{user_question}' - Playing filler from {filler_speaker}")
        filler_base64, filler_raw = load_filler_audio(filler_speaker)
        next_speaker = "David" if filler_speaker == "Emma" else "Emma"
        if filler_base64:
            # Add half-second delay before filler audio
            print("⏳ Adding 0.5 second delay before filler audio")
            await asyncio.sleep(2.5)
            
            # Calculate filler audio duration and start blocking
            if filler_raw:
                filler_duration = calculate_audio_duration(filler_raw)
                session.start_audio_blocking(filler_duration)
            
            # Send filler audio after delay
            filler_message = {
                "type": "response",
                "audio": {
                    "base64Wav": filler_base64
                }
            }
            print(f"📤 SENDING FILLER AUDIO from {filler_speaker}")
            await session.websocket.send_text(json.dumps(filler_message))
        
        # Since user is only allowed one message, always trigger transcript transition
        # This is the ONLY response path since disconnect_trigger is always True
        if session and session.initial_timestamp_ms and session.audio_id:
            print(f"🔄 USER SENT MESSAGE - TRIGGERING TRANSCRIPT TRANSITION (Connection ID: {id(session.websocket)})")
            
            # Load transcript and find transition point
            transcript_data = load_transcript_by_audio_id(session.audio_id)
            if transcript_data:
                paused_segment_with_id, interactions_with_ids, closest_message_timestamp_ms = find_closest_segment_and_extract_interactions(
                    transcript_data, session.initial_timestamp_ms
                )
                
                if paused_segment_with_id and interactions_with_ids:
                    print(f"User paused at segment: {paused_segment_with_id.get('text', '')[:50]}...")
                    response_data, elevenlabs_audio, resume_timestamp_ms = await generate_transcript_transition(
                        summary, conversation_history, interactions_with_ids, paused_segment_with_id, user_question, next_speaker
                    )
                    
                    # Add resume timestamp to response
                    if resume_timestamp_ms:
                        response_data["resume_timestamp_ms"] = resume_timestamp_ms
                    
                    response_data["disconnect_trigger"] = True

                    return response_data, elevenlabs_audio
       
    except Exception as e:
        print(f"Response generation error: {e}")
        return []

async def generate_combined_ai_speech_with_elevenlabs(messages):
    """Generate combined audio from multiple AI personas using ElevenLabs.
    
    Returns:
        tuple: (base64_encoded_audio, raw_audio_data)
    """
    try:
        combined_audio_chunks = []
        
        for message in messages:
            print(f"Generating audio for {message['voice']} voice: {message['text']} time {time.time()}")
            voice_id = os.environ.get("ELEVENLABS_MALE_VOICE_ID") if message["voice"] == "male" else os.environ.get("ELEVENLABS_FEMALE_VOICE_ID")
            
            headers = {
                "xi-api-key": get_elevenlabs_api_key(),
                "Content-Type": "application/json"
            }
            
            payload = {
                "text": message["text"],
                "model_id": get_elevenlabs_model(),
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "use_speaker_boost": True
                }
            }
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=15
                ) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        combined_audio_chunks.append(audio_data)
                    else:
                        print(f"ElevenLabs API error for {message['voice']} voice: {response.__dict__}")
        
        if combined_audio_chunks:
            # For simplicity, we'll concatenate the audio chunks
            # In a production system, you might want to add small silences between speakers
            combined_audio = b''.join(combined_audio_chunks)
            base64_audio = base64.b64encode(combined_audio).decode('utf-8')
            return base64_audio, combined_audio
        else:
            return "", b""
            
    except Exception as e:
        print(f"Combined audio generation error: {e}")
        return "", b""

@router.websocket("/ws/audio")
async def chatbot_websocket(websocket: WebSocket) -> None:

    await manager.connect(websocket)

    conv_session = InterractionSession(websocket)

    while True :
        message = await websocket.receive()
        
        # Only log non-audio messages to avoid terminal spam
        if message.get("type") == "websocket.disconnect":
            print(f"🔴 CLIENT REQUESTED DISCONNECT (Connection ID: {id(websocket)})")
            break

        raw_text = message.get("text")
        if raw_text:
            try:
                data = json.loads(raw_text)
                # Only log non-audio messages in detail
                if data.get('type') != 'audio':
                    print(f"📥 INCOMING WEBSOCKET MESSAGE (Connection ID: {id(websocket)})")
                    print(f"   Message type: {message.get('type')}")
                    print(f"   Parsed data type: {data.get('type')}")
                    print(f"   Data content: {data}")
            except json.JSONDecodeError as e:
                print(f"📥 INCOMING WEBSOCKET MESSAGE (Connection ID: {id(websocket)})")
                print(f"   ❌ JSON DECODE ERROR: {e}")
                print(f"   Raw text: {raw_text[:200]}...")
                continue
        else:
            print(f"📥 INCOMING WEBSOCKET MESSAGE (Connection ID: {id(websocket)})")
            print(f"   ⚠️  EMPTY MESSAGE TEXT")
            continue

        if data.get("type") == "summary":
            print(f"🔧 PROCESSING SUMMARY MESSAGE")
            content = json.loads(data["content"])
            conv_session.summary = content["summary"]
            print(f"   Summary set: {conv_session.summary[:100]}...")
            
        if data.get("type") == "session_init":
            print(f"🚀 PROCESSING SESSION_INIT MESSAGE")
            # Handle initial session data with timestamp and audio ID
            content = json.loads(data["content"])
            # Parse timestamp using the new function that handles MM:SS.mmm format
            timestamp_ms = content.get("timestamp_ms")
            conv_session.initial_timestamp_ms = parse_timestamp_to_ms(timestamp_ms)
            if conv_session.initial_timestamp_ms is None and timestamp_ms is not None:
                print(f"   ❌ Invalid timestamp_ms in session_init: {timestamp_ms}, setting to None")
            conv_session.audio_id = content.get("audio_id")
            conv_session.summary = content.get("summary")
            print(f"   ✅ Session initialized with timestamp: {conv_session.initial_timestamp_ms}ms, audio_id: {conv_session.audio_id}")
            print(f"   Connection ID: {id(websocket)}")
            
        if data.get("type") == "audio":
            # print("received audio")
            audio = data["audio"]
            pcm_array = audio["pcm16Array"]
            audio_int16 = np.array(pcm_array, dtype=np.int16)
            audio_bytes = audio_int16.tobytes()
            await conv_session.push_audio(audio_bytes)