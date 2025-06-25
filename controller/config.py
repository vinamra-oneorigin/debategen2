import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()

# Required API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Model configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")

# Validate required API keys
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in environment variables or .env file.")
    raise ValueError("OPENAI_API_KEY is required for operation.")

if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLABS_API_KEY is not set. ElevenLabs TTS features will be unavailable.")

# Default voice settings
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "en-US-Standard-B")
DEFAULT_VOICE_STYLE = os.getenv("DEFAULT_VOICE_STYLE", "neutral")
DEFAULT_SPEAKER = os.getenv("DEFAULT_SPEAKER", "male")

# Audio quality parameters
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "22050"))
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "128k")
AUDIO_FORMAT = os.getenv("AUDIO_FORMAT", "wav")

# File path constants
DATA_DIR = os.getenv("DATA_DIR", "data")
PDF_UPLOAD_DIR = os.path.join(DATA_DIR, "pdfs")
AUDIO_OUTPUT_DIR = os.path.join(DATA_DIR, "audio")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# Ensure directories exist
for d in [DATA_DIR, PDF_UPLOAD_DIR, AUDIO_OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# Fallbacks and additional config
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))  # For PDF chunking

def get_openai_api_key():
    """Return the OpenAI API key, raise if missing."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required.")
    return OPENAI_API_KEY

def get_elevenlabs_api_key():
    """Return the ElevenLabs API key, or None if not set."""
    return ELEVENLABS_API_KEY

def get_openai_model():
    """Return the configured OpenAI model."""
    return OPENAI_MODEL

def get_elevenlabs_model():
    """Return the configured ElevenLabs model."""
    return ELEVENLABS_MODEL

def get_voice_settings():
    """Return a dict of default voice settings."""
    return {
        "voice": DEFAULT_VOICE,
        "style": DEFAULT_VOICE_STYLE,
        "speaker": DEFAULT_SPEAKER,
        "sample_rate": AUDIO_SAMPLE_RATE,
        "bitrate": AUDIO_BITRATE,
        "format": AUDIO_FORMAT,
    }

def get_paths():
    """Return a dict of important file paths."""
    return {
        "data": DATA_DIR,
        "pdf_upload": PDF_UPLOAD_DIR,
        "audio_output": AUDIO_OUTPUT_DIR,
        "temp": TEMP_DIR,
    }
