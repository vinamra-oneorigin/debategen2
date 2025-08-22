"""
api_server.py

FastAPI Server for Podcast Generation API

Provides REST endpoints for:
- PDF upload and podcast generation
- Health checks and status monitoring
"""

import os
import time
import asyncio
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import existing controllers
from controller.utils import extract_text_from_pdf, clean_extracted_text
from controller.content_analyzer_async import (
    chunk_text_for_gpt, 
    analyze_content_completely_async,
    summarize_transcript_async,
)
from controller.script_generator_async import generate_podcast_script_async
from controller.voice_generator_async import process_dialogue_markers_async
from controller.audio_processor_async import (
    process_complete_audio_async,
    cleanup_temp_files_async
)
from voice_chat_websocket import router as audio_ws_router

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_audio")
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# FastAPI app initialization
app = FastAPI(
    title="Podcast Generator API",
    description="AI-powered podcast generation from PDF documents with audio chat functionality",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static directory for audio files
app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")

# Include WebSocket router
app.include_router(audio_ws_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Background task for cleanup
async def cleanup_old_files():
    """Clean up old audio files (older than 1 hour)"""
    try:
        current_time = time.time()
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                if file_age > 3600:  # 1 hour
                    os.remove(file_path)
    except Exception as e:
        print(f"Cleanup error: {e}")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/summarize-pdf")
async def summarize_pdf(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(..., description="PDF to summarize")
):
    """
    Summarize the uploaded PDF content (not the podcast transcript).
    Returns JSON with a single field: {"summary": string}.
    """
    try:
        if not pdf_file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        temp_pdf_path = os.path.join(TEMP_DIR, f"temp_{int(time.time())}_{pdf_file.filename}")
        with open(temp_pdf_path, "wb") as buffer:
            content = await pdf_file.read()
            buffer.write(content)

        raw_text = extract_text_from_pdf(temp_pdf_path)
        cleaned_text = clean_extracted_text(raw_text)
        if len(cleaned_text) < 100:
            raise HTTPException(status_code=400, detail="PDF contains insufficient text content")

        text_chunks = chunk_text_for_gpt(cleaned_text, max_tokens=4000)
        analysis = await analyze_content_completely_async(text_chunks)
        if not analysis.get("summary"):
            raise HTTPException(status_code=500, detail="Failed to summarize PDF content")

        # Cleanup temp file
        background_tasks.add_task(lambda: os.remove(temp_pdf_path) if os.path.exists(temp_pdf_path) else None)

        return JSONResponse(content={"summary": analysis.get("summary", "")})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate-podcast")
async def generate_podcast_file(
    background_tasks: BackgroundTasks,
    request: Request,
    pdf_file: UploadFile = File(..., description="PDF file to convert to podcast"),
    voice_config: str = Form("male_female"),
    length_minutes: int = Form(25),
    tone: str = Form("engaging"),
    focus: str = Form("balanced"),
    technical_level: str = Form("intermediate"),
    humor_level: int = Form(4)
):
    """
    Generate a podcast from the uploaded PDF and return the MP3 file directly.
    Includes a concise transcript-derived summary in headers.
    """
    start_time = time.time()

    try:
        if not pdf_file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        temp_pdf_path = os.path.join(TEMP_DIR, f"temp_{int(time.time())}_{pdf_file.filename}")
        with open(temp_pdf_path, "wb") as buffer:
            content = await pdf_file.read()
            buffer.write(content)

        raw_text = extract_text_from_pdf(temp_pdf_path)
        cleaned_text = clean_extracted_text(raw_text)
        if len(cleaned_text) < 100:
            raise HTTPException(status_code=400, detail="PDF contains insufficient text content")

        text_chunks = chunk_text_for_gpt(cleaned_text, max_tokens=4000)
        analysis = await analyze_content_completely_async(text_chunks)
        if not analysis.get("summary"):
            raise HTTPException(status_code=500, detail="Failed to analyze PDF content")

        topic_title = f"Discussion of {Path(pdf_file.filename).stem}"
        script_result = await generate_podcast_script_async(
            podcast_topic=topic_title,
            target_length_minutes=length_minutes,
            tone=tone,
            content_focus=focus,
            technical_level=technical_level,
            inclusion_of_humor=humor_level
        )

        if not script_result or not script_result.get("script"):
            raise HTTPException(status_code=500, detail="Failed to generate podcast script")

        script = script_result["script"]

        speakers = list(set([segment.get('speaker', 'Unknown') for segment in script]))
        voice_map = {}
        if voice_config == 'male_female':
            for i, speaker in enumerate(speakers):
                voice_map[speaker] = 'male' if i % 2 == 0 else 'female'
        else:
            for i, speaker in enumerate(speakers):
                voice_map[speaker] = 'female' if i % 2 == 0 else 'male'

        audio_files = await process_dialogue_markers_async(
            script=script,
            voice_map=voice_map,
            speed=1.0,
            output_format="mp3"
        )
        if not audio_files:
            raise HTTPException(status_code=500, detail="Failed to generate audio segments")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"podcast_{Path(pdf_file.filename).stem}_{timestamp}"
        final_audio_path = await process_complete_audio_async(
            audio_paths=audio_files,
            pause_ms=500,
            effects=["normalize"],
            output_filename=filename,
            output_format="mp3",
            bitrate="192k"
        )
        if not final_audio_path:
            raise HTTPException(status_code=500, detail="Failed to combine audio segments")

        background_tasks.add_task(cleanup_temp_files_async, audio_files)
        background_tasks.add_task(lambda: os.remove(temp_pdf_path) if os.path.exists(temp_pdf_path) else None)

        total_time = time.time() - start_time
        file_size = os.path.getsize(final_audio_path) / (1024 * 1024)
        
        return FileResponse(
            path=final_audio_path,
            media_type="audio/mpeg",
            filename=f"{filename}.mp3",
            headers={
                "X-Processing-Time": str(round(total_time, 2)),
                "X-Segments-Count": str(len(script)),
                "X-File-Size-MB": str(round(file_size, 2)),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

######### Demo endpoints for testing without actual processing #########

# @app.post("/summarize-pdf")
# async def summarize_pdf_demo(
#     pdf_file: UploadFile
# ):
#     """
#     Demo endpoint: returns an empty summary string regardless of input.
#     """
#     return JSONResponse(content={"summary": "This document talks about the types of families in the animal kingdom"})


# @app.post("/generate-podcast")
# async def generate_podcast_demo(
#     background_tasks: BackgroundTasks,
#     request: Request,
#     pdf_file:UploadFile
# ):
#     """
#     Demo endpoint: returns a fixed MP3 from the Desktop path provided.
#     """
#     filename = "podcast_Chapter 7_ The Photoelectric Effect - Light's Quantum Revolution_20250818_162006.mp3"
#     desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
#     file_path = os.path.join(desktop_dir, filename)

#     if not os.path.exists(file_path):
#         raise HTTPException(
#             status_code=404,
#             detail=f"Demo MP3 not found at {file_path}. Place the file on Desktop or adjust the path."
#         )

#     return FileResponse(
#         path=file_path,
#         media_type="audio/mpeg",
#         filename=filename
#     )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("🎙️ Podcast Generator API starting up...")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🗂️ Temp directory: {TEMP_DIR}")
    
    # Check environment variables
    required_env_vars = ["OPENAI_API_KEY", "ELEVENLABS_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️ Warning: Missing environment variables: {', '.join(missing_vars)}")
    else:
        print("✅ Environment variables configured")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Podcast Generator API shutting down...")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )