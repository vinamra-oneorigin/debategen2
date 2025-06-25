"""
debate2.py

Production Podcast Debate Generator

Fast, optimized Gradio app that converts PDFs to AI-generated debate podcasts
with async processing, real-time progress tracking, and professional UX.
"""

import os
import time
import asyncio
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import gradio as gr

# Import async controllers
from controller.utils import extract_text_from_pdf, clean_extracted_text
from controller.content_analyzer_async import (
    chunk_text_for_gpt, 
    analyze_content_completely_async
)
from controller.script_generator_async import generate_podcast_script_async
from controller.voice_generator_async import process_dialogue_markers_async
from controller.audio_processor_async import (
    process_complete_audio_async,
    cleanup_temp_files_async
)

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Access control
ACCESS_PASSWORD = "podcast2024"  # Change this to your desired password

async def process_pdf_to_podcast(
    pdf_file,
    voice_config: str,
    length_minutes: int,
    tone: str,
    focus: str,
    technical_level: str,
    humor_level: int,
    progress
) -> Tuple[Optional[str], str]:
    """
    Complete async pipeline to convert PDF to podcast.
    Returns (audio_file_path, status_message)
    """
    start_time = time.time()
    
    try:
        # Update progress
        progress(0.1, desc="Extracting text from PDF")
        
        # Step 1: Extract and clean text
        if not pdf_file or not os.path.exists(pdf_file.name):
            return None, "Error: No PDF file provided or file not found"
        
        raw_text = extract_text_from_pdf(pdf_file.name)
        cleaned_text = clean_extracted_text(raw_text)
        
        if len(cleaned_text) < 100:
            return None, "Error: PDF contains insufficient text content"
        
        text_chunks = chunk_text_for_gpt(cleaned_text, max_tokens=4000)
        
        progress(0.2, desc=f"Analyzing content ({len(text_chunks)} chunks)")
        
        # Step 2: Concurrent content analysis
        analysis = await analyze_content_completely_async(text_chunks)
        
        if not analysis["key_points"] or not analysis["summary"]:
            return None, "Error: Failed to analyze PDF content"
        
        progress(0.4, desc="Generating podcast script")
        
        # Step 3: Generate script
        topic_title = f"Discussion of {os.path.basename(pdf_file.name).replace('.pdf', '')}"
        
        script_result = await generate_podcast_script_async(
            podcast_topic=topic_title,
            target_length_minutes=length_minutes,
            tone=tone,
            content_focus=focus,
            technical_level=technical_level,
            inclusion_of_humor=humor_level
        )
        
        if not script_result or not script_result.get("script"):
            return None, "Error: Failed to generate podcast script"
        
        script = script_result["script"]
        
        progress(0.6, desc=f"Generating voices ({len(script)} segments)")
        
        # Step 4: Create voice mapping
        speakers = list(set([segment.get('speaker', 'Unknown') for segment in script]))
        voice_map = {}
        
        if voice_config == 'male_female':
            for i, speaker in enumerate(speakers):
                voice_map[speaker] = 'male' if i % 2 == 0 else 'female'
        else:  # female_male
            for i, speaker in enumerate(speakers):
                voice_map[speaker] = 'female' if i % 2 == 0 else 'male'
        
        # Step 5: Generate audio concurrently
        audio_files = await process_dialogue_markers_async(
            script=script,
            voice_map=voice_map,
            speed=1.0,
            output_format="mp3"
        )
        
        if not audio_files:
            return None, "Error: Failed to generate audio segments"
        
        progress(0.8, desc="Processing and combining audio")
        
        # Step 6: Process and combine audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"podcast_{os.path.basename(pdf_file.name).replace('.pdf', '')}_{timestamp}"
        
        final_audio_path = await process_complete_audio_async(
            audio_paths=audio_files,
            pause_ms=500,
            effects=["normalize"],
            output_filename=filename,
            output_format="mp3",
            bitrate="192k"
        )
        
        if not final_audio_path:
            return None, "Error: Failed to combine audio segments"
        
        progress(0.9, desc="Cleaning up temporary files")
        
        # Step 7: Cleanup
        deleted_count = await cleanup_temp_files_async(audio_files)
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        file_size = os.path.getsize(final_audio_path) / (1024 * 1024)  # MB
        
        progress(1.0, desc="Podcast generation complete")
        
        status_msg = (
            f"**Podcast Generated Successfully**\n\n"
            f"**File:** {os.path.basename(final_audio_path)}\n"
            f"**Size:** {file_size:.1f} MB\n"
            f"**Processing Time:** {total_time:.1f} seconds\n"
            f"**Segments:** {len(script)} dialogue turns\n"
            f"**Cleaned up:** {deleted_count} temporary files\n\n"
            f"**Ready to download and listen**"
        )
        
        return final_audio_path, status_msg
        
    except Exception as e:
        error_msg = f"**Error:** {str(e)}"
        return None, error_msg

def create_gradio_interface():
    """
    Create and configure the Gradio interface.
    """
    
    with gr.Blocks(
        title="Podcast/Debate Generator",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        .progress-container {
            margin: 20px 0;
        }
        """
    ) as app:
        
        gr.Markdown("""
        # Podcast/Debate Generator
        
        Transform any PDF document into an engaging AI-generated podcast debate between two hosts.
        Upload your PDF and configure the podcast parameters below.
        """)
        
        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### Upload & Configuration")
                
                pdf_upload = gr.File(
                    label="Upload PDF Document",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                voice_config = gr.Dropdown(
                    choices=["male_female", "female_male"],
                    value="male_female",
                    label="Voice Configuration",
                    info="First speaker to Second speaker"
                )
                
                length_slider = gr.Slider(
                    minimum=2,
                    maximum=60,
                    value=10,
                    step=1,
                    label="Podcast Length (minutes)"
                )
                
                tone_dropdown = gr.Dropdown(
                    choices=["formal", "casual", "engaging", "debate"],
                    value="engaging",
                    label="Tone"
                )
                
                focus_dropdown = gr.Dropdown(
                    choices=["educational", "entertaining", "debate", "balanced"],
                    value="balanced",
                    label="Content Focus"
                )
                
                technical_dropdown = gr.Dropdown(
                    choices=["beginner", "intermediate", "expert", "general"],
                    value="general",
                    label="Technical Level"
                )
                
                humor_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=2,
                    step=1,
                    label="Humor Level"
                )
                
                gr.Markdown("### Access Control")
                
                password_input = gr.Textbox(
                    label="Access Password",
                    type="password",
                    placeholder="Enter password to enable generation",
                    info="Password required for podcast generation"
                )
                
                generate_btn = gr.Button(
                    "Generate Podcast",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Outputs
            with gr.Column(scale=1):
                gr.Markdown("### Output & Progress")
                
                status_display = gr.Markdown("")
                
                audio_output = gr.Audio(
                    label="Generated Podcast",
                    type="filepath"
                )
                
                download_btn = gr.File(
                    label="Download Podcast",
                    visible=False,
                    interactive=False
                )
        
        # Event handlers
        def process_podcast(pdf_file, voice_cfg, length, tone, focus, tech_level, humor, password, progress=gr.Progress()):
            """Async wrapper for podcast processing with progress updates."""
            
            # Validate password first
            if password != ACCESS_PASSWORD:
                return (
                    None,  # audio_output
                    "**Error:** Invalid password. Please enter the correct password to generate podcasts.",  # status_display
                    None  # download_btn
                )
            
            # Process the podcast
            audio_path, status_msg = asyncio.run(process_pdf_to_podcast(
                pdf_file, voice_cfg, length, tone, focus, tech_level, humor, progress
            ))
            
            # Update interface
            if audio_path:
                return (
                    audio_path,  # audio_output
                    status_msg,  # status_display
                    audio_path  # download_btn (same as audio_output for download)
                )
            else:
                return (
                    None,  # audio_output
                    status_msg,  # status_display
                    None  # download_btn
                )
        
        generate_btn.click(
            fn=process_podcast,
            inputs=[
                pdf_upload,
                voice_config,
                length_slider,
                tone_dropdown,
                focus_dropdown,
                technical_dropdown,
                humor_slider,
                password_input
            ],
            outputs=[
                audio_output,
                status_display,
                download_btn
            ],
            show_progress=True
        )
        
    
    return app

def main():
    """
    Launch the Gradio application.
    """
    print("Starting Podcast Debate Generator...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Check environment variables
    required_env_vars = ["OPENAI_API_KEY", "ELEVENLABS_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file before running the app.")
        return
    
    print("Environment variables configured")
    
    # Create and launch the app
    app = create_gradio_interface()
    
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7900,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()