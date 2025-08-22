# 🎙️ Async Podcast Generator

**Transform any PDF document into an engaging AI-generated podcast debate between two hosts.**

This is a production-ready, high-performance application that uses async processing to convert PDFs into professional-quality podcast discussions with AI-generated voices. The application features both a FastAPI backend for programmatic access and a Gradio web interface for interactive use.

## ✨ Features

- **🚀 Lightning Fast**: 3-5x faster than traditional sequential processing
- **🎭 Professional Voices**: High-quality TTS using ElevenLabs Flash v2.5
- **📊 Real-time Progress**: Live updates during processing
- **🎨 Customizable**: Configure tone, length, technical level, and humor
- **💾 Auto Cleanup**: Automatic temporary file management
- **🔄 Robust Processing**: Retry logic and graceful error handling
- **📱 Modern UI**: Clean, intuitive Gradio interface

## 🏗️ Architecture

### **Dual Interface Design**
- **FastAPI Backend** (`app.py`): Production-ready REST API with endpoints for podcast generation and PDF summarization
- **Gradio Demo** (`gradio_demo.py`): Interactive web interface for easy testing and demonstration
- **WebSocket Support**: Real-time voice chat functionality with VAD (Voice Activity Detection)

### **Async Processing Pipeline**
```
PDF Upload → Text Extraction → Content Analysis → Script Generation → Voice Synthesis → Audio Processing → Final Export
     ↓              ↓               ↓                ↓                ↓                ↓              ↓
   Instant      1-2 seconds    3-5 seconds     5-10 seconds     10-30 seconds    2-5 seconds    Instant
```

### **Performance Optimizations**
- **Concurrent API Calls**: Multiple GPT requests processed simultaneously
- **Rate-Limited TTS**: Up to 5 concurrent voice generations with automatic throttling  
- **Parallel Audio Processing**: Audio files loaded and processed concurrently
- **Memory Efficient**: Streaming processing for large files
- **Smart Caching**: Reduces redundant API calls

## 🚀 Quick Start

### 1. **Installation with UV (Recommended)**
```bash
# Clone or download this directory
cd debategen2

# Install UV if not already installed
pip install uv

# Install dependencies with UV (faster and more reliable)
uv sync
```

### 2. **Configuration**
```bash
# Create your environment file
touch .env

# Edit .env with your API keys
nano .env
```

**Required API Keys:**
- **OpenAI API Key**: For content analysis and script generation
- **ElevenLabs API Key**: For high-quality voice synthesis
- **Voice IDs**: Get from your ElevenLabs voice library

### 3. **Launch**

**For Gradio Demo Interface:**
```bash
# Launch the interactive Gradio interface
uv run gradio_demo.py
```
Open your browser to: **http://localhost:7900**

**For FastAPI Backend (Production):**
```bash
# Launch the FastAPI backend server
uv run app.py
```

## 🎛️ Usage

### **Gradio Web Interface (gradio_demo.py)**
1. **📄 Upload PDF**: Drag and drop any PDF document
2. **⚙️ Configure Settings**:
   - **Voice Configuration**: Choose speaker genders
   - **Length**: 2-60 minutes
   - **Tone**: Formal, casual, engaging, or debate
   - **Focus**: Educational, entertaining, debate, or balanced
   - **Technical Level**: Beginner to expert
   - **Humor Level**: 0-10 scale
3. **🚀 Generate**: Click "Generate Podcast" and watch progress
4. **📥 Download**: Get your MP3 file when complete

### **FastAPI Backend (app.py)**
**Available Endpoints:**
- `POST /generate-podcast`: Convert PDF to podcast audio file
- `POST /summarize-pdf`: Get a summary of PDF content
- `GET /health`: Health check endpoint
- `WebSocket /ws/audio-chat`: Real-time voice chat with AI

**API Documentation:** Visit `http://localhost:8000/docs` when running the backend

### **WebSocket Voice Chat**
Real-time voice conversation with AI using Voice Activity Detection (VAD):
- Automatic speech detection and processing
- Integration with Groq and ElevenLabs for fast responses
- Support for conversation context and PDF content discussion

### **Example Output**
- **File Size**: 5-15 MB (depending on length)
- **Quality**: 192kbps MP3, professional audio levels
- **Processing Time**: 30-90 seconds (depending on complexity)
- **Format**: Ready-to-publish podcast episode

## 📁 Project Structure

```
debategen2/
├── app.py                              # FastAPI backend server
├── gradio_demo.py                      # Gradio web interface
├── voice_chat_websocket.py             # WebSocket voice chat functionality
├── vad_utils.py                        # Voice Activity Detection utilities
├── controller/
│   ├── config.py                       # Configuration management
│   ├── utils.py                        # PDF processing utilities
│   ├── content_analyzer_async.py       # Async content analysis
│   ├── script_generator_async.py       # Async script generation
│   ├── voice_generator_async.py        # Async voice synthesis
│   └── audio_processor_async.py        # Async audio processing
├── snakers4_silero-vad_master/         # VAD model files
│   ├── files/
│   │   ├── lang_dict_95.json
│   │   ├── lang_group_dict_95.json
│   │   ├── silero_vad.jit
│   │   └── silero_vad.onnx
│   ├── hubconf.py
│   └── utils_vad.py
├── output_audio/                       # Generated audio files
├── pyproject.toml                      # UV project configuration
├── uv.lock                             # UV lock file
└── README.md                          # This file
```

## ⚡ Performance Metrics

| Component | Processing Method | Speed Improvement |
|-----------|------------------|-------------------|
| **Content Analysis** | Concurrent GPT calls | **3-4x faster** |
| **Script Generation** | Optimized prompts + async | **2x faster** |
| **Voice Synthesis** | 5 concurrent TTS requests | **5x faster** |
| **Audio Processing** | Parallel loading/combining | **2x faster** |
| **Overall Pipeline** | End-to-end async | **3-5x faster** |

## 🛠️ Technical Details

### **Async Controllers**
- **Rate Limiting**: Semaphores prevent API overload
- **Retry Logic**: Exponential backoff for failed requests
- **Error Handling**: Graceful degradation with user feedback
- **Memory Management**: Streaming processing for large files

### **API Integration**
- **OpenAI GPT-4**: Content analysis and script generation
- **ElevenLabs Flash v2.5**: High-speed, high-quality TTS
- **Fallback Chain**: ElevenLabs → OpenAI TTS for reliability

### **Audio Pipeline**
- **Concurrent Generation**: Multiple voice segments simultaneously
- **Professional Quality**: 192kbps MP3 with normalization
- **Smart Timing**: Natural pauses between speakers
- **Auto Cleanup**: Temporary files removed after processing

## 🔧 Configuration Options

### **Voice Settings**
```env
ELEVENLABS_MALE_VOICE_ID=your-male-voice-id
ELEVENLABS_FEMALE_VOICE_ID=your-female-voice-id
ELEVENLABS_MODEL=eleven_flash_v2_5
```

### **Performance Tuning**
- **Concurrent TTS**: Adjust `MAX_CONCURRENT_REQUESTS` in `voice_generator_async.py`
- **API Rate Limits**: Modify `MAX_CONCURRENT_API_CALLS` in async controllers
- **Memory Usage**: Configure chunk sizes in `content_analyzer_async.py`

## 🎯 Use Cases

- **📚 Educational Content**: Convert research papers to accessible discussions
- **📰 News Analysis**: Transform articles into engaging debates
- **📖 Book Summaries**: Create podcast episodes from document summaries
- **🏢 Corporate Training**: Convert training materials to audio format
- **🎓 Academic Discussions**: Generate debates on complex topics

## 📊 Quality Assurance

### **Content Quality**
- **Smart Analysis**: Extracts key points and discussion topics
- **Natural Dialogue**: Conversational flow between distinct personas
- **Customizable Depth**: Adjustable technical level and humor

### **Audio Quality**
- **Professional Voices**: ElevenLabs premium voice models
- **Optimal Pacing**: Natural pauses and speaking rhythm
- **Consistent Levels**: Audio normalization and compression
- **High Fidelity**: 192kbps MP3 for excellent quality

## 🆘 Troubleshooting

### **Common Issues**
1. **"Missing API Keys"**: Ensure `.env` file has valid OpenAI and ElevenLabs keys
2. **"PDF Processing Failed"**: Check PDF is text-based (not scanned image)
3. **"Voice Generation Slow"**: Verify ElevenLabs API quota and voice IDs
4. **"Port Already in Use"**: Change port in `gradio_demo.py` (line 478) or `app.py` (line 314)

### **Performance Tips**
- **Shorter PDFs**: 5-20 pages work best for optimal speed
- **Clear Text**: PDFs with clean, extractable text perform better
- **Stable Internet**: Concurrent API calls require good connectivity
- **Sufficient RAM**: 4GB+ recommended for larger documents

## 🚀 UV Package Manager Benefits

This project now uses **UV** for faster, more reliable dependency management:

### **Why UV?**
- **⚡ 10-100x faster** than pip for dependency resolution
- **🔒 Deterministic builds** with lockfile support
- **🎯 Better dependency resolution** prevents conflicts
- **📦 Unified toolchain** for Python project management
- **🔄 Seamless pip compatibility** - works with existing requirements.txt

### **UV Commands**
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package-name

# Add development dependency
uv add --dev pytest

# Run scripts with UV
uv run python script.py

# Update all dependencies
uv lock --upgrade
```

## 🔄 Updates & Support

This is a production-ready application with:
- ✅ **Comprehensive error handling**
- ✅ **Performance optimizations** 
- ✅ **Professional UI/UX**
- ✅ **Production logging**
- ✅ **Resource management**
- ✅ **Modern UV package management**

For issues or enhancements, check the main project repository or create a new issue.

---

**🎉 Ready to transform your documents into engaging podcasts!**

### Quick Start Commands
```bash
# For interactive demo
uv run gradio_demo.py

# For production API
uv run app.py
```