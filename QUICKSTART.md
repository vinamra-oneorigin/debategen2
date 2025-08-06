# 🚀 Quick Start Guide (UV)

## Get Running in 3 Steps with UV Package Manager

**Requirements:** Python 3.10+ (due to Gradio 5.0+ dependency)

### 1. **Setup**
```bash
# Install UV if not already installed
pip install uv

# Setup project
python setup.py
```

### 2. **Configure**
```bash
# Edit with your API keys
nano .env
```

**Required Keys:**
- `OPENAI_API_KEY` - From OpenAI dashboard
- `ELEVENLABS_API_KEY` - From ElevenLabs account  
- `ELEVENLABS_MALE_VOICE_ID` - From ElevenLabs voice library
- `ELEVENLABS_FEMALE_VOICE_ID` - From ElevenLabs voice library

### 3. **Launch**
```bash
uv run run.py
```

**Open:** http://localhost:7900

## 💡 First Use

1. **Upload** any PDF document
2. **Configure** settings (defaults work great!)
3. **Click** "Generate Podcast"
4. **Wait** 30-90 seconds
5. **Download** your MP3 podcast!

## 🎯 Best Results

- **PDF Quality**: Text-based PDFs (not scanned images)
- **Length**: 5-15 pages for optimal processing speed
- **Content**: Academic papers, articles, reports work great
- **Duration**: 5-15 minutes for engaging episodes

That's it! Your production-ready podcast generator is live! 🎉