# My XTTS API Server (Vietnamese Optimized)

This project is a customized version of [daswer123/xtts-api-server](https://github.com/daswer123/xtts-api-server), specifically optimized and tailored for **Vietnamese text-to-speech (TTS)** and production environments.

## 🌟 Key Features & Customizations

1. **Vietnamese Text Normalization**: Built-in support for normalizing Vietnamese text (expanding abbreviations, reading numbers, dates, punctuation handling) using a custom `vi_normalizer` pipeline.
2. **Thivux Model Integration**: Configured to seamlessly use the Thivux Vietnamese XTTS model by default.
3. **SSE Progress Streaming**: Enhanced the `/tts_from_files` API endpoint to return server-sent events (SSE) indicating real-time audio rendering progress, perfect for web frontends to display loading bars.
4. **Speaker Audio Pre-processing**: Automatically trims silence and normalizes volume of reference speaker audio files using `librosa`, improving the stability and quality of voice cloning.
5. **Asynchronous Request Queue**: Built-in request queuing mechanism to prevent server crashes or memory issues during high concurrent traffic.

## 📦 Installation

It is recommended to run this server in a Python virtual environment.

```bash
# Clone the repository
git clone https://github.com/namnguyenanalyst/my_xtts_api_server
cd my_xtts_api_server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Starting the Server

```bash
python -m xtts_api_server --listen
```

This will start the server on `http://0.0.0.0:8020`. 

You can check the interactive API documentation at:
- **Swagger UI**: [http://localhost:8020/docs](http://localhost:8020/docs)

## 📖 API Usage Example

### Generating Audio with Progress Tracking

The main endpoint for generating audio is `/tts_to_audio/` or `/tts_to_file`. To use the new SSE progress tracking, use the customized `/tts_from_files` endpoint which yields progress chunks before completing the audio generation.

```javascript
const response = await fetch("http://localhost:8020/tts_from_files", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: "Xin chào, đây là hệ thống tổng hợp giọng nói tiếng Việt.",
    speaker_name: "your_speaker_wav_file_name_without_extension",
    language: "vi"
  })
});

// Read the stream to get progress updates (e.g. {"progress": 45})
```

## 👥 Adding New Speakers

Simply add a `.wav` file (mono, 22050Hz, 16-bit recommended) of the person's voice to the `speakers/` folder. The system will automatically preprocess the audio (trim silence, normalize volume) before using it for cloning. You can then pass the filename (without the `.wav` extension) as the `speaker_name` parameter in your API requests.

## 🙏 Credits

- Original Repository: [daswer123/xtts-api-server](https://github.com/daswer123/xtts-api-server)
- Core TTS Engine: [Coqui TTS (XTTSv2)](https://github.com/coqui-ai/TTS)
