# YouTube Transcript Analyzer

A FastAPI application that analyzes YouTube video transcripts using AI to extract and summarize key topics.

## Features

- YouTube video audio extraction and transcription
- Topic extraction using Google's Gemini AI
- Topic expansion and summarization with vector-based QA
- Clean web interface with real-time processing feedback
- Rate-limited AI interactions for stability

## Requirements

- Python 3.11
- FFmpeg
- API Keys for:
  - AssemblyAI (Transcription)
  - Google Gemini (AI)
  - NVIDIA AI Endpoints (Embeddings)
  - AstraDB (Vector Database)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Meetpatel006/edubiz-v1.git
cd youtube-transcript-analyzer
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

4. Configure your environment variables in `.env`:
```
ASSEMBLYAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
NVIDIA_API_KEY=your_key_here
ASTRA_DB_TOKEN=your_token_here
ASTRA_DB_ENDPOINT=your_endpoint_here
```

## Running Locally

```bash
uvicorn temp:app --host 0.0.0.0 --port 8001
```

Visit `http://localhost:8001` in your browser.

## Deployment on Render

1. Fork this repository
2. Create a new Web Service on Render
3. Connect your forked repository
4. Add the required environment variables:
   - ASSEMBLYAI_API_KEY
   - GOOGLE_API_KEY
   - NVIDIA_API_KEY
   - ASTRA_DB_TOKEN
   - ASTRA_DB_ENDPOINT
5. Deploy!

The `render.yaml` file is already configured to:
- Install dependencies
- Install FFmpeg
- Start the FastAPI server

## Using the API

### Web Interface
Visit the root URL (`/`) to use the web interface. Simply paste a YouTube URL and click "Analyze".

### API Endpoints
- `GET /`: Web interface
- `POST /process`: Process a YouTube video URL
- `GET /style.css`: Get the CSS styling

## Project Structure

- `temp.py`: Main application file
- `requirements.txt`: Python dependencies
- `runtime.txt`: Python version specification
- `render.yaml`: Render deployment configuration
- `.env`: Environment variables (not in repo)
- `downloads/`: Temporary audio files
- `transcripts/`: Generated transcripts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use this project as you wish.
