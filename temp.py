import os
import time
import yt_dlp
from astrapy import DataAPIClient
from langchain_core.language_models import LanguageModelLike
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_astradb import AstraDBVectorStore
import random
import hashlib # Added for unique collection names
import uvicorn # To run the server
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, Response # Added Response for CSS
# Removed StaticFiles and Jinja2Templates as they are no longer needed
from pydantic import BaseModel # For potential request/response models if needed
import re # Import regex for cleaning topic names
import html # Import html module for escaping

# For create_extraction_chain, use this import
from langchain.chains import create_extraction_chain, RetrievalQA

# Update chat prompts import
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.messages import HumanMessage


# For summarize chain, stick with this import
from langchain.chains.summarize import load_summarize_chain

# Add tenacity for better retry handling
from typing import List, Optional
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted


# --- HTML and CSS Content ---

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Transcript Analyzer</title>
    <link rel="stylesheet" href="/style.css">
</head>
<body>
    <div class="container">
        <h1>YouTube Transcript Analyzer</h1>
        <form action="/process" method="post" id="process-form">
            <label for="video_url">Enter YouTube Video URL:</label>
            <input type="url" id="video_url" name="video_url" required>
            <button type="submit">Analyze</button>
        </form>
        <div id="loading" style="display: none;">
            <p>Processing... This may take a few minutes.</p>
            <div class="loader"></div>
        </div>
    </div>
    <script>
        const form = document.getElementById('process-form');
        const loadingDiv = document.getElementById('loading');
        if (form) {
            form.addEventListener('submit', () => {
                if (loadingDiv) {
                    loadingDiv.style.display = 'block';
                }
            });
        }
    </script>
</body>
</html>
"""

RESULTS_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Results</title>
    <link rel="stylesheet" href="/style.css">
</head>
<body>
    <div class="container">
        <h1>Analysis Results for Video ID: {video_id}</h1>
        <a href="/">Analyze another video</a>
        <h2>Extracted & Expanded Topics</h2>
        {content}
    </div>
</body>
</html>
"""

STYLE_CSS = """
body {
    font-family: sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 20px;
    background-color: #f4f4f4;
    color: #333;
}

.container {
    max-width: 800px;
    margin: auto;
    background: #fff;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

h1, h2 {
    color: #333;
    margin-bottom: 20px;
    text-align: center;
}

form {
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
}

label {
    display: block;
    margin-bottom: 8px;
    font-weight: bold;
}

input[type="url"] {
    width: 80%;
    padding: 12px;
    margin-bottom: 15px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 16px;
}

button {
    display: inline-block;
    background-color: #5cb85c;
    color: white;
    padding: 12px 25px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s ease;
}

button:hover {
    background-color: #4cae4c;
}

.topic {
    background-color: #e9e9e9;
    margin-bottom: 20px;
    padding: 15px;
    border-radius: 5px;
    border-left: 5px solid #5cb85c;
}

.topic h3 {
    margin-top: 0;
    color: #444;
}

.error {
    color: #d9534f;
    background-color: #f2dede;
    border: 1px solid #ebccd1;
    padding: 15px;
    border-radius: 4px;
    margin-bottom: 20px;
}

a {
    display: inline-block;
    margin-bottom: 20px;
    color: #0275d8;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

#loading {
    text-align: center;
    margin-top: 20px;
}

.loader {
    border: 5px solid #f3f3f3; /* Light grey */
    border-top: 5px solid #5cb85c; /* Green */
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 10px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
"""

# -----------------------------
# 1. Setup: AssemblyAI Transcription
# -----------------------------
# Load environment variables
import assemblyai as aai
from dotenv import load_dotenv
load_dotenv()

# Get API keys from environment variables
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_ENDPOINT")

if not ASSEMBLYAI_API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY environment variable is required")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable is required")
if not ASTRA_DB_TOKEN or not ASTRA_DB_ENDPOINT:
    raise ValueError("ASTRA_DB_TOKEN and ASTRA_DB_ENDPOINT environment variables are required")

aai.settings.api_key = ASSEMBLYAI_API_KEY

def download_audio(video_url, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    else:
        video_id = video_url.split("/")[-1].split("?")[0]

    output_path = os.path.join(output_dir, f"{video_id}")

    if os.path.exists(output_path + ".mp3"):
        print(f"✅ Audio already downloaded: {output_path}.mp3")
        return output_path + ".mp3"

    ydl_opts = {
        'format': 'ba/bestaudio',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'noplaylist': True,
        # FFmpeg will be available in PATH after installation via render.yaml
        'quiet': True,
        'no_warnings': True,
    }
    try:
        print(f"Attempting to download audio to: {output_path}.mp3")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_occurred = False
            error_message = "Unknown yt-dlp error"
            def error_hook(d):
                nonlocal error_occurred, error_message
                if d['status'] == 'error':
                    print(f"DEBUG: yt-dlp reported an error: {d.get('error', 'Unknown error')}")
                    error_message = d.get('error', 'Unknown error')
                    error_occurred = True
                elif d['status'] == 'finished':
                     print(f"DEBUG: yt-dlp finished processing {d.get('filename')}")
                elif d['status'] == 'downloading':
                     pass

            ydl.add_progress_hook(error_hook)
            ydl.download([video_url])

        if error_occurred:
             raise Exception(f"yt-dlp failed: {error_message}")

        final_path = output_path + ".mp3"
        if not os.path.exists(final_path):
            print(f"DEBUG: File not found immediately at {final_path}, waiting...")
            time.sleep(3)
            if not os.path.exists(final_path):
                files_in_dir = []
                try:
                    files_in_dir = os.listdir(output_dir)
                except FileNotFoundError:
                     print(f"DEBUG: Output directory '{output_dir}' not found.")
                related_files = [f for f in files_in_dir if video_id in f]
                print(f"DEBUG: Files found in '{output_dir}' containing video ID '{video_id}': {related_files}")
                raise FileNotFoundError(f"Failed to create audio file at the expected path after download and wait: {final_path}")

        print(f"✅ Audio downloaded successfully: {final_path}")
        return final_path
    except Exception as e:
        print(f"❌ Error downloading audio: {str(e)}")
        raise

def get_or_create_transcript(video_id, audio_path):
    try:
        os.makedirs("transcripts", exist_ok=True)
        transcript_path = os.path.join("transcripts", f"{video_id}.txt")

        if os.path.exists(transcript_path):
            print(f"📝 Loading existing transcript for {video_id}")
            with open(transcript_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    return content
                print("Existing transcript is empty, recreating...")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at {audio_path}")

        print(f"🔍 Using audio file: {audio_path}")
        print("🎯 Starting transcription...")
        transcriber = aai.Transcriber()

        try:
            transcript = transcriber.transcribe(audio_path)
            print("✅ Transcription complete.")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript.text)
            return transcript.text
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "authentication" in error_str.lower():
                print("❌ AssemblyAI authentication error. Check your API key.")
            elif "timeout" in error_str.lower():
                print("❌ Transcription request timed out. The audio file might be too large.")
            else:
                print(f"❌ Error during AssemblyAI transcription: {error_str}")
            raise

    except Exception as e:
        print(f"❌ Error during transcription process: {str(e)}")
        raise

# -----------------------------
# 2. Google Gemini LLM (Rate Limited)
# -----------------------------
class RateLimitedGemini(LanguageModelLike):
    def __init__(self, api_key, model, temperature=0, request_timeout=180):
        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            request_timeout=request_timeout,
            max_retries=3
        )
        self.min_delay = 1.0
        self.last_request_time = 0
        
    @retry(
        retry=retry_if_exception_type(ResourceExhausted),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def invoke(self, *args, **kwargs):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed + random.uniform(0.1, 0.5)
            print(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        try:
            self.last_request_time = time.time()
            return self.llm.invoke(*args, **kwargs)
        except ResourceExhausted as e:
            print(f"API quota exceeded. Retrying after exponential backoff: {str(e)}")
            raise
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            raise
            
    def run(self, query, *args, **kwargs):
        from langchain_core.messages import HumanMessage
        result = self.invoke([HumanMessage(content=query)])
        return result

# Initialize LLMs
gemini4_base = ChatGoogleGenerativeAI( # Keep base for non-rate-limited tasks if needed
    api_key=GOOGLE_API_KEY,
    model="gemini-1.5-flash", # Using 1.5 flash as 2.0 might not exist
    temperature=0,
    request_timeout=180
)

gemini4_rate_limited = RateLimitedGemini( # Use this for QA
    api_key=GOOGLE_API_KEY,
    model="gemini-1.5-flash",
    temperature=0,
    request_timeout=180
)

# Initialize NVIDIA Embeddings
nvidia_embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v1",
    api_key=NVIDIA_API_KEY,
    truncate="NONE",
)

# --- FastAPI Setup ---
app = FastAPI()

# -----------------------------
# Core Processing Logic
# -----------------------------
async def process_video(video_url: str):
    """Downloads, transcribes, and analyzes a YouTube video."""
    video_id = "unknown"
    try:
        # (a) Get Video ID and Transcript
        if "v=" in video_url:
            video_id_full = video_url.split("v=")[1].split("&")[0]
        else:
            video_id_full = video_url.split("/")[-1].split("?")[0]
        video_id = video_id_full[:8]
        print(f"Using Video ID suffix for collection: {video_id}")
        audio_path = download_audio(video_url, output_dir="downloads")
        transcript = get_or_create_transcript(video_id_full, audio_path)

        # (b) Split Transcript for Topic Extraction
        text_splitter_large = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "], chunk_size=10000, chunk_overlap=2200
        )
        docs_large = text_splitter_large.create_documents([transcript])
        print(f"Docs for topic extraction: {len(docs_large)}")

        # (c) Define Topic Extraction Prompts
        map_template = """Extract topic names and a brief 1-sentence description from the transcript. Format as: Topic: Brief Description."""
        system_message_prompt_map = SystemMessagePromptTemplate.from_template(map_template)
        human_message_prompt_map = HumanMessagePromptTemplate.from_template("Transcript: {text}")
        chat_prompt_map = ChatPromptTemplate.from_messages([system_message_prompt_map, human_message_prompt_map])
        combine_template = """Consolidate the extracted topics, deduplicate, and format as: Topic: Brief Description."""
        system_message_prompt_combine = SystemMessagePromptTemplate.from_template(combine_template)
        human_message_prompt_combine = HumanMessagePromptTemplate.from_template("Transcript: {text}")
        chat_prompt_combine = ChatPromptTemplate.from_messages([system_message_prompt_combine, human_message_prompt_combine])

        # (d) Extract Topics
        try:
            print("Starting topic extraction...")
            summarize_chain = load_summarize_chain(
                gemini4_base, chain_type="map_reduce",
                map_prompt=chat_prompt_map, combine_prompt=chat_prompt_combine,
            )
            # Use invoke instead of run for newer LangChain versions
            topics_result = summarize_chain.invoke({"input_documents": docs_large})
            topics_found = topics_result.get('output_text', '') # Adjust key if necessary
            print("Extracted Topics Text:")
            print(topics_found)
        except Exception as e:
            print(f"Error during topic extraction: {str(e)}")
            topics_found = "Topic 1: General discussion.\nTopic 2: Key points."

        # (e) Structure and Clean Topics
        topics_structured = []
        topic_name_cleaner = re.compile(r"^\s*[\*#-]+\s*\**(.+?)\**\s*$", re.MULTILINE)
        for line in topics_found.split('\n'):
            if ':' in line:
                topic_name_raw, description = line.split(':', 1)
                match = topic_name_cleaner.match(topic_name_raw.strip())
                cleaned_topic_name = match.group(1).strip() if match else topic_name_raw.strip()
                cleaned_topic_name = re.sub(r"[\*_`]", "", cleaned_topic_name) # Remove leftover markdown
                if cleaned_topic_name: # Ensure topic name is not empty after cleaning
                    topics_structured.append({
                        'topic_name': cleaned_topic_name,
                        'description': description.strip(),
                    })
        print("Structured Topics:")
        print(topics_structured)

        # (f) Split Transcript for QA
        text_splitter_small = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
        docs_small = text_splitter_small.create_documents([transcript])
        print(f"Docs for QA: {len(docs_small)}")

        # (g) Setup Vector Store
        try:
            astra_token = ASTRA_DB_TOKEN
            astra_endpoint = ASTRA_DB_ENDPOINT
            collection_name = f"transcript_{video_id}"
            print(f"Connecting to Astra DB collection: {collection_name}")
            vector_store = AstraDBVectorStore(
                embedding=nvidia_embeddings, collection_name=collection_name,
                token=astra_token, api_endpoint=astra_endpoint, namespace="default_keyspace"
            )
            print(f"Adding {len(docs_small)} documents to vector store...")
            vector_store.add_documents(docs_small)
            print("✅ Documents added successfully.")
            retriever = vector_store.as_retriever(k=4)
            print("✅ Retriever created.")
        except Exception as e:
            print(f"❌ Error setting up Astra DB Vector Store: {str(e)}")
            raise

        # (i) Setup QA Chain
        system_template_qa = """Given the transcript context, summarize the topic provided in 5 sentences or less. Focus only on relevant information from the context. Context: {context}"""
        messages_qa = [
            SystemMessagePromptTemplate.from_template(system_template_qa),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        CHAT_PROMPT_QA = ChatPromptTemplate.from_messages(messages_qa)
        qa = RetrievalQA.from_chain_type(
            llm=gemini4_rate_limited, # Use rate-limited LLM
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={'prompt': CHAT_PROMPT_QA}
        )

        # (j) Expand Topics using QA Chain
        print("\nExpanding Topics:")
        expanded_results = []
        for topic in topics_structured[:5]: # Limit results
            query = f"{topic['topic_name']}: {topic['description']}"
            expanded_topic_text = "Error processing this topic."
            try:
                print(f"Running QA for topic: {topic['topic_name']}")
                # No need for manual rate limit here if using RateLimitedGemini in QA chain
                qa_result = qa.invoke({"query": query})
                expanded_topic_text = qa_result.get('result', 'No result found.')
            except Exception as e:
                print(f"Error processing topic '{topic['topic_name']}': {str(e)}")

            expanded_results.append({
                "topic_name": topic['topic_name'],
                "description": topic['description'],
                "expanded_summary": expanded_topic_text
            })
            print(f"--- Topic: {topic['topic_name']} ---")
            print(f"Expanded Summary:\n{expanded_topic_text}")
            print("-" * 20)
            # Optional: add a small delay anyway if needed
            # time.sleep(0.5)

        return video_id, expanded_results

    except Exception as e:
        print(f"❌ An error occurred during video processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main input form."""
    return HTMLResponse(content=INDEX_HTML)

@app.get("/style.css")
async def get_css():
    """Serves the CSS content."""
    return Response(content=STYLE_CSS, media_type="text/css")

@app.post("/process", response_class=HTMLResponse)
async def handle_process(video_url: str = Form(...)):
    """Handles form submission, processes the video, and shows results."""
    results_data = []
    error_message = None
    processed_video_id = "N/A"
    results_html_content = ""

    try:
        print(f"Received request to process URL: {video_url}")
        processed_video_id, results_data = await process_video(video_url)
        print(f"Processing complete for video ID: {processed_video_id}")

        # Build HTML content for results
        if results_data:
            for topic in results_data:
                # Use html.escape for safer HTML escaping
                topic_name_safe = html.escape(topic.get('topic_name', ''))
                description_safe = html.escape(topic.get('description', ''))
                # Escape and replace newlines with <br> for the summary
                summary_safe = html.escape(topic.get('expanded_summary', '')).replace('\n', '<br>')

                results_html_content += f"""
                <div class="topic">
                    <h3>{topic_name_safe}</h3>
                    <p><strong>Original Description:</strong> {description_safe}</p>
                    <p><strong>Expanded Summary:</strong></p>
                    <p>{summary_safe}</p>
                </div>
                """
        else:
             results_html_content = "<p>No topics were extracted or expanded.</p>"

    except HTTPException as http_exc:
        error_message = html.escape(http_exc.detail) # Escape error message
        print(f"HTTPException during processing: {error_message}")
        results_html_content = f'<p class="error">Error during processing: {error_message}</p>'
    except Exception as e:
        error_message = html.escape(f"An unexpected error occurred: {str(e)}") # Escape error message
        print(f"Unexpected error during processing: {error_message}")
        results_html_content = f'<p class="error">An unexpected error occurred: {error_message}</p>'
        # Optionally log the full traceback here for debugging

    # Format the final HTML using the template string
    final_html = RESULTS_HTML_TEMPLATE.format(
        video_id=html.escape(processed_video_id), # Escape video ID
        content=results_html_content # Content is already escaped or generated safely
    )
    return HTMLResponse(content=final_html)


# --- Main Execution ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    print(f"Starting FastAPI server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
