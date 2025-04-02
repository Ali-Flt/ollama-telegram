from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from piper import PiperVoice
import io
import logging
import os
from multiprocessing import Process
from pathlib import Path
import wave
import tempfile
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Piper TTS Server", version="1.0")

log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = logging.getLevelName(log_level_str)
logging.basicConfig(level=log_level)

cuda = os.getenv("USE_CUDA", "False")
VOICE_MODEL = os.getenv("TTS_VOICE", "en_US-lessac-medium")
VOICE_LANG_1 = VOICE_MODEL.split('_')[0]
VOICE_LANG_2 = VOICE_MODEL.split('-')[0]
VOICE_NAME = VOICE_MODEL.split('-')[1]
VOICE_STYLE = VOICE_MODEL.split('-')[-1]
VOICE_MODEL += ".onnx"
DATA_DIR = os.getenv("PIPER_DATA_DIR", "/app/voices")
MODEL_PATH = Path(DATA_DIR) / VOICE_MODEL
JSON_PATH = Path(DATA_DIR) / f"{VOICE_MODEL}.json"

try:
    if not (MODEL_PATH.exists() and JSON_PATH.exists()) :
        logging.info(f"Downloading voice model {VOICE_MODEL}...")
        MODEL_PATH.parent.mkdir(exist_ok=True)
        import requests
        url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{VOICE_LANG_1}/{VOICE_LANG_2}/{VOICE_NAME}/{VOICE_STYLE}/{VOICE_MODEL}"
        response = requests.get(url)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{VOICE_LANG_1}/{VOICE_LANG_2}/{VOICE_NAME}/{VOICE_STYLE}/{VOICE_MODEL}.json"
        response = requests.get(url)
        response.raise_for_status()
        with open(JSON_PATH, "wb") as f:
            f.write(response.content)
except Exception as e:
    logging.error(f"Failed to initialize voice model: {e}")
    raise

def tts_worker(output_path, voice_path, text):
    try:
        audio_buffer = tts(voice_path, text)
        with open(output_path, 'wb') as f:
            f.write(audio_buffer.getvalue())
    except Exception as e:
        with open(output_path, 'w') as f:
            f.write(f"ERROR: {str(e)}")
    
def run_tts(voice_path, text):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "output.mp3")
        p = Process(target=tts_worker, args=(temp_file, voice_path, text))
        p.start()
        p.join()
        with open(temp_file, 'rb') as f:
            content = f.read()
    if content.startswith(b"ERROR:"):
        raise RuntimeError(content.decode('utf-8')[7:])
    return io.BytesIO(content)

def create_audio_buffer(raw_audio, sample_rate=22050):
    buffer = io.BytesIO()
    audio_segment = AudioSegment(
        data=raw_audio, 
        sample_width=2, 
        frame_rate=sample_rate, 
        channels=1
    )
    audio_segment.export(buffer, format="mp3")
    buffer.seek(0)
    return buffer

def tts(voice_path, text):
    model = PiperVoice.load(voice_path, use_cuda=cuda)
    raw_audio = io.BytesIO()
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wav_file:
            model.synthesize(text, wav_file)
        raw_audio = wav_io.getvalue()
    return create_audio_buffer(raw_audio, model.config.sample_rate)

@app.get("/synthesize")
async def synthesize(text: str):
    """Convert text to speech and return WAV audio"""
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter is required")
    
    if len(text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long (max 1000 characters)")
    
    try:
        audio_buffer = run_tts(MODEL_PATH, text)
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mp3",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3",
                "X-Voice-Model": VOICE_MODEL
            }
        )
    except Exception as e:
        logging.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail="Speech synthesis failed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "voice_model": VOICE_MODEL,
        "service": "Piper TTS"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
