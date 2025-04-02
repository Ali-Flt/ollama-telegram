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

device = os.getenv("DEVICE", "cuda")
use_cuda = True if device == "cuda" else False
voice_model = os.getenv("TTS_VOICE", "en_US-lessac-medium")
voice_lang_1 = voice_model.split('_')[0]
voice_lang_2 = voice_model.split('-')[0]
voice_name = voice_model.split('-')[1]
voice_style = voice_model.split('-')[-1]
voice_model += ".onnx"
data_dir = os.getenv("PIPER_DATA_DIR", "/app/voices")
model_path = Path(data_dir) / voice_model
json_path = Path(data_dir) / f"{voice_model}.json"

try:
    if not (model_path.exists() and json_path.exists()) :
        logging.info(f"Downloading voice model {voice_model}...")
        model_path.parent.mkdir(exist_ok=True)
        import requests
        url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{voice_lang_1}/{voice_lang_2}/{voice_name}/{voice_style}/{voice_model}"
        response = requests.get(url)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            f.write(response.content)
        url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{voice_lang_1}/{voice_lang_2}/{voice_name}/{voice_style}/{voice_model}.json"
        response = requests.get(url)
        response.raise_for_status()
        with open(json_path, "wb") as f:
            f.write(response.content)
except Exception as e:
    logging.error(f"Failed to initialize voice model: {e}")
    raise

def tts_worker(output_path, text):
    try:
        audio_buffer = tts(text)
        with open(output_path, 'wb') as f:
            f.write(audio_buffer.getvalue())
    except Exception as e:
        with open(output_path, 'w') as f:
            f.write(f"ERROR: {str(e)}")
    
def run_tts(text):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "output.mp3")
        p = Process(target=tts_worker, args=(temp_file, text))
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

def tts(text):
    model = PiperVoice.load(model_path, use_cuda=use_cuda)
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
        audio_buffer = run_tts(text)
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mp3",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3",
                "X-Voice-Model": voice_model
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
        "voice_model": voice_model,
        "service": "Piper TTS"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
