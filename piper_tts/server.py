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
VOICE_MODEL = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
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

def convert_wav_to_ogg(input_wav_path, output_ogg_path=None):
    if not os.path.exists(input_wav_path):
        raise FileNotFoundError(f"Input file not found: {input_wav_path}")
    if output_ogg_path is None:
        base_path = os.path.splitext(input_wav_path)[0]
        output_ogg_path = f"{base_path}.ogg"
    try:
        audio = AudioSegment.from_wav(input_wav_path)
        audio.export(output_ogg_path, format='ogg', codec="opus", bitrate="48k", parameters=['-strict', '-2'])
        return output_ogg_path
    except Exception as e:
        raise Exception(f"Failed to convert WAV to OGG: {str(e)}")
    
    
def run_tts(voice_path, text):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav = os.path.join(temp_dir, "output.wav")
        temp_ogg = os.path.join(temp_dir, "output.ogg")
        p = Process(target=tts_worker, args=(temp_wav, voice_path, text))
        p.start()
        p.join()
        convert_wav_to_ogg(temp_wav, temp_ogg)
        with open(temp_ogg, 'rb') as f:
            content = f.read()
        if content.startswith(b"ERROR:"):
            raise RuntimeError(content.decode('utf-8')[7:])
        return io.BytesIO(content)


def create_wav_buffer(audio_data, sample_rate=22050):
    """Create a proper WAV file buffer with correct headers"""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    buffer.seek(0)
    return buffer

def tts(voice_path, text):
    model = PiperVoice.load(voice_path, use_cuda=cuda)
    raw_audio = io.BytesIO()
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wav_file:
            model.synthesize(text, wav_file)
        raw_audio = wav_io.getvalue()
    return create_wav_buffer(raw_audio, model.config.sample_rate)


@app.get("/synthesize")
async def synthesize(text: str):
    """Convert text to speech and return WAV audio"""
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter is required")
    
    if len(text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long (max 1000 characters)")
    
    try:
        ogg_buffer = run_tts(MODEL_PATH, text)
        return StreamingResponse(
            ogg_buffer,
            media_type="audio/ogg",
            headers={
                "Content-Disposition": "attachment; filename=speech.ogg",
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
