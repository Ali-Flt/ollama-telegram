from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import logging
import os
import multiprocessing
from multiprocessing import Process
import tempfile
from pydub import AudioSegment
from dotenv import load_dotenv
from kokoro import KPipeline
import io
import soundfile as sf

device = os.getenv("DEVICE", "cuda")
load_dotenv()
app = FastAPI(title="Kokoro TTS Server", version="1.0")
voice_model = os.getenv("TTS_VOICE", "af_heart")

log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = logging.getLevelName(log_level_str)
logging.basicConfig(level=log_level)


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

def create_audio_buffer(audio_file):
    buffer = io.BytesIO()
    audio_segment = AudioSegment.from_file(audio_file)
    audio_segment.export(buffer, format="mp3")
    buffer.seek(0)
    return buffer

def tts(text):
    pipeline = KPipeline(lang_code='a', device=device)
    generator = pipeline(text, voice=voice_model)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "output.wav")
        for _, (_, _, audio) in enumerate(generator):
            sf.write(temp_file, audio, 24000)
        return create_audio_buffer(temp_file)

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
                "X-Voice-Model": 'KokoroV1'
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
        "service": "Kokoro TTS"
    }

if __name__ == "__main__":
    import uvicorn
    multiprocessing.set_start_method('spawn', force=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
