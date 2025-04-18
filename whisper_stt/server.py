from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from typing import Annotated
import logging
from multiprocessing import Process, Queue
from faster_whisper import WhisperModel
import uuid
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Whisper STT Server", version="1.0")

log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = logging.getLevelName(log_level_str)
logging.basicConfig(level=log_level)

# Load configuration from environment
model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
device = os.getenv("DEVICE", "cpu")
compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

def transcribe(temp_path: str, model_size: str, device: str, compute_type: str):
    """Function to run in separate process for transcription"""
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        segments, info = model.transcribe(temp_path)
        transcription = " ".join(segment.text for segment in segments)
        return {
            "language": info.language,
            "duration": info.duration,
            "transcription": transcription
        }
    except Exception as e:
        raise e

def transcribe_worker(queue, temp_path: str, model_size: str, device: str, compute_type: str):
    transcript = transcribe(temp_path, model_size, device, compute_type)
    queue.put(transcript)
    
def run_transcribe(temp_path: str, model_size: str, device: str, compute_type: str):
    queue = Queue()
    p = Process(target=transcribe_worker, args=(queue, temp_path, model_size, device, compute_type))
    p.start()
    p.join()
    return queue.get()


@app.post("/transcribe")
async def transcribe_audio(file: Annotated[UploadFile, File(description="Audio file to transcribe")]):
    try:
        os.makedirs("/tmp/whisper", exist_ok=True)
        temp_path = f"/tmp/whisper/{uuid.uuid4()}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        logging.info(f"Transcribing file: {file.filename}")
        result = run_transcribe(temp_path, model_size, device, compute_type)
        os.remove(temp_path)
        return result

    except Exception as e:
        logging.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Whisper STT"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
