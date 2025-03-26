from fastapi import FastAPI, File, UploadFile, HTTPException
from faster_whisper import WhisperModel
import os
from typing import Annotated
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from environment
model_size = os.getenv("MODEL_SIZE", "small")
device = os.getenv("DEVICE", "cpu")
compute_type = os.getenv("COMPUTE_TYPE", "int8")

logger.info(f"Loading Whisper model (size={model_size}, device={device})")
model = WhisperModel(
    model_size,
    device=device,
    compute_type=compute_type
)

@app.post("/transcribe")
async def transcribe_audio(file: Annotated[UploadFile, File(description="Audio file to transcribe")]):
    try:
        # Create temp directory if it doesn't exist
        os.makedirs("/tmp/whisper", exist_ok=True)
        temp_path = f"/tmp/whisper/{file.filename}"
        
        # Save the uploaded file
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        logger.info(f"Transcribing file: {file.filename}")
        segments, info = model.transcribe(temp_path)
        transcription = " ".join(segment.text for segment in segments)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "language": info.language,
            "duration": info.duration,
            "transcription": transcription
        }
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
