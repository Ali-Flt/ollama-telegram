import requests
from requests.exceptions import RequestException
from dotenv import load_dotenv
import os
import logging
import re

load_dotenv()
whisper_url = os.getenv("WHISPER_SERVICE_URL")
tts_url = os.getenv("TTS_SERVICE_URL")
tts_voice = os.getenv("TTS_VOICE", "af_bella")
log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = logging.getLevelName(log_level_str)
logging.basicConfig(level=log_level)

def clean_text_for_tts(text):
    # Remove content inside <think>...</think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove special characters and emojis (excluding basic punctuation)
    text = re.sub(r'[*_~`^{}\[\]<>|\\/#@+=%&$€£¥¢§¶•]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def speech_to_text(audio_path: str) -> dict:
    try:
        with open(audio_path, 'rb') as f:
            response = requests.post(
                f"{whisper_url}/transcribe",
                files={'file': (os.path.basename(audio_path), f)}
            )
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        logging.error(f"Error communicating with whisper service: {e}")
        return None

def text_to_speech(text: str, output_file: str = "output.mp3"):
    text = clean_text_for_tts(text)
    logging.debug(f"Cleaned prompt: {text}")
    if "piper" in tts_url:
        response = requests.get(
            f"{tts_url}/synthesize",
            params={"text": text},
            stream=True
        )
        response.raise_for_status()
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    elif "kokoro" in tts_url:
        response = requests.get(
            f"{tts_url}/synthesize",
            params={"text": text},
            stream=True
        )
        response.raise_for_status()
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
