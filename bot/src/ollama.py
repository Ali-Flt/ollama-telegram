import aiohttp
from aiohttp import ClientTimeout
import logging
import os
import json
from dotenv import load_dotenv

load_dotenv()
log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = logging.getLevelName(log_level_str)
logging.basicConfig(level=log_level)
ollama_url = os.getenv("OLLAMA_URL")
timeout = os.getenv("TIMEOUT", "3000")
keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "0")

async def manage_model(action: str, model_name: str):
    async with aiohttp.ClientSession() as session:
        url = f"{ollama_url}/api/{action}"
        
        if action == "pull":
            # Use the exact payload structure from the curl example
            data = json.dumps({"name": model_name})
            headers = {
                'Content-Type': 'application/json'
            }
            logging.info(f"Pulling model: {model_name}")
            logging.info(f"Request URL: {url}")
            logging.info(f"Request Payload: {data}")
            
            async with session.post(url, data=data, headers=headers) as response:
                logging.info(f"Pull model response status: {response.status}")
                response_text = await response.text()
                logging.info(f"Pull model response text: {response_text}")
                return response
        elif action == "delete":
            data = json.dumps({"name": model_name})
            headers = {
                'Content-Type': 'application/json'
            }
            async with session.delete(url, data=data, headers=headers) as response:
                return response
        else:
            logging.error(f"Unsupported model management action: {action}")
            return None
        
async def model_list():
    async with aiohttp.ClientSession() as session:
        url = f"{ollama_url}/api/tags"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data["models"]
            else:
                return []
                
async def generate(payload: dict, modelname: str, prompt: str):
    client_timeout = ClientTimeout(total=int(timeout))
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        url = f"{ollama_url}/api/chat"

        # Prepare the payload according to Ollama API specification
        ollama_payload = {
            "model": modelname,
            "messages": payload.get("messages", []),
            "stream": payload.get("stream", True),
            "keep_alive": keep_alive
        }

        try:
            logging.info(f"Sending request to Ollama API: {url}")
            logging.info(f"Payload: {json.dumps(ollama_payload, indent=2)}")

            async with session.post(url, json=ollama_payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logging.error(f"API Error: {response.status} - {error_text}")
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"API Error: {error_text}"
                    )

                buffer = b""
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError as e:
                                logging.error(f"JSON Decode Error: {e}")
                                logging.error(f"Problematic line: {line}")

        except aiohttp.ClientError as e:
            logging.error(f"Client Error during request: {e}")
            raise
