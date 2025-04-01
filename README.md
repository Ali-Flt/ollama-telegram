<div align="center">
  <br>
  <a href="">
    <img src=".github/ollama-telegram-readme.png" width="200" height="200">
  </a>
  <h1>🦙 Ollama Telegram Bot</h1>
  <p>
    <b>Chat with your LLM, using Telegram bot!</b><br>
    <b>Feel free to contribute!</b><br>
  </p>
  <br>
  <p align="center">
    <img src="https://img.shields.io/docker/pulls/ruecat/ollama-telegram?style=for-the-badge">
  </p>
  <br>
</div>

## Features
Here's features that you get out of the box:

- [x] Fully dockerized bot
- [x] Response streaming without ratelimit with **SentenceBySentence** method
- [x] Mention [@] bot in group to receive answer

## Roadmap
- [x] Docker config & automated tags by [StanleyOneG](https://github.com/StanleyOneG), [ShrirajHegde](https://github.com/ShrirajHegde)
- [x] History and `/reset` by [ShrirajHegde](https://github.com/ShrirajHegde)
- [ ] Add more API-related functions [System Prompt Editor, Ollama Version fetcher, etc.]
- [ ] Redis DB integration
- [ ] Update bot UI
## This fork's progress
- [x] Add voice message support via OpenAI Whisper
- [x] Add speech to speech support via OpenAI Whisper and Piper TTS
- [x] Fix permission issues
- [x] Add option to send the bot an album of photos
- [x] Filesystem cleaning
- [x] User management improvement
## Prerequisites
- [Telegram-Bot Token](https://core.telegram.org/bots#6-botfather)

## Installation (Non-Docker)
+ Install latest [Python](https://python.org/downloads)
+ Clone Repository
    ```
    git clone https://github.com/ruecat/ollama-telegram
    ```
+ Install requirements from requirements.txt
    ```
    pip install -r requirements.txt
    ```
+ Enter all values in .env.example

+ Rename .env.example -> .env

+ Launch bot

    ```
    python3 run.py
    ```
## Installation (Docker Image)
The official image is available at dockerhub: [ruecat/ollama-telegram](https://hub.docker.com/r/ruecat/ollama-telegram)

+ Download [.env.example](https://github.com/ruecat/ollama-telegram/blob/main/.env.example) file, rename it to .env and populate the variables.
+ Create `docker-compose.yml` (optionally: uncomment GPU part of the file to enable Nvidia GPU)

    ```yml
    version: '3.8'
    services:
      ollama-telegram:
        image: ruecat/ollama-telegram
        container_name: ollama-telegram
        restart: on-failure
        env_file:
          - ./.env
      
      ollama-server:
        image: ollama/ollama:latest
        container_name: ollama-server
        volumes:
          - ./ollama:/root/.ollama
        
        # Uncomment to enable NVIDIA GPU
        # Otherwise runs on CPU only:
    
        # deploy:
        #   resources:
        #     reservations:
        #       devices:
        #         - driver: nvidia
        #           count: all
        #           capabilities: [gpu]
    
        restart: always
        ports:
          - '11434:11434'
    ```

+ Start the containers

    ```sh
    docker compose up -d
    ```


## Installation (Build your own Docker image)
+ Clone Repository
    ```
    git clone https://github.com/ruecat/ollama-telegram
    ```

+ Enter all values in .env.example

+ Rename .env.example -> .env

+ Run ONE of the following docker compose commands to start:
    1. To run ollama in docker container (optionally: uncomment GPU part of docker-compose.yml file to enable Nvidia GPU)
        ```
        docker compose up --build -d
        ```

    2. To run ollama from locally installed instance (mainly for **MacOS**, since docker image doesn't support Apple GPU acceleration yet):
        ```
        docker compose up --build -d ollama-tg
        ```


## Credits
+ [Ollama](https://github.com/jmorganca/ollama)

## Libraries used
+ [Aiogram 3.x](https://github.com/aiogram/aiogram)
