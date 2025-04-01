<div align="center">
  <br>
    <img src=".github/ollama-telegram-readme.png" width="200" height="200">
  <h1>🦙 Ollama Telegram Bot</h1>
  <p>
    <b>Chat with your LLM, using Telegram bot!</b><br>
    <b>This repository is based on <a href="https://github.com/ruecat/ollama-telegram">this project</a> with many fixes and added features!</b><br>    <b>Feel free to contribute!</b><br>
  </p>
</div>

## Progress since fork
- [x] Add voice message support via OpenAI Whisper
- [x] Add speech to speech support via OpenAI Whisper and Piper TTS
- [x] Fix permission issues
- [x] Add option to send the bot an album of photos
- [x] Filesystem cleaning
- [x] User management improvement
## Prerequisites
- [Telegram-Bot Token](https://core.telegram.org/bots#6-botfather)
## Installation (Build your own Docker image)
+ Clone Repository
    ```
    git clone https://github.com/Ali-Flt/ollama-telegram
    ```
+ Enter all values in .env.example
+ Rename .env.example -> .env
+ To run ollama in docker container
  ```
  docker compose up --build -d
  ```
## Credits
+ [Ollama](https://github.com/jmorganca/ollama)
+ [Original ollama-telegram](https://github.com/ruecat/ollama-telegram)
