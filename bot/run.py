from aiogram import Bot, Dispatcher, types, BaseMiddleware
from aiogram.enums import ParseMode
from aiogram.filters.command import Command, CommandStart
from aiogram.types import Message
from aiogram.types import FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
import telegramify_markdown
import telegramify_markdown.customize as customize
customize.strict_markdown = False
import tempfile
from pathlib import Path
from func.interactions import *
import asyncio
import traceback
import io
import base64
import sqlite3
import os
import requests
from requests.exceptions import RequestException
import json
from abc import ABC
from typing import Callable, Dict, Any, Awaitable
        
whisper_url = os.getenv("WHISPER_SERVICE_URL")
tts_url = os.getenv("TTS_SERVICE_URL")
bot = Bot(token=token)
dp = Dispatcher()
start_kb = InlineKeyboardBuilder()
settings_kb = InlineKeyboardBuilder()

start_kb.row(
    types.InlineKeyboardButton(text="ℹ️ About", callback_data="about"),
    types.InlineKeyboardButton(text="⚙️ Settings", callback_data="settings"),
    types.InlineKeyboardButton(text="📝 Register", callback_data="register"),
)
settings_kb.row(
    types.InlineKeyboardButton(text="🔄 Switch LLM", callback_data="switchllm"),
    types.InlineKeyboardButton(text="🗑️ Delete LLM", callback_data="delete_model"),
)
settings_kb.row(
    types.InlineKeyboardButton(text="📋 Select System Prompt", callback_data="select_prompt"),
    types.InlineKeyboardButton(text="🗑️ Delete System Prompt", callback_data="delete_prompt"), 
)
settings_kb.row(
    types.InlineKeyboardButton(text="📋 List Users and remove User", callback_data="list_users"),
)

commands = [
    types.BotCommand(command="start", description="Start"),
    types.BotCommand(command="reset", description="Reset Chat"),
    types.BotCommand(command="history", description="Look through messages"),
    types.BotCommand(command="pullmodel", description="Pull a model from Ollama"),
    types.BotCommand(command="addsystemprompt", description="Add a system prompt"),
]

modelname = os.getenv("INITMODEL")
mention = None
CHAT_TYPE_GROUP = "group"
CHAT_TYPE_SUPERGROUP = "supergroup"


class AlbumMiddleware(BaseMiddleware, ABC):
    """This middleware is for capturing media groups."""

    album_data: dict = {}

    def __init__(self, latency: int | float = 0.01):
        """
        You can provide custom latency to make sure
        albums are handled properly in highload.
        """
        self.latency = latency
        super().__init__()

    async def __call__(
        self,
        handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: Dict[str, Any],
    ) -> Any:
        if not event.media_group_id:
            return await handler(event, data)

        try:
            self.album_data[event.media_group_id].append(event)
            return  # Tell aiogram to cancel handler for this group element
        except KeyError:
            self.album_data[event.media_group_id] = [event]
            await asyncio.sleep(self.latency)

            event.model_config["is_last"] = True
            data["album"] = self.album_data[event.media_group_id]

            result = await handler(event, data)

            if event.media_group_id and event.model_config.get("is_last"):
                del self.album_data[event.media_group_id]

            return result

def init_db():
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, name TEXT, selected_prompt_id INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  role TEXT,
                  content TEXT,
                  images TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS system_prompts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  prompt TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()
    
def get_user_messages(user_id):
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("SELECT role, content, images FROM chats WHERE user_id = ? ORDER BY id", (user_id,))
    messages = [{"role": role, "content": content, "images": json.loads(images) if images else []} for (role, content, images) in c.fetchall()]
    conn.close()
    return messages

def delete_user_chats(user_id):
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("DELETE FROM chats WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    
def register_user(user_id, user_name):
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?)", (user_id, user_name, -1))
    conn.commit()
    conn.close()

def save_chat_message(user_id, role, content, images=[]):
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_id, role, content, images) VALUES (?, ?, ?, ?)",
              (user_id, role, content, json.dumps(images)))
    conn.commit()
    conn.close()

def transcribe_audio(audio_path: str) -> dict:
    try:
        with open(audio_path, 'rb') as f:
            response = requests.post(
                f"{whisper_url}/transcribe",
                files={'file': (os.path.basename(audio_path), f)}
            )
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        print(f"Error communicating with whisper service: {e}")
        return None

def text_to_speech(text: str, output_file: str = "output.wav"):
    response = requests.get(
        f"{tts_url}/synthesize",
        params={"text": text},
        stream=True
    )
    response.raise_for_status()
    with open(output_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

@dp.callback_query(lambda query: query.data == "register")
@perms_allowed
async def register_callback_handler(query: types.CallbackQuery):
    user_id = query.from_user.id
    user_name = query.from_user.full_name
    register_user(user_id, user_name)
    await query.answer("You have been registered successfully!")

async def get_bot_info():
    global mention
    if mention is None:
        get = await bot.get_me()
        mention = f"@{get.username}"
    return mention

@dp.message(CommandStart())
@perms_allowed
async def command_start_handler(message: Message) -> None:
    start_message = f"Welcome, <b>{message.from_user.full_name}</b>!"
    await message.answer(
        start_message,
        parse_mode=ParseMode.HTML,
        reply_markup=start_kb.as_markup(),
        disable_web_page_preview=True,
    )

@dp.message(Command("reset"))
@perms_allowed
async def command_reset_handler(message: Message) -> None:
    user_id = message.from_user.id
    delete_user_chats(user_id)
    logging.info(f"Chat has been reset for {message.from_user.first_name}")
    await bot.send_message(
        chat_id=message.chat.id,
        text="Chat has been reset",
    )

@dp.message(Command("history"))
@perms_allowed
async def command_get_context_handler(message: Message) -> None:
    user_id = message.from_user.id
    messages = get_user_messages(user_id)
    if messages:
        context = ""
        for msg in messages:
            context += f"*{msg['role'].capitalize()}*: {msg['content']}\n"
        context = telegramify_markdown.markdownify(context)
        await bot.send_message(
            chat_id=message.chat.id,
            text=context,
            parse_mode="MarkdownV2"
        )
    else:
        await bot.send_message(
            chat_id=message.chat.id,
            text="No chat history available for this user",
        )

@dp.message(Command("addsystemprompt"))
@perms_allowed
async def add_system_prompt_handler(message: Message):
    prompt_text = message.text.split(maxsplit=1)[1] if len(message.text.split()) > 1 else None  # Get the prompt text from the command arguments
    if prompt_text:
        add_system_prompt(message.from_user.id, prompt_text)
        await message.answer("Private prompt added successfully.")
    else:
        await message.answer("Please provide a prompt text to add.")

@dp.message(Command("pullmodel"))
@perms_admins
async def pull_model_handler(message: Message) -> None:
    model_name = message.text.split(maxsplit=1)[1] if len(message.text.split()) > 1 else None  # Get the model name from the command arguments
    logging.info(f"Downloading {model_name}")
    if model_name:
        response = await manage_model("pull", model_name)
        if response.status == 200:
            await message.answer(f"Model '{model_name}' is being pulled.")
        else:
            await message.answer(f"Failed to pull model '{model_name}': {response.reason}")
    else:
        await message.answer("Please provide a model name to pull.")

@dp.callback_query(lambda query: query.data == "settings")
@perms_allowed
async def settings_callback_handler(query: types.CallbackQuery):
    await bot.send_message(
        chat_id=query.message.chat.id,
        text=f"Choose the right option.",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
        reply_markup=settings_kb.as_markup()
    )

@dp.callback_query(lambda query: query.data == "switchllm")
@perms_allowed
async def switchllm_callback_handler(query: types.CallbackQuery):
    models = await model_list()
    switchllm_builder = InlineKeyboardBuilder()
    for model in models:
        modelname = model["name"]
        modelfamilies = ""
        if model["details"]["families"]:
            modelicon = {"llama": "🦙", "clip": "📷"}
            try:
                modelfamilies = "".join(
                    [modelicon[family] for family in model["details"]["families"]]
                )
            except KeyError as e:
                modelfamilies = f"✨"
        switchllm_builder.row(
            types.InlineKeyboardButton(
                text=f"{modelname} {modelfamilies}", callback_data=f"model_{modelname}"
            )
        )
    await query.message.edit_text(
        f"{len(models)} models available.\n🦙 = Regular\n🦙📷 = Multimodal", reply_markup=switchllm_builder.as_markup(),
    )

@dp.callback_query(lambda query: query.data.startswith("model_"))
@perms_admins
async def model_callback_handler(query: types.CallbackQuery):
    global modelname
    modelname = query.data.split("model_")[1]
    await query.answer(f"Chosen model: {modelname}")

@dp.callback_query(lambda query: query.data == "about")
@perms_admins
async def about_callback_handler(query: types.CallbackQuery):
    dotenv_model = os.getenv("INITMODEL")
    global modelname
    await bot.send_message(
        chat_id=query.message.chat.id,
        text=f"<b>Your LLMs</b>\nCurrently using: <code>{modelname}</code>\nDefault in .env: <code>{dotenv_model}</code>\nThis project is under <a href='https://github.com/ruecat/ollama-telegram/blob/main/LICENSE'>MIT License.</a>\n<a href='https://github.com/ruecat/ollama-telegram'>Source Code</a>",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )

@dp.callback_query(lambda query: query.data == "list_users")
@perms_admins
async def list_users_callback_handler(query: types.CallbackQuery):
    users = get_all_users_from_db()
    user_kb = InlineKeyboardBuilder()
    for user_id, user_name in users:
        user_kb.row(types.InlineKeyboardButton(text=f"{user_name} ({user_id})", callback_data=f"remove_{user_id}"))
    user_kb.row(types.InlineKeyboardButton(text="Cancel", callback_data="cancel_remove"))
    await query.message.answer("Select a user to remove:", reply_markup=user_kb.as_markup())

@dp.callback_query(lambda query: query.data.startswith("remove_"))
@perms_admins
async def remove_user_from_list_handler(query: types.CallbackQuery):
    user_id = int(query.data.split("_")[1])
    if remove_user_from_db(user_id):
        await query.answer(f"User {user_id} has been removed.")
        await query.message.edit_text(f"User {user_id} has been removed.")
    else:
        await query.answer(f"User {user_id} not found.")

@dp.callback_query(lambda query: query.data == "cancel_remove")
@perms_admins
async def cancel_remove_handler(query: types.CallbackQuery):
    await query.message.edit_text("User removal cancelled.")

@dp.callback_query(lambda query: query.data == "select_prompt")
@perms_allowed
async def select_prompt_callback_handler(query: types.CallbackQuery):
    prompts = get_system_prompts(user_id=query.from_user.id)
    prompt_kb = InlineKeyboardBuilder()
    for prompt in prompts:
        prompt_id, _, prompt_text, _ = prompt
        prompt_kb.row(
            types.InlineKeyboardButton(
                text=prompt_text, callback_data=f"prompt_{prompt_id}"
            )
        )
    await query.message.edit_text(
        f"{len(prompts)} system prompts available.", reply_markup=prompt_kb.as_markup()
    )

@dp.callback_query(lambda query: query.data.startswith("prompt_"))
@perms_allowed
async def prompt_callback_handler(query: types.CallbackQuery):
    selected_prompt_id = int(query.data.split("prompt_")[1])
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("UPDATE users SET selected_prompt_id = ? WHERE id = ?", (selected_prompt_id, query.from_user.id))
    conn.commit()
    conn.close()
    await query.answer(f"Prompt selected.")

@dp.callback_query(lambda query: query.data == "delete_prompt")
@perms_allowed
async def delete_prompt_callback_handler(query: types.CallbackQuery):
    prompts = get_system_prompts(user_id=query.from_user.id)
    delete_prompt_kb = InlineKeyboardBuilder()
    for prompt in prompts:
        prompt_id, _, prompt_text, _ = prompt
        delete_prompt_kb.row(
            types.InlineKeyboardButton(
                text=prompt_text, callback_data=f"delete_prompt_{prompt_id}"
            )
        )
    await query.message.edit_text(
        f"{len(prompts)} system prompts available for deletion.", reply_markup=delete_prompt_kb.as_markup()
    )

@dp.callback_query(lambda query: query.data.startswith("delete_prompt_"))
@perms_allowed
async def delete_prompt_confirm_handler(query: types.CallbackQuery):
    prompt_id = int(query.data.split("delete_prompt_")[1])
    delete_ystem_prompt(prompt_id)
    await query.answer(f"Prompt deleted.")

@dp.callback_query(lambda query: query.data == "delete_model")
@perms_admins
async def delete_model_callback_handler(query: types.CallbackQuery):
    models = await model_list()
    delete_model_kb = InlineKeyboardBuilder()
    for model in models:
        modelname = model["name"]
        delete_model_kb.row(
            types.InlineKeyboardButton(
                text=modelname, callback_data=f"delete_model_{modelname}"
            )
        )
    await query.message.edit_text(
        f"{len(models)} models available for deletion.", reply_markup=delete_model_kb.as_markup()
    )

@dp.callback_query(lambda query: query.data.startswith("delete_model_"))
@perms_admins
async def delete_model_confirm_handler(query: types.CallbackQuery):
    modelname = query.data.split("delete_model_")[1]
    response = await manage_model("delete", modelname)
    if response.status == 200:
        await query.answer(f"Deleted model: {modelname}")
    else:
        await query.answer(f"Failed to delete model: {modelname}")

@dp.message()
@perms_allowed
async def handle_message(message: types.Message, album: list[Message] = None):
    await get_bot_info()
    photos = []
    if album:
        caption = None
        for file in album:
            caption = caption or file.caption
            if file.photo:
                photos.append(file.photo[-1])
        if message.caption is None:
            message = message.copy(update={"caption": caption})   
    elif message.photo:
        photos.append(message.photo[-1])
    if message.chat.type == "private":
        await ollama_request(message, photos=photos)
        return
    if await is_mentioned_in_group_or_supergroup(message):
        thread = await collect_message_thread(message)
        prompt = format_thread_for_prompt(thread)
        await ollama_request(message, prompt, photos)

async def is_mentioned_in_group_or_supergroup(message: types.Message):
    if message.chat.type not in ["group", "supergroup"]:
        return False
    
    is_mentioned = (
        (message.text and message.text.startswith(mention)) or
        (message.caption and message.caption.startswith(mention))
    )
    
    is_reply_to_bot = (
        message.reply_to_message and 
        message.reply_to_message.from_user.id == bot.id
    )
    
    return is_mentioned or is_reply_to_bot

async def collect_message_thread(message: types.Message, thread=None):
    if thread is None:
        thread = []
    
    thread.insert(0, message)
    
    if message.reply_to_message:
        await collect_message_thread(message.reply_to_message, thread)
    
    return thread

def format_thread_for_prompt(thread):
    prompt = "Conversation thread:\n\n"
    for msg in thread:
        sender = "User" if msg.from_user.id != bot.id else "Bot"
        content = msg.text or msg.caption or "[No text content]"
        prompt += f"{sender}: {content}\n\n"
    
    prompt += "History:"
    return prompt

async def process_images(photos):
    images_base64 = []
    if photos:
        for photo in photos:
            image_buffer = io.BytesIO()
            await bot.download(photo, destination=image_buffer)
            image_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
            images_base64.append(image_base64)
    return images_base64

async def handle_response(message, response_data, full_response):
    if full_response == "":
        return
    if response_data.get("done"):
        await send_response(message, full_response, response_data)
        logging.info(
            f"[Response]: '{full_response}' for {message.from_user.first_name} {message.from_user.last_name}"
        )
        return True
    return False

async def send_response(message, full_response_stripped, response_data):
    if message.content_type == "voice":
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "response.ogg")
            text_to_speech(full_response_stripped, file_path)
            voice_msg = FSInputFile(file_path)
            await bot.send_voice(
                chat_id=message.chat.id,
                voice=voice_msg,
            )
    else:
        text = f"{full_response_stripped}\n\n⚙️ {modelname}\nGenerated in {response_data.get('total_duration') / 1e9:.2f}s."
        # Escape Markdown special characters to prevent formatting issues
        text = telegramify_markdown.markdownify(text)
        # A negative message.chat.id is a group message
        if message.chat.id < 0 or message.chat.id == message.from_user.id:
            await bot.send_message(
                chat_id=message.chat.id,
                text=text,
                parse_mode="MarkdownV2"
            )
        else:
            await bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=message.message_id,
                text=text,
                parse_mode="MarkdownV2"
            )

async def ollama_request(message: types.Message, prompt: str = None, photos: list = []):
    try:
        full_response = ""
        await bot.send_chat_action(message.chat.id, "typing")
        images_base64 = await process_images(photos)
        
        # Determine the prompt
        if prompt is None:
            prompt = message.text or message.caption

        # Retrieve and prepare system prompt if selected
        system_prompt = None
        conn = sqlite3.connect('data/users.db')
        c = conn.cursor()
        c.execute("SELECT selected_prompt_id FROM users WHERE id = ?", (message.from_user.id,))
        selected_prompt_id = c.fetchone()
        selected_prompt_id = selected_prompt_id[0] if selected_prompt_id is not None else -1
        logging.info(f"Selected prompt ID: {selected_prompt_id}")
        conn.close()
        system_prompts = get_system_prompts(user_id=message.from_user.id)
        messages = []
        if system_prompts:
            for sp in system_prompts:
                if sp[0] == selected_prompt_id:
                    system_prompt = sp[2]
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                        })
                    break
            if system_prompt is None:
                logging.warning(f"Selected prompt ID {selected_prompt_id} not found for user {message.from_user.id}")
        if message.content_type == "voice":
            voice = message.voice
            with tempfile.TemporaryDirectory() as temp_dir:
                file = await bot.get_file(voice.file_id)
                file_path = Path(temp_dir) / f"{voice.file_unique_id}.ogg"                
                await bot.download_file(file.file_path, str(file_path))
                result = transcribe_audio(file_path)
            prompt = result.get('transcription', 'No transcription available')
            logging.info(f"Voice prompt: {prompt}")
        save_chat_message(message.from_user.id, "user", prompt, images_base64)
        messages.extend(get_user_messages(message.from_user.id))
        logging.info(
            f"[OllamaAPI]: Processing '{prompt}' for {message.from_user.first_name} {message.from_user.last_name}"
        )
        payload = {
            "model": modelname,
            "messages": messages,
            "stream": True,
        }
        # Generate response
        async for response_data in generate(payload, modelname, prompt):
            msg = response_data.get("message")
            if msg is None:
                continue
            chunk = msg.get("content", "")
            full_response += chunk

            if any([c in chunk for c in ".\n!?"]) or response_data.get("done"):
                full_response_stripped = full_response.strip()
                if await handle_response(message, response_data, full_response_stripped):
                    save_chat_message(message.from_user.id, "assistant", full_response_stripped)
                    break

    except Exception as e:
        print(f"-----\n[OllamaAPI-ERR] CAUGHT FAULT!\n{traceback.format_exc()}\n-----")
        await bot.send_message(
            chat_id=message.chat.id,
            text=f"Something went wrong: {str(e)}",
            parse_mode=ParseMode.HTML,
        )
        
async def main():
    init_db()
    allowed_ids = load_allowed_ids_from_db()
    print(f"allowed_ids: {allowed_ids}")
    middleware = AlbumMiddleware()
    await bot.set_my_commands(commands)
    dp.message.outer_middleware(middleware)
    await dp.start_polling(bot, skip_update=True)

if __name__ == "__main__":
    asyncio.run(main())
