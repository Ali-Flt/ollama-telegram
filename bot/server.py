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
import asyncio
import traceback
import io
import base64
import os
from abc import ABC
from typing import Callable, Dict, Any, Awaitable
from typing import Union
from functools import wraps
from dotenv import load_dotenv
import logging

from src.db import init_db, register_user, delete_user_chats, get_user_messages, add_system_prompt,\
    get_all_users_from_db, remove_user_from_db, get_system_prompts, update_selected_system_prompt,\
        delete_system_prompt, save_chat_message, get_selected_system_prompt
from src.ollama import model_list, manage_model, generate
from src.voice import text_to_speech, speech_to_text

load_dotenv()
token = os.getenv("TOKEN")
allowed_ids = list(map(int, os.getenv("USER_IDS", "").split(",")))
admin_ids = list(map(int, os.getenv("ADMIN_IDS", "").split(",")))
allow_all_users_in_groups = bool(int(os.getenv("ALLOW_ALL_USERS_IN_GROUPS", "0")))
log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = logging.getLevelName(log_level_str)
logging.basicConfig(level=log_level)

bot = Bot(token=token)
dp = Dispatcher()
start_kb = InlineKeyboardBuilder()
settings_kb = InlineKeyboardBuilder()

start_kb.row(
    types.InlineKeyboardButton(text="ℹ️ About", callback_data="about"),
    types.InlineKeyboardButton(text="⚙️ Settings", callback_data="settings"),
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
    types.BotCommand(command="adduser", description="Add a user to the allowed list"),
]

modelname = os.getenv("INITMODEL")
mention = None
CHAT_TYPE_GROUP = "group"
CHAT_TYPE_SUPERGROUP = "supergroup"

def perms_allowed(func):
    @wraps(func)
    async def wrapper(update: Union[types.Message, types.CallbackQuery], *args, **kwargs):
        if isinstance(update, types.CallbackQuery):
            message = update.message
            query = update
        else:
            message = update
            query = None
        user_id = query.from_user.id if query else message.from_user.id
        if user_id in admin_ids or user_id in allowed_ids:
            if query:
                return await func(query, *args, **kwargs)
            elif message:
                return await func(message, *args, **kwargs)
        else: 
            if query:
                if message:
                    if message.chat.type in ["supergroup", "group"]:
                        return
                await query.answer("Access Denied")
            if message:
                if message.chat.type in ["supergroup", "group"]:
                    if allow_all_users_in_groups:
                        return await func(message, *args, **kwargs)
                    return
                await message.answer("Access Denied")
    return wrapper

def perms_admins(func):
    @wraps(func)
    async def wrapper(update: Union[types.Message, types.CallbackQuery], *args, **kwargs):
        if isinstance(update, types.CallbackQuery):
            message = update.message
            query = update
        else:
            message = update
            query = None
        user_id = query.from_user.id if query else message.from_user.id 
        if user_id in admin_ids:
            if query:
                return await func(query, *args, **kwargs)
            elif message:
                return await func(message, *args, **kwargs)
        else:
            if query:
                if message:
                    if message.chat.type in ["supergroup", "group"]:
                        return
                await query.answer("Access Denied")
                logging.info(
                    f"[QUERY] {message.from_user.first_name} {message.from_user.last_name}({message.from_user.id}) is not allowed to use this bot."
                )
            elif message:
                if message.chat.type in ["supergroup", "group"]:
                    return
                await message.answer("Access Denied")
                logging.info(
                    f"[MSG] {message.from_user.first_name} {message.from_user.last_name}({message.from_user.id}) is not allowed to use this bot."
                )
    return wrapper


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
    register_user(message.from_user.id, message.from_user.full_name)
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

@dp.message(Command("adduser"))
@perms_admins
async def add_user_handler(message: Message):
    global allowed_ids
    user_id = int(message.text.split(maxsplit=1)[1]) if len(message.text.split()) > 1 else None
    if user_id:
        if user_id not in allowed_ids:
            allowed_ids.append(user_id)
            logging.info(f"Allowed IDs: {allowed_ids}")
            await message.answer("User added to the allowed list.")
        else:
            await message.answer("User already in the allowed list.")
    else:
        await message.answer("Please provide a user ID to add.")

        
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
        text=f"<b>Your LLMs</b>\nCurrently using: <code>{modelname}</code>\nDefault in .env: <code>{dotenv_model}</code>\nThis project is under <a href='https://github.com/Ali-Flt/ollama-telegram/blob/main/LICENSE'>MIT License.</a>\n<a href='https://github.com/Ali-Flt/ollama-telegram'>Source Code</a>",
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
    global allowed_ids
    user_id = int(query.data.split("_")[1])
    allowed_ids = [id for id in allowed_ids if id != user_id]
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
    update_selected_system_prompt(query.from_user.id, selected_prompt_id)
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
    delete_system_prompt(prompt_id)
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
        if prompt is None:
            prompt = message.text or message.caption
        if message.content_type == "voice":
            voice = message.voice
            with tempfile.TemporaryDirectory() as temp_dir:
                file = await bot.get_file(voice.file_id)
                file_path = Path(temp_dir) / f"{voice.file_unique_id}.ogg"                
                await bot.download_file(file.file_path, str(file_path))
                result = speech_to_text(file_path)
            prompt = result.get('transcription', 'No transcription available')
            logging.info(f"Voice prompt: {prompt}")
        messages = []
        system_prompt = get_selected_system_prompt(message.from_user.id)
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
                }) 
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
    global allowed_ids
    init_db()
    db_users = get_all_users_from_db()
    for id, _ in db_users:
        if id not in allowed_ids:
            allowed_ids.append(id)
    middleware = AlbumMiddleware()
    await bot.set_my_commands(commands)
    dp.message.outer_middleware(middleware)
    await dp.start_polling(bot, skip_update=True)

if __name__ == "__main__":
    asyncio.run(main())
