import logging
import os
import json
import sqlite3
from dotenv import load_dotenv
import json

load_dotenv()
default_prompt = os.getenv("DEFAULT_PROMPT", "You are a helpful AI voice assistant. We are interacting via voice so keep responses concise, no more than to a sentence or two unless the user specifies a longer response. Do not use special characters and only respond in plain text.")
log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = logging.getLevelName(log_level_str)
logging.basicConfig(level=log_level)

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
    
def update_selected_system_prompt(user_id, selected_prompt_id):
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("UPDATE users SET selected_prompt_id = ? WHERE id = ?", (selected_prompt_id, user_id))
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
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    if c.fetchall():
        return
    c.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?)", (user_id, user_name, -1))
    conn.commit()
    add_system_prompt(user_id, default_prompt)
    c.execute("SELECT id FROM system_prompts WHERE user_id = ?", (user_id,))
    selected_prompt_id = c.fetchall()[-1][0]
    update_selected_system_prompt(user_id, selected_prompt_id)
    conn.close()

def save_chat_message(user_id, role, content, images=[]):
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_id, role, content, images) VALUES (?, ?, ?, ?)",
              (user_id, role, content, json.dumps(images)))
    conn.commit()
    conn.close()
            
def add_system_prompt(user_id, prompt):
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("INSERT INTO system_prompts (user_id, prompt) VALUES (?, ?)",
              (user_id, prompt))
    conn.commit()
    conn.close()

def get_selected_system_prompt(user_id):
    system_prompt = None
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("SELECT selected_prompt_id FROM users WHERE id = ?", (user_id,))
    selected_prompt_id = c.fetchone()
    selected_prompt_id = selected_prompt_id[0] if selected_prompt_id is not None else -1
    logging.info(f"Selected prompt ID: {selected_prompt_id}")
    conn.close()
    system_prompts = get_system_prompts(user_id=user_id)
    if system_prompts:
        for sp in system_prompts:
            if sp[0] == selected_prompt_id:
                system_prompt = sp[2]
                break
        if system_prompt is None:
            logging.warning(f"Selected prompt ID {selected_prompt_id} not found for user {user_id}")
    return system_prompt

def get_system_prompts(user_id=None):
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    query = "SELECT * FROM system_prompts WHERE 1=1"
    params = []
    if user_id is not None:
        query += " AND user_id = ?"
        params.append(user_id)
    c.execute(query, params)
    prompts = c.fetchall()
    conn.close()
    return prompts

def delete_system_prompt(prompt_id):
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("DELETE FROM system_prompts WHERE id = ?", (prompt_id,))
    conn.commit()
    conn.close()

def get_all_users_from_db():
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("SELECT id, name FROM users")
    users = c.fetchall()
    conn.close()
    return users

def remove_user_from_db(user_id):
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    removed = c.rowcount > 0
    conn.commit()
    conn.close()
    return removed
