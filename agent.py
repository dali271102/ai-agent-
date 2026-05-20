import os
import sys
import time
import json
import asyncio
import base64
import threading
import subprocess
import tempfile
import winreg
import webbrowser
import queue
import importlib
import importlib.util
import inspect
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime, timedelta

import psutil
import requests
import pyautogui
import pygetwindow as gw
from PIL import Image, ImageDraw, ImageFont, ImageTk
from dotenv import load_dotenv

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters, CallbackQueryHandler
)

import whisper

# ============= CONFIGURATION =============
load_dotenv()
DATA_DIR        = Path("data")
PLUGINS_DIR     = Path("plugins")
MEMORY_DIR      = Path("memory")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
TOP_K           = 3

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "")
try:
    ALLOWED_USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0") or "0")
except ValueError:
    ALLOWED_USER_ID = 0

# ── FREE MODEL REGISTRY ──────────────────────────────────────────────────────
FREE_MODELS = {
    # Groq (free tier, very fast)
    "llama-3.1-8b  [Groq]":        {"provider": "groq",      "model": "llama-3.1-8b-instant"},
    "llama-3.3-70b [Groq]":        {"provider": "groq",      "model": "llama-3.3-70b-versatile"},
    "mixtral-8x7b  [Groq]":        {"provider": "groq",      "model": "mixtral-8x7b-32768"},
    "gemma2-9b     [Groq]":        {"provider": "groq",      "model": "gemma2-9b-it"},
    # OpenRouter free models
    "mistral-7b    [OpenRouter]":   {"provider": "openrouter","model": "mistralai/mistral-7b-instruct:free"},
    "phi-3-mini    [OpenRouter]":   {"provider": "openrouter","model": "microsoft/phi-3-mini-128k-instruct:free"},
    "llama-3.1-8b  [OpenRouter]":   {"provider": "openrouter","model": "meta-llama/llama-3.1-8b-instruct:free"},
    "qwen2-7b      [OpenRouter]":   {"provider": "openrouter","model": "qwen/qwen-2-7b-instruct:free"},
    # Ollama local models (if running)
    "ollama:llama3":                {"provider": "ollama",    "model": "llama3"},
    "ollama:mistral":               {"provider": "ollama",    "model": "mistral"},
    "ollama:phi3":                  {"provider": "ollama",    "model": "phi3"},
}

# ============= XP COLOR PALETTE =============
XP = {
    "bg":           "#ECE9D8",
    "bg_dark":      "#D4D0C8",
    "btn":          "#D4D0C8",
    "title_bar":    "#0A246A",
    "title_fg":     "#FFFFFF",
    "highlight":    "#316AC5",
    "highlight_fg": "#FFFFFF",
    "chat_bg":      "#FFFFFF",
    "user_fg":      "#0A246A",
    "bot_fg":       "#1A5C1A",
    "tg_fg":        "#8B008B",
    "system_fg":    "#808080",
    "error_fg":     "#CC0000",
    "plugin_fg":    "#B8520A",
    "font":         "Tahoma",
    "font_size":    9,
    "taskbar_bg":   "#245EDC",
    "taskbar_btn":  "#3A7FDE",
}

# ============= PERSISTENT MEMORY =============
class PersistentMemory:
    """
    Stores conversation history, facts, and summaries to disk.
    Survives restarts. Provides sliding-window + summary compression.
    """
    def __init__(self, memory_dir: Path = MEMORY_DIR, max_window: int = 30):
        self.dir        = memory_dir
        self.dir.mkdir(exist_ok=True)
        self.max_window = max_window
        self.history:   List[Dict] = []
        self.facts:     Dict[str, str] = {}
        self.summary:   str = ""
        self._load()

    def _path(self, name): return self.dir / f"{name}.json"

    def _load(self):
        for attr, default in [("history", []), ("facts", {}), ("summary", "")]:
            p = self._path(attr)
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    setattr(self, attr, data)
                except Exception:
                    pass

    def save(self):
        for attr in ("history", "facts", "summary"):
            try:
                self._path(attr).write_text(
                    json.dumps(getattr(self, attr), ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
            except Exception:
                pass

    def add(self, role: str, text: str):
        self.history.append({
            "role": role, "text": text,
            "ts": datetime.now().isoformat(timespec="seconds")
        })
        if len(self.history) > self.max_window * 2:
            self._compress()
        self.save()

    def remember_fact(self, key: str, value: str):
        self.facts[key] = value
        self.save()

    def get_context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[Earlier summary]\n{self.summary}\n")
        if self.facts:
            facts_str = "\n".join(f"  • {k}: {v}" for k, v in list(self.facts.items())[-10:])
            parts.append(f"[Known facts about user]\n{facts_str}\n")
        recent = self.history[-self.max_window:]
        for turn in recent:
            parts.append(f"{turn['role'].title()}: {turn['text']}")
        return "\n".join(parts)

    def _compress(self):
        old  = self.history[:-self.max_window]
        keep = self.history[-self.max_window:]
        old_text = "\n".join(f"{t['role']}: {t['text']}" for t in old)
        self.summary = (self.summary + "\n" + old_text).strip()[-4000:]
        self.history = keep
        self.save()

    def clear(self):
        self.history = []
        self.facts   = {}
        self.summary = ""
        self.save()

    def search(self, query: str, top_k: int = 5) -> List[str]:
        q    = query.lower()
        hits = [f"{t['role']}: {t['text']}" for t in self.history if q in t['text'].lower()]
        return hits[-top_k:]


# ============= PLUGIN SYSTEM =============
class Plugin:
    """Base class every plugin must inherit from."""
    name:        str = "unnamed"
    description: str = ""
    version:     str = "1.0"
    author:      str = ""

    def __init__(self, agent_ref):
        self.agent = agent_ref

    def on_load(self): pass
    def on_unload(self): pass

    def handle(self, command: str) -> Optional[str]:
        return None

    def get_commands(self) -> Dict[str, str]:
        return {}


# ── EXAMPLE PLUGIN SOURCE CODE (written to /plugins on first run) ────────────
# These are kept as module-level constants so we don't have to worry about
# triple-quote / indentation interactions inside method bodies.

_WEATHER_PLUGIN_SRC = '''"""
Weather Plugin — fetches current weather using wttr.in (free, no API key).
"""
from plugins import Plugin
from typing import Optional
import requests

class WeatherPlugin(Plugin):
    name        = "weather"
    description = "Get current weather for any city (free, no API key)"
    version     = "1.0"
    author      = "built-in"

    def get_commands(self):
        return {"weather": "weather in [city]  — get current weather"}

    def handle(self, command: str) -> Optional[str]:
        cmd = command.lower().strip()
        if not (cmd.startswith("weather") or "weather in" in cmd or "what is the weather" in cmd):
            return None
        city = ""
        for trigger in ["weather in ", "weather for ", "what is the weather in ", "weather "]:
            if trigger in cmd:
                city = cmd.split(trigger, 1)[1].strip()
                break
        if not city:
            return "❓ Specify a city: weather in Paris"
        try:
            r = requests.get(f"https://wttr.in/{requests.utils.quote(city)}?format=3", timeout=8)
            if r.status_code == 200:
                return f"🌤️ {r.text.strip()}"
            return f"❌ Could not fetch weather for {city}"
        except Exception as e:
            return f"❌ Weather error: {e}"
'''

_CLIPBOARD_PLUGIN_SRC = '''"""
Clipboard Plugin — read and write the Windows clipboard.
"""
from plugins import Plugin
from typing import Optional
import subprocess

class ClipboardPlugin(Plugin):
    name        = "clipboard"
    description = "Read/write the clipboard"
    version     = "1.0"
    author      = "built-in"

    def get_commands(self):
        return {
            "clipboard read":  "Read clipboard contents",
            "clipboard write": "clipboard write [text] — write text to clipboard",
            "clipboard clear": "Clear the clipboard",
        }

    def handle(self, command: str) -> Optional[str]:
        cmd = command.lower().strip()
        if "clipboard" not in cmd:
            return None

        if "read" in cmd or "show clipboard" in cmd or "get clipboard" in cmd:
            try:
                r = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
                                   capture_output=True, text=True, timeout=5)
                text = (r.stdout or "").strip()
                return f"📋 Clipboard:\\n{text}" if text else "📋 Clipboard is empty"
            except Exception as e:
                return f"❌ {e}"

        if "write" in cmd or "copy" in cmd:
            for trigger in ["clipboard write ", "clipboard copy "]:
                idx = cmd.find(trigger)
                if idx != -1:
                    text = command[idx + len(trigger):]
                    # Single quotes inside a PS single-quoted string are escaped by doubling them
                    escaped = text.replace("'", "''")
                    ps_cmd = f"Set-Clipboard -Value '{escaped}'"
                    try:
                        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                                       capture_output=True, text=True, timeout=5)
                        return f"📋 Copied to clipboard: {text[:60]}"
                    except Exception as e:
                        return f"❌ {e}"
            return "❓ Format: clipboard write [text]"

        if "clear" in cmd:
            try:
                subprocess.run(["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value $null"],
                               capture_output=True, text=True, timeout=5)
                return "📋 Clipboard cleared"
            except Exception as e:
                return f"❌ {e}"
        return None
'''

_REMINDER_PLUGIN_SRC = '''"""
Reminder Plugin — call-like Telegram notification with snooze/dismiss.
"""
from typing import Optional
import threading, time, re, json
from datetime import datetime, timedelta
from pathlib import Path
from plugins import Plugin

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
except ImportError:
    InlineKeyboardButton = None
    InlineKeyboardMarkup = None

REMINDERS_FILE = Path("memory/reminders.json")


class ReminderPlugin(Plugin):
    name        = "reminder"
    description = "Set reminders with call-like Telegram notifications + snooze"
    version     = "2.0"
    author      = "upgraded"

    def __init__(self, agent_ref):
        super().__init__(agent_ref)
        self.pending  = {}
        self._counter = 0
        self._lock    = threading.Lock()
        self._load_saved()

    def _load_saved(self):
        if not REMINDERS_FILE.exists():
            return
        try:
            saved = json.loads(REMINDERS_FILE.read_text(encoding="utf-8"))
            now   = datetime.now().timestamp()
            for r in saved:
                delay = r["fire_at"] - now
                if delay > 0:
                    rid = r.get("id", self._next_id())
                    with self._lock:
                        self.pending[rid] = {"task": r["task"], "fire_at": r["fire_at"]}
                    self._schedule(r["task"], delay, rid)
        except Exception:
            pass

    def _save(self):
        REMINDERS_FILE.parent.mkdir(exist_ok=True)
        data = [
            {"id": rid, "task": info["task"], "fire_at": info["fire_at"]}
            for rid, info in self.pending.items()
        ]
        try:
            REMINDERS_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def get_commands(self):
        return {
            "remind":          "remind me in [N] seconds/minutes/hours to [task]",
            "list reminders":  "show all pending reminders",
            "cancel reminder": "cancel reminder [id]",
        }

    def handle(self, command: str) -> Optional[str]:
        cmd = command.lower().strip()

        if "list reminders" in cmd or "show reminders" in cmd or "my reminders" in cmd:
            return self._list_reminders()

        if "cancel reminder" in cmd or "delete reminder" in cmd:
            return self._cancel_reminder(cmd)

        if not (cmd.startswith("remind") or "remind me" in cmd):
            return None

        m = re.search(r"in\\s+(\\d+)\\s*(second|minute|hour|sec|min|hr)s?", cmd)
        task_match = re.search(r"\\bto\\s+(.+)$", cmd)

        if not m:
            return "❓ Format: remind me in 10 seconds to drink water"

        amount = int(m.group(1))
        unit   = m.group(2)
        task   = task_match.group(1).strip() if task_match else "something"

        mult  = {"second":1,"sec":1,"minute":60,"min":60,"hour":3600,"hr":3600}
        delay = amount * mult.get(unit, 60)

        fire_at = datetime.now() + timedelta(seconds=delay)
        rid     = self._next_id()

        with self._lock:
            self.pending[rid] = {"task": task, "fire_at": fire_at.timestamp()}
        self._save()
        self._schedule(task, delay, rid)

        return (
            f"⏰ Reminder #{rid} set!\\n"
            f"   Task: {task}\\n"
            f"   Fires at: {fire_at.strftime('%H:%M:%S')}"
        )

    def _next_id(self):
        self._counter += 1
        return self._counter

    def _schedule(self, task, delay, rid):
        def _fire():
            time.sleep(delay)
            self._fire_reminder(task, rid)
        threading.Thread(target=_fire, daemon=True).start()

    def _fire_reminder(self, task, rid):
        with self._lock:
            self.pending.pop(rid, None)
        self._save()

        agent = self.agent

        # XP chat popup
        if hasattr(agent, "root") and hasattr(agent, "_system_msg"):
            try:
                agent.root.after(0, lambda: agent._system_msg(
                    f"📞 ══════════════════════\\n"
                    f"   REMINDER #{rid}\\n"
                    f"   🔔 {task}\\n"
                    f"   ══════════════════════", "error"
                ))
            except Exception:
                pass

        # Telegram call-like notification
        tg = getattr(agent, "tg_bot", None)
        if not (tg and getattr(tg, "running", False) and getattr(tg, "_loop", None)):
            return

        chat_id = getattr(tg, "_last_chat_id", None)
        if not chat_id:
            return

        import asyncio

        if InlineKeyboardMarkup is None:
            try:
                asyncio.run_coroutine_threadsafe(
                    tg.app.bot.send_message(chat_id, f"⏰ REMINDER: {task}"),
                    tg._loop
                )
            except Exception:
                pass
            return

        text = (
            "📞 *INCOMING REMINDER*\\n"
            "━━━━━━━━━━━━━━━━━━━━\\n"
            f"🔔  *{task}*\\n"
            "━━━━━━━━━━━━━━━━━━━━"
        )
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Dismiss",      callback_data=f"rem_dismiss:{rid}"),
            InlineKeyboardButton("⏰ Snooze 5min", callback_data=f"rem_snooze:{rid}:{task}"),
        ]])

        async def _send():
            await tg.app.bot.send_message(
                chat_id, text,
                parse_mode="Markdown",
                reply_markup=keyboard
            )
        try:
            asyncio.run_coroutine_threadsafe(_send(), tg._loop)
        except Exception:
            pass

    def _list_reminders(self):
        with self._lock:
            items = list(self.pending.items())
        if not items:
            return "📋 No pending reminders."
        lines = [f"📋 Pending reminders ({len(items)}):"]
        for rid, info in items:
            fire = datetime.fromtimestamp(info["fire_at"]).strftime("%H:%M:%S")
            lines.append(f"  #{rid} — {info['task']}  (at {fire})")
        lines.append("\\nCancel: cancel reminder [id]")
        return "\\n".join(lines)

    def _cancel_reminder(self, cmd):
        m = re.search(r"(\\d+)", cmd)
        if not m:
            return "❓ Format: cancel reminder 1"
        rid = int(m.group(1))
        with self._lock:
            if rid in self.pending:
                del self.pending[rid]
                self._save()
                return f"✅ Reminder #{rid} cancelled."
        return f"❌ Reminder #{rid} not found."
'''

_PLUGINS_INIT_SRC = (
    "from pathlib import Path\n"
    "import sys\n"
    "sys.path.insert(0, str(Path(__file__).parent.parent))\n"
    "from plugin_base import Plugin\n"
)

_PLUGIN_BASE_SRC = (
    "from typing import Optional, Dict\n\n"
    "class Plugin:\n"
    "    name = 'unnamed'\n"
    "    description = ''\n"
    "    version = '1.0'\n"
    "    author = ''\n"
    "    def __init__(self, agent_ref): self.agent = agent_ref\n"
    "    def on_load(self): pass\n"
    "    def on_unload(self): pass\n"
    "    def handle(self, command):\n"
    "        return None\n"
    "    def get_commands(self):\n"
    "        return {}\n"
)


class PluginManager:
    """
    Discovers, loads, and manages plugins from the plugins/ directory.
    Each plugin is a .py file with a class that inherits from Plugin.
    """
    def __init__(self, plugins_dir: Path, agent_ref, log_cb=None):
        self.dir      = plugins_dir
        self.dir.mkdir(exist_ok=True)
        self.agent    = agent_ref
        self.log      = log_cb or print
        self.plugins:  Dict[str, Plugin] = {}
        self._write_example_plugins()

    def _write_example_plugins(self):
        wp = self.dir / "weather_plugin.py"
        if not wp.exists():
            wp.write_text(_WEATHER_PLUGIN_SRC, encoding="utf-8")

        cp = self.dir / "clipboard_plugin.py"
        if not cp.exists():
            cp.write_text(_CLIPBOARD_PLUGIN_SRC, encoding="utf-8")

        rp = self.dir / "reminder_plugin.py"
        if not rp.exists():
            rp.write_text(_REMINDER_PLUGIN_SRC, encoding="utf-8")

        init = self.dir / "__init__.py"
        if not init.exists():
            init.write_text(_PLUGINS_INIT_SRC, encoding="utf-8")

        base = Path("plugin_base.py")
        if not base.exists():
            base.write_text(_PLUGIN_BASE_SRC, encoding="utf-8")

    def load_all(self) -> List[str]:
        loaded = []
        parent = str(self.dir.parent.resolve())
        if parent not in sys.path:
            sys.path.insert(0, parent)
        for py_file in self.dir.glob("*_plugin.py"):
            result = self._load_file(py_file)
            if result:
                loaded.append(result)
        return loaded

    def _load_file(self, path: Path) -> Optional[str]:
        try:
            spec   = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if (hasattr(cls, "handle") and hasattr(cls, "name")
                        and getattr(cls, "name", "unnamed") != "unnamed"
                        and cls.__module__ == module.__name__):
                    instance = cls(self.agent)
                    instance.on_load()
                    self.plugins[cls.name] = instance
                    self.log(f"🔌 Plugin loaded: {cls.name} v{cls.version}")
                    return cls.name
        except Exception as e:
            self.log(f"❌ Plugin error ({path.name}): {e}")
        return None

    def reload(self, name: str) -> bool:
        for path in self.dir.glob("*_plugin.py"):
            result = self._load_file(path)
            if result == name:
                return True
        return False

    def unload(self, name: str):
        if name in self.plugins:
            try:
                self.plugins[name].on_unload()
            except Exception:
                pass
            del self.plugins[name]

    def handle(self, command: str) -> Optional[str]:
        for plugin in list(self.plugins.values()):
            try:
                result = plugin.handle(command)
                if result is not None:
                    return result
            except Exception as e:
                self.log(f"⚠️ Plugin '{plugin.name}' error: {e}")
        return None

    def list_plugins(self) -> str:
        if not self.plugins:
            return "🔌 No plugins loaded. Add *_plugin.py files to /plugins/"
        lines = ["🔌 Loaded plugins:"]
        for p in self.plugins.values():
            cmds = ", ".join(p.get_commands().keys()) or "—"
            lines.append(f"  • {p.name} v{p.version} — {p.description}")
            lines.append(f"    Commands: {cmds}")
        return "\n".join(lines)


# ============= MULTI-MODEL LLM MANAGER =============
class LLMManager:
    """
    Manages multiple free LLM providers: Groq, OpenRouter, Ollama.
    Supports streaming via callbacks.
    """
    def __init__(self, log_cb=None):
        self.log           = log_cb or print
        self.current_key   = "llama-3.1-8b  [Groq]"
        self.groq_key      = os.getenv("GROQ_API_KEY", "")
        self.openrouter_key= os.getenv("OPENROUTER_API_KEY", "")
        self.ollama_url    = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self._llm          = None
        self._lock         = threading.Lock()

    def switch_model(self, key: str) -> bool:
        if key not in FREE_MODELS:
            return False
        self.current_key = key
        self._llm = None
        return True

    def chat(self, messages: List[Dict], stream_cb: Optional[Callable[[str], None]] = None) -> str:
        cfg = FREE_MODELS[self.current_key]
        provider = cfg["provider"]
        model    = cfg["model"]

        if provider == "groq":
            return self._chat_groq(model, messages, stream_cb)
        elif provider == "openrouter":
            return self._chat_openrouter(model, messages, stream_cb)
        elif provider == "ollama":
            return self._chat_ollama(model, messages, stream_cb)
        raise ValueError(f"Unknown provider: {provider}")

    def _chat_groq(self, model: str, messages: List[Dict], stream_cb) -> str:
        if not self.groq_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        url     = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.groq_key}",
                   "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages,
                   "temperature": 0.7, "stream": stream_cb is not None}
        r = requests.post(url, headers=headers, json=payload,
                          stream=stream_cb is not None, timeout=60)
        r.raise_for_status()
        return self._handle_stream(r, stream_cb) if stream_cb else \
               r.json()["choices"][0]["message"]["content"]

    def _chat_openrouter(self, model: str, messages: List[Dict], stream_cb) -> str:
        if not self.openrouter_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
        url     = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.openrouter_key}",
                   "Content-Type": "application/json",
                   "HTTP-Referer": "https://agent-xp.local"}
        payload = {"model": model, "messages": messages,
                   "temperature": 0.7, "stream": stream_cb is not None}
        r = requests.post(url, headers=headers, json=payload,
                          stream=stream_cb is not None, timeout=60)
        r.raise_for_status()
        return self._handle_stream(r, stream_cb) if stream_cb else \
               r.json()["choices"][0]["message"]["content"]

    def _chat_ollama(self, model: str, messages: List[Dict], stream_cb) -> str:
        url     = f"{self.ollama_url}/api/chat"
        payload = {"model": model, "messages": messages, "stream": stream_cb is not None}
        r = requests.post(url, json=payload,
                          stream=stream_cb is not None, timeout=120)
        r.raise_for_status()
        if stream_cb:
            full = ""
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        tok  = data.get("message", {}).get("content", "")
                        if tok:
                            stream_cb(tok)
                            full += tok
                    except Exception:
                        pass
            return full
        return r.json()["message"]["content"]

    def _handle_stream(self, response, cb: Callable[[str], None]) -> str:
        full = ""
        for line in response.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                    tok = delta.get("content", "")
                    if tok:
                        cb(tok)
                        full += tok
                except Exception:
                    pass
        return full

    def available_models(self) -> List[str]:
        return list(FREE_MODELS.keys())

    def current_info(self) -> str:
        cfg = FREE_MODELS[self.current_key]
        return f"{self.current_key}  ({cfg['provider']})"


# ============= AGENT PERSONAS =============
class AgentPersona:
    PERSONAS = {
        "assistant": {
            "name": "Assistant", "emoji": "🤖",
            "description": "Helpful, efficient, and professional AI assistant",
            "traits": "Be helpful, concise, and efficient. Focus on getting tasks done.",
            "response_style": "professional, clear, solution-oriented",
            "image_style": "professional, clean, minimalist"
        },
        "friend": {
            "name": "Friend", "emoji": "👋",
            "description": "Casual, friendly conversational companion",
            "traits": "Be warm, empathetic, and conversational.",
            "response_style": "conversational, warm, engaging",
            "image_style": "friendly, warm, casual"
        },
        "mentor": {
            "name": "Mentor", "emoji": "🎓",
            "description": "Wise, experienced guide and teacher",
            "traits": "Be knowledgeable, patient, and encouraging.",
            "response_style": "wise, instructive, supportive",
            "image_style": "professional, authoritative, calm"
        },
        "creative": {
            "name": "Creative Partner", "emoji": "🎨",
            "description": "Creative brainstorming partner",
            "traits": "Be imaginative, open-minded, and playful.",
            "response_style": "creative, inspiring, imaginative",
            "image_style": "artistic, colorful, creative"
        },
        "analyst": {
            "name": "Analyst", "emoji": "📊",
            "description": "Data-driven analytical thinker",
            "traits": "Be logical, precise, and analytical.",
            "response_style": "analytical, structured, data-focused",
            "image_style": "clean, organized, minimal"
        },
        "coder": {
            "name": "Code Expert", "emoji": "💻",
            "description": "Senior software engineer and code reviewer",
            "traits": "Give concise, correct code. Explain briefly. Prefer working examples.",
            "response_style": "technical, precise, code-first",
            "image_style": "dark theme, code, technical"
        },
    }

    def __init__(self):
        self.current_persona  = "assistant"
        self.persona_history  = []

    def switch_persona(self, name: str) -> bool:
        if name in self.PERSONAS:
            self.persona_history.append(self.current_persona)
            self.current_persona = name
            return True
        return False

    def get_current_persona(self) -> Dict:
        return self.PERSONAS[self.current_persona]

    def get_persona_prompt(self) -> str:
        p = self.get_current_persona()
        return (
            f"You are {p['name']}, an AI {p['description']}.\n"
            f"PERSONALITY: {p['traits']}\n"
            f"STYLE: {p['response_style']}\n"
        )


# ============= TASK MANAGER =============
class TaskManager:
    def __init__(self):
        self.tasks: List[Dict] = []

    def add_task(self, task: str, priority: str = "normal") -> str:
        self.tasks.append({"task": task, "priority": priority,
                           "created": datetime.now().isoformat(), "status": "pending"})
        return f"✅ Task added: {task}"

    def complete_task(self, index: int) -> str:
        if 0 <= index < len(self.tasks):
            self.tasks[index]["status"] = "done"
            return f"✅ Task completed: {self.tasks[index]['task']}"
        return "❌ Invalid task index"

    def list_tasks(self) -> str:
        if not self.tasks:
            return "📋 No tasks."
        lines = ["📋 Tasks:"]
        for i, t in enumerate(self.tasks):
            icon = "✅" if t["status"] == "done" else "⏳"
            lines.append(f"  {i}. {icon} [{t['priority']}] {t['task']}")
        return "\n".join(lines)

    def get_active_context(self) -> str:
        pending = [t for t in self.tasks if t["status"] == "pending"]
        if pending:
            return f"\nPending tasks: {', '.join(t['task'] for t in pending[:3])}"
        return ""

    def clear_tasks(self) -> str:
        self.tasks = []
        return "🧹 Tasks cleared"


# ============= FILE TEMPLATES =============
def _tpl_py(name):
    return f'#!/usr/bin/env python3\n"""\n{name}\nCreated: {datetime.now():%Y-%m-%d %H:%M}\n"""\n\ndef main():\n    print("Hello from {name}!")\n\nif __name__ == "__main__":\n    main()\n'

def _tpl_js(name):
    return f'// {name}\n// Created: {datetime.now():%Y-%m-%d %H:%M}\n"use strict";\n\nfunction main() {{\n    console.log("Hello from {name}!");\n}}\nmain();\n'

def _tpl_html(name):
    return f'<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <title>{name}</title>\n</head>\n<body>\n    <h1>{name}</h1>\n</body>\n</html>\n'

def _tpl_md(name):
    return f'# {name}\n\n> Created: {datetime.now():%Y-%m-%d %H:%M}\n\n## Description\n\nWrite here.\n'

def _tpl_json(name):
    return json.dumps({"name": name, "created": f"{datetime.now():%Y-%m-%d}", "version": "1.0.0"}, indent=2)

FILE_TEMPLATES = {
    ".py": _tpl_py, ".js": _tpl_js, ".ts": _tpl_js,
    ".html": _tpl_html, ".css": lambda n: f"/* {n} */\n* {{ box-sizing: border-box; }}\n",
    ".json": _tpl_json, ".md": _tpl_md, ".txt": lambda n: f"{n}\n{'='*40}\n\n",
    ".sh": lambda n: f"#!/bin/bash\n# {n}\necho 'Hello from {n}!'\n",
    ".bat": lambda n: f"@echo off\necho Hello from {n}!\npause\n",
    ".ps1": lambda n: f"# {n}\nWrite-Host 'Hello from {n}!'\n",
    ".yaml": lambda n: f"# {n}\nname: {Path(n).stem}\nversion: 1.0.0\n",
    ".yml":  lambda n: f"# {n}\nname: {Path(n).stem}\n",
    ".sql":  lambda n: f"-- {n}\nSELECT 1;\n",
    ".env":  lambda n: f"# {n}\nAPP_NAME={Path(n).stem}\nDEBUG=true\n",
    ".gitignore": lambda n: "__pycache__/\n*.pyc\n.env\nnode_modules/\n.DS_Store\n",
    ".rs":   lambda n: f'// {n}\nfn main() {{\n    println!("Hello from {n}!");\n}}\n',
    ".go":   lambda n: f'// {n}\npackage main\nimport "fmt"\nfunc main() {{\n    fmt.Println("Hello from {n}!")\n}}\n',
    ".cpp":  lambda n: f'// {n}\n#include <iostream>\nusing namespace std;\nint main() {{\n    cout << "Hello!" << endl;\n    return 0;\n}}\n',
    ".c":    lambda n: f'// {n}\n#include <stdio.h>\nint main() {{\n    printf("Hello!\\n");\n    return 0;\n}}\n',
}


# ============= LOCAL PC AGENT =============
class LocalPCAgent:
    APP_MAP = {
        "notepad": "notepad.exe", "calculator": "calc.exe", "calc": "calc.exe",
        "paint": "mspaint.exe", "explorer": "explorer.exe",
        "chrome": "chrome.exe", "firefox": "firefox.exe", "edge": "msedge.exe",
        "word": "WINWORD.EXE", "excel": "EXCEL.EXE", "powerpoint": "POWERPNT.EXE",
        "vlc": "vlc.exe", "spotify": "Spotify.exe", "discord": "Discord.exe",
        "vscode": "code.exe", "pycharm": "pycharm64.exe",
        "terminal": "cmd.exe", "cmd": "cmd.exe", "powershell": "powershell.exe",
        "task manager": "taskmgr.exe", "control panel": "control.exe",
        "settings": "ms-settings:", "obs": "obs64.exe",
        "steam": "steam.exe", "winamp": "winamp.exe",
    }

    def __init__(self, log_cb=None):
        self.log_cb = log_cb or print
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE    = 0.3

    def handle(self, command: str) -> Optional[str]:
        cmd = command.lower().strip()

        # File operations
        if any(w in cmd for w in ["create file","make file","new file","create a file","make a file","creer fichier","créer fichier","nouveau fichier"]):
            return self._create_file(command)
        if any(w in cmd for w in ["find file","search file","locate file"]):
            return self._find_files(command)
        if any(w in cmd for w in ["list files","show files","files in","what's in","whats in"]):
            return self._list_folder(command)
        if any(w in cmd for w in ["open file","open the file","ouvrir fichier"]):
            return self._open_file(command)
        if any(w in cmd for w in ["delete file","remove file"]):
            return self._delete_file(command)
        if "rename file" in cmd:
            return self._rename_file(command)
        if "move file" in cmd:
            return self._move_file(command)
        if "copy file" in cmd:
            return self._copy_file(command)
        if any(w in cmd for w in ["read file","show file content","cat file"]):
            return self._read_file(command)
        if any(w in cmd for w in ["open desktop","open documents","open downloads","open pictures","open videos","open music"]):
            return self._open_location(command)

        # Apps
        if any(w in cmd for w in ["list apps","installed apps","what apps","show apps","my apps"]):
            return self._list_installed_apps()
        if any(w in cmd for w in ["plugins","list plugins","show plugins"]):
            return "→ plugin_list"

        # Smart open detection: open <file-with-ext> → open_file
        if cmd.startswith(("open ","launch ","run ")):
            parts = cmd.split(" ", 1)
            if len(parts) > 1:
                first_tok = parts[1].strip().split()[0].strip('"\'')
                if "." in first_tok and first_tok.lower() not in self.APP_MAP:
                    return self._open_file(command)
        if any(cmd.startswith(w) for w in ["open ","launch ","start ","run "]):
            return self._open_app(command)
        if any(w in cmd for w in ["close ","kill app","stop app"]):
            return self._close_app(command)

        # System stats
        if any(w in cmd for w in ["cpu","ram","memory usage"]):
            return self._system_stats()
        if any(w in cmd for w in ["disk","storage","free space"]):
            return self._disk_stats()
        if any(w in cmd for w in ["battery","power level"]):
            return self._battery_info()
        if any(w in cmd for w in ["system info","pc info","computer info","about my pc"]):
            return self._full_system_info()
        if any(w in cmd for w in ["network","wifi","ip address","internet info"]):
            return self._network_info()
        if any(w in cmd for w in ["top processes","heavy processes","most cpu","most ram"]):
            return self._top_processes()

        # Processes
        if any(w in cmd for w in ["running","processes","active apps","what is running"]):
            return self._running_processes()
        if any(w in cmd for w in ["kill process","end process","stop process"]):
            return self._kill_process(command)

        # Registry / Startup
        if any(w in cmd for w in ["startup apps","startup programs","manage startup"]):
            return self._list_startup()
        if any(w in cmd for w in ["environment variables","env vars"]):
            return self._env_vars()

        # Screenshot / type / hotkey
        if any(w in cmd for w in ["screenshot","screen capture"]):
            return self._take_screenshot()
        if cmd.startswith("type "):
            return self._type_text(command)
        if cmd.startswith("hotkey ") or cmd.startswith("press "):
            return self._send_hotkey(command)

        # Web
        if any(w in cmd for w in ["go to","open website","browse","navigate to"]):
            return self._open_url(command)
        if any(w in cmd for w in ["search for","google ","look up"]):
            return self._search_web(command)

        # Volume / brightness
        if "volume" in cmd:
            return self._control_volume(command)
        if "brightness" in cmd:
            return self._control_brightness(command)

        # Power
        if "cancel shutdown" in cmd or "abort shutdown" in cmd:
            subprocess.run("shutdown /a", shell=True)
            return "✅ Shutdown cancelled"
        if any(w in cmd for w in ["shut down","shutdown","turn off pc","power off"]):
            return self._shutdown_pc()
        if any(w in cmd for w in ["restart","reboot"]):
            return self._restart_pc()
        if any(w in cmd for w in ["sleep","hibernate"]):
            return self._sleep_pc()
        if "lock" in cmd and "pc" in cmd:
            return self._lock_pc()

        return None

    # ── FILE OPERATIONS ───────────────────────────────────────────────────────

    def _create_file(self, command: str) -> str:
        import re
        raw = command.strip()

        triggers = [
            "create a file called","make a file called","new file called","create file called",
            "create a file named","make a file named","creer fichier","créer fichier",
            "nouveau fichier","create a file","make a file","new file","create file","make file",
        ]
        work = raw
        for t in sorted(triggers, key=len, reverse=True):
            if work.lower().startswith(t):
                work = work[len(t):].strip()
                break

        content_override = None
        for sep in ["with content:","with content","containing:","content:"]:
            if sep in work.lower():
                idx = work.lower().index(sep)
                content_override = work[idx+len(sep):].strip()
                work = work[:idx].strip()
                break

        save_dir = Path.home() / "Desktop"
        loc_map  = {
            "desktop": Path.home()/"Desktop", "documents": Path.home()/"Documents",
            "downloads": Path.home()/"Downloads", "pictures": Path.home()/"Pictures",
            "videos": Path.home()/"Videos", "music": Path.home()/"Music",
            "home": Path.home(),
        }
        for loc_kw, loc_path in loc_map.items():
            for prep in [f"in {loc_kw}", f"on {loc_kw}", f"to {loc_kw}"]:
                if prep in work.lower():
                    save_dir = loc_path
                    # Remove only the matched prep (case-insensitive)
                    work = re.sub(re.escape(prep), "", work, flags=re.IGNORECASE).strip()
                    break

        filename = work.strip().strip('"\'')
        if not filename or "." not in filename:
            m = re.search(r'[\w\-]+\.\w+', raw)
            if m:
                filename = m.group(0)
            else:
                return ("❓ Specify the filename with extension.\n"
                        "   Example: create file hello.py")

        safe_name = "".join(c for c in filename if c not in '<>:"/\\|?*').strip()
        ext  = Path(safe_name).suffix.lower()
        stem = Path(safe_name).stem

        if content_override:
            content = content_override
        elif safe_name.lower() == "dockerfile":
            content = 'FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\nCMD ["python","main.py"]\n'
        elif ext in FILE_TEMPLATES:
            content = FILE_TEMPLATES[ext](safe_name)
        else:
            content = f"# {safe_name}\n# Created: {datetime.now():%Y-%m-%d %H:%M}\n\n"

        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / safe_name
        if file_path.exists():
            counter = 1
            while file_path.exists():
                file_path = save_dir / f"{stem}_{counter}{ext}"
                counter += 1

        file_path.write_text(content, encoding="utf-8")
        try:
            os.startfile(str(file_path))
        except Exception:
            pass

        lines   = content.strip().splitlines()
        preview = "\n".join(f"  {l}" for l in lines[:6])
        more    = f"\n  ... ({len(lines)-6} more lines)" if len(lines) > 6 else ""
        return (
            f"✅ File created!\n\n"
            f"  📄 {file_path.name}\n"
            f"  📁 {file_path.parent}\n"
            f"  💾 {file_path.stat().st_size} bytes\n\n"
            f"  📝 Preview:\n{preview}{more}"
        )

    def _open_file(self, command: str) -> str:
        import re
        raw = command.strip()
        triggers = ["open the file","ouvrir le fichier","ouvrir fichier","open file","launch","open","run"]
        work = raw
        for t in sorted(triggers, key=len, reverse=True):
            if work.lower().startswith(t.lower() + " "):
                work = work[len(t):].strip()
                break

        loc_map = {
            "desktop": Path.home()/"Desktop", "documents": Path.home()/"Documents",
            "downloads": Path.home()/"Downloads", "pictures": Path.home()/"Pictures",
            "videos": Path.home()/"Videos", "music": Path.home()/"Music",
        }
        specific_dir = None
        for loc_kw, loc_path in loc_map.items():
            for prep in [f" from {loc_kw}", f" in {loc_kw}", f" on {loc_kw}"]:
                if prep in work.lower():
                    specific_dir = loc_path
                    work = re.sub(re.escape(prep), "", work, flags=re.IGNORECASE).strip()
                    break

        filename = work.strip().strip('"\'')
        if not filename:
            return "❓ Specify file: open file report.pdf"

        candidate = Path(filename)
        if candidate.is_absolute() and candidate.exists():
            os.startfile(str(candidate))
            return f"✅ Opened: {candidate.name}"

        dirs = [specific_dir] if specific_dir else [
            Path.home()/"Desktop", Path.home()/"Documents",
            Path.home()/"Downloads", Path.home(), Path.home()/"Pictures",
        ]
        for d in dirs:
            if not d or not d.exists():
                continue
            try:
                exact = d / filename
                if exact.exists():
                    os.startfile(str(exact))
                    return f"✅ Opened: {exact.name}\n  📁 {exact.parent}"
                for m in d.rglob(filename):
                    if m.is_file():
                        os.startfile(str(m))
                        return f"✅ Opened: {m.name}\n  📁 {m.parent}"
            except (PermissionError, Exception):
                pass
        return f"❌ File '{filename}' not found."

    def _delete_file(self, command: str) -> str:
        import re
        m = re.search(r'(?:delete|remove) file\s+"?([^"]+)"?', command, re.IGNORECASE)
        if not m:
            return "❓ Format: delete file notes.txt"
        filename = m.group(1).strip()
        p = Path(filename)
        if not p.is_absolute():
            for d in [Path.home()/"Desktop", Path.home()/"Documents", Path.home()/"Downloads"]:
                candidate = d / filename
                if candidate.exists():
                    p = candidate
                    break
        if p.exists() and p.is_file():
            try:
                p.unlink()
                return f"🗑️ Deleted: {p.name}"
            except Exception as e:
                return f"❌ {e}"
        return f"❌ File not found: {filename}"

    def _rename_file(self, command: str) -> str:
        import re
        m = re.search(r'rename file\s+"?([^"]+?)"?\s+to\s+"?([^"]+)"?$', command, re.IGNORECASE)
        if not m:
            return "❓ Format: rename file old.txt to new.txt"
        src_name, dst_name = m.group(1).strip(), m.group(2).strip()
        for d in [Path.home()/"Desktop", Path.home()/"Documents", Path.home()/"Downloads"]:
            src = d / src_name
            if src.exists():
                try:
                    dst = src.parent / dst_name
                    src.rename(dst)
                    return f"✏️ Renamed: {src_name} → {dst_name}"
                except Exception as e:
                    return f"❌ {e}"
        return f"❌ File not found: {src_name}"

    def _move_file(self, command: str) -> str:
        import re, shutil
        m = re.search(r'move file\s+"?([^"]+?)"?\s+to\s+"?([^"]+)"?$', command, re.IGNORECASE)
        if not m:
            return "❓ Format: move file notes.txt to documents"
        src_name, dst_str = m.group(1).strip(), m.group(2).strip()
        loc_map = {"desktop": Path.home()/"Desktop", "documents": Path.home()/"Documents",
                   "downloads": Path.home()/"Downloads", "pictures": Path.home()/"Pictures",
                   "videos": Path.home()/"Videos", "music": Path.home()/"Music"}
        dst = loc_map.get(dst_str.lower(), Path(dst_str))
        for d in [Path.home()/"Desktop", Path.home()/"Documents", Path.home()/"Downloads"]:
            src = d / src_name
            if src.exists():
                try:
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst / src_name))
                    return f"📦 Moved: {src_name} → {dst}"
                except Exception as e:
                    return f"❌ {e}"
        return f"❌ File not found: {src_name}"

    def _copy_file(self, command: str) -> str:
        import re, shutil
        m = re.search(r'copy file\s+"?([^"]+?)"?\s+to\s+"?([^"]+)"?$', command, re.IGNORECASE)
        if not m:
            return "❓ Format: copy file notes.txt to downloads"
        src_name, dst_str = m.group(1).strip(), m.group(2).strip()
        loc_map = {"desktop": Path.home()/"Desktop", "documents": Path.home()/"Documents",
                   "downloads": Path.home()/"Downloads", "pictures": Path.home()/"Pictures",
                   "videos": Path.home()/"Videos", "music": Path.home()/"Music"}
        dst = loc_map.get(dst_str.lower(), Path(dst_str))
        for d in [Path.home()/"Desktop", Path.home()/"Documents", Path.home()/"Downloads"]:
            src = d / src_name
            if src.exists():
                try:
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dst / src_name))
                    return f"📋 Copied: {src_name} → {dst}"
                except Exception as e:
                    return f"❌ {e}"
        return f"❌ File not found: {src_name}"

    def _read_file(self, command: str) -> str:
        import re
        m = re.search(r'(?:read|show|cat) file\s+"?([^"]+)"?', command, re.IGNORECASE)
        if not m:
            return "❓ Format: read file notes.txt"
        filename = m.group(1).strip()
        for d in [Path.home()/"Desktop", Path.home()/"Documents", Path.home()/"Downloads", Path.home()]:
            p = d / filename
            if p.exists() and p.is_file():
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                    preview = text[:1000]
                    more = f"\n... ({len(text)-1000} more chars)" if len(text) > 1000 else ""
                    return f"📄 {p.name}:\n{'─'*40}\n{preview}{more}"
                except Exception as e:
                    return f"❌ Cannot read: {e}"
        return f"❌ File not found: {filename}"

    def _find_files(self, command: str) -> str:
        query = command.lower()
        for kw in ["find file","search file","locate file"]:
            query = query.replace(kw, "").strip()
        query = query.strip(": ").strip()
        if not query:
            return "❓ Specify: find file report.pdf"
        results = []
        for d in [Path.home(), Path.home()/"Desktop",
                  Path.home()/"Documents", Path.home()/"Downloads"]:
            if d.exists():
                try:
                    for m in d.rglob(f"*{query}*"):
                        results.append(str(m))
                        if len(results) >= 20:
                            break
                except (PermissionError, Exception):
                    pass
            if len(results) >= 20:
                break
        if results:
            return f"🔍 {len(results)} result(s):\n" + "\n".join(f"  📄 {r}" for r in results)
        return f"❌ No files found: '{query}'"

    def _list_folder(self, command: str) -> str:
        cmd = command.lower()
        folder = Path.home()
        for kw in ["desktop","documents","downloads","pictures","videos","music"]:
            if kw in cmd:
                folder = Path.home() / kw.capitalize()
                break
        if not folder.exists():
            return f"❌ Folder not found: {folder}"
        try:
            items = list(folder.iterdir())
        except Exception as e:
            return f"❌ {e}"
        files   = sorted([i for i in items if i.is_file()])
        folders = sorted([i for i in items if i.is_dir()])
        out = f"📁 {folder}\n  {len(folders)} folders, {len(files)} files\n"
        if folders:
            out += "📂 Folders:\n" + "".join(f"  • {f.name}\n" for f in folders[:15])
        if files:
            out += "📄 Files:\n"
            for f in files[:20]:
                try:
                    sz = f.stat().st_size
                    out += f"  • {f.name}  ({sz//1024}KB)\n"
                except Exception:
                    out += f"  • {f.name}\n"
        return out

    def _open_location(self, command: str) -> str:
        cmd = command.lower()
        for kw in ["desktop","documents","downloads","pictures","videos","music"]:
            if kw in cmd:
                p = Path.home() / kw.capitalize()
                subprocess.Popen(f'explorer "{p}"')
                return f"📁 Opened: {p}"
        subprocess.Popen(f'explorer "{Path.home()}"')
        return f"📁 Opened: {Path.home()}"

    # ── APPS ──────────────────────────────────────────────────────────────────

    def _list_installed_apps(self) -> str:
        apps = []
        reg_paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        ]
        for rp in reg_paths:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, rp)
                for i in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        sk = winreg.OpenKey(key, winreg.EnumKey(key, i))
                        try:
                            name = winreg.QueryValueEx(sk, "DisplayName")[0]
                            if name and name.strip():
                                apps.append(name.strip())
                        except FileNotFoundError:
                            pass
                        finally:
                            winreg.CloseKey(sk)
                    except Exception:
                        pass
                winreg.CloseKey(key)
            except Exception:
                pass
        apps = sorted(set(apps))
        out = f"📋 {len(apps)} installed apps:\n"
        for a in apps[:50]:
            out += f"  • {a}\n"
        if len(apps) > 50:
            out += f"  ... and {len(apps)-50} more\n"
        return out

    def _open_app(self, command: str) -> str:
        cmd = command.lower()
        for keyword, exe in self.APP_MAP.items():
            if keyword in cmd:
                if exe.startswith("ms-"):
                    subprocess.Popen(f"start {exe}", shell=True)
                    return f"✅ Opened: {keyword.title()}"
                try:
                    r = subprocess.run(f'where "{exe}"', shell=True,
                                       capture_output=True, text=True, timeout=5)
                    if r.returncode == 0 and r.stdout.strip():
                        first = r.stdout.strip().splitlines()[0]
                        subprocess.Popen(f'"{first}"', shell=True)
                        return f"✅ Opened: {keyword.title()}"
                except Exception:
                    pass
                try:
                    subprocess.run(
                        ["powershell", "-NoProfile", "-Command", f"Start-Process '{exe}'"],
                        capture_output=True, text=True, timeout=8
                    )
                    return f"✅ Opened: {keyword.title()}"
                except Exception as e:
                    return f"❌ Could not open '{keyword}': {e}"

        for trigger in ["open ","launch ","start ","run "]:
            if cmd.startswith(trigger):
                app_name = command[len(trigger):].strip().strip('"\'')
                try:
                    subprocess.run(
                        ["powershell", "-NoProfile", "-Command", f"Start-Process '{app_name}'"],
                        capture_output=True, text=True, timeout=8
                    )
                    return f"✅ Opened: {app_name}"
                except Exception as e:
                    return f"❌ {e}"
        return "❓ App not recognized"

    def _close_app(self, command: str) -> str:
        cmd = command.lower()
        for kw, exe in self.APP_MAP.items():
            if kw in cmd:
                proc_name = exe.replace(".exe", "").lower()
                killed = 0
                for p in psutil.process_iter(['name']):
                    try:
                        name = (p.info.get('name') or "").lower()
                        if proc_name and proc_name in name:
                            p.kill()
                            killed += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                        pass
                return (f"✅ Closed {killed} instance(s) of {kw}"
                        if killed else f"⚠️ {kw} not running")
        return "❓ Specify app: close chrome"

    # ── SYSTEM STATS ──────────────────────────────────────────────────────────

    def _system_stats(self) -> str:
        cpu  = psutil.cpu_percent(interval=1)
        ram  = psutil.virtual_memory()
        freq = psutil.cpu_freq()
        freq_str = f"{freq.current:.0f}MHz" if freq else "N/A"
        out  = "📊 SYSTEM STATS\n"
        out += f"  🖥️  CPU: {cpu}% @ {freq_str} ({psutil.cpu_count()} cores)\n"
        out += f"  🧠 RAM: {ram.used//(1024**3)}GB / {ram.total//(1024**3)}GB ({ram.percent}%)\n"
        return out

    def _disk_stats(self) -> str:
        out = "💾 DISK STORAGE\n"
        for part in psutil.disk_partitions():
            try:
                u   = psutil.disk_usage(part.mountpoint)
                pct = u.percent
                bar = "█"*int(pct/10) + "░"*(10-int(pct/10))
                out += (f"\n  {part.device}\n"
                        f"  [{bar}] {pct}%\n"
                        f"  Total:{u.total//(1024**3)}GB  "
                        f"Used:{u.used//(1024**3)}GB  "
                        f"Free:{u.free//(1024**3)}GB\n")
            except (PermissionError, Exception):
                pass
        return out

    def _battery_info(self) -> str:
        b = psutil.sensors_battery()
        if not b:
            return "🔌 No battery (desktop)"
        status = "Charging 🔌" if b.power_plugged else "On Battery 🔋"
        if b.secsleft and b.secsleft > 0 and b.secsleft != psutil.POWER_TIME_UNLIMITED:
            tl = f"{b.secsleft//3600}h {(b.secsleft%3600)//60}m"
        else:
            tl = "Calculating..."
        return f"🔋 {b.percent:.1f}%  {status}  Time left: {tl}"

    def _full_system_info(self) -> str:
        import platform
        out  = "🖥️ SYSTEM INFO\n"
        out += f"  OS:        {platform.system()} {platform.release()}\n"
        out += f"  CPU:       {platform.processor()[:50]}\n"
        out += f"  Cores:     {psutil.cpu_count()} ({psutil.cpu_count(logical=False)} physical)\n"
        out += f"  RAM:       {psutil.virtual_memory().total//(1024**3)} GB\n"
        out += f"  Python:    {platform.python_version()}\n"
        out += f"  Hostname:  {platform.node()}\n"
        boot = datetime.fromtimestamp(psutil.boot_time())
        out += f"  Boot time: {boot:%Y-%m-%d %H:%M}\n"
        return out

    def _network_info(self) -> str:
        import socket
        out = "🌐 NETWORK\n"
        try:
            out += f"  Hostname: {socket.gethostname()}\n"
            out += f"  Local IP: {socket.gethostbyname(socket.gethostname())}\n"
        except Exception:
            pass
        st   = psutil.net_io_counters()
        out += f"  Sent:     {st.bytes_sent//(1024**2)} MB\n"
        out += f"  Received: {st.bytes_recv//(1024**2)} MB\n"
        for iface, addrs in list(psutil.net_if_addrs().items())[:5]:
            for addr in addrs:
                if addr.family == 2:  # AF_INET
                    out += f"  • {iface}: {addr.address}\n"
        return out

    def _top_processes(self) -> str:
        procs = []
        for p in psutil.process_iter(['name','pid','cpu_percent','memory_info']):
            try:
                info = p.info
                mem_mb = (info['memory_info'].rss if info['memory_info'] else 0)//(1024**2)
                procs.append((info['name'] or "?", info['pid'],
                               info['cpu_percent'] or 0, mem_mb))
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                pass
        procs.sort(key=lambda x: x[3], reverse=True)
        out = f"⚙️ TOP PROCESSES (by RAM)\n  {'Name':<28}{'PID':<8}{'CPU%':<8}RAM\n  " + "-"*48 + "\n"
        for name, pid, cpu, mem in procs[:15]:
            out += f"  {name:<28}{pid:<8}{cpu:<8.1f}{mem} MB\n"
        return out

    def _running_processes(self) -> str:
        return self._top_processes()

    def _kill_process(self, command: str) -> str:
        cmd = command.lower()
        for trigger in ["kill process","end process","stop process"]:
            if trigger in cmd:
                target = cmd.split(trigger, 1)[1].strip()
                if not target:
                    return "❓ Specify: kill process chrome"
                killed = 0
                for proc in psutil.process_iter(['name','pid']):
                    try:
                        name = (proc.info.get('name') or "").lower()
                        if target in name:
                            proc.kill()
                            killed += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                        pass
                return (f"✅ Killed {killed} process(es): '{target}'"
                        if killed else f"⚠️ Not found: '{target}'")
        return "❓ Specify: kill process chrome"

    def _list_startup(self) -> str:
        reg_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
        out = "🚀 STARTUP PROGRAMS\n"
        for hive in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
            try:
                key = winreg.OpenKey(hive, reg_path)
                for i in range(winreg.QueryInfoKey(key)[1]):
                    try:
                        name, val, _ = winreg.EnumValue(key, i)
                        out += f"  • {name}: {str(val)[:60]}\n"
                    except Exception:
                        pass
                winreg.CloseKey(key)
            except Exception:
                pass
        return out

    def _env_vars(self) -> str:
        out = "🌍 ENVIRONMENT VARIABLES (first 30)\n"
        for i, (k, v) in enumerate(os.environ.items()):
            if i >= 30:
                break
            out += f"  {k} = {v[:60]}\n"
        return out

    # ── UI CONTROLS ───────────────────────────────────────────────────────────

    def _type_text(self, command: str) -> str:
        text = command[5:].strip()
        if not text:
            return "❓ Format: type Hello World"
        pyautogui.typewrite(text, interval=0.03)
        return f"⌨️ Typed: {text}"

    def _send_hotkey(self, command: str) -> str:
        cmd = command.lower()
        for trigger in ["hotkey ","press "]:
            if cmd.startswith(trigger):
                keys = [k.strip() for k in cmd[len(trigger):].strip().split("+") if k.strip()]
                if not keys:
                    return "❓ Format: hotkey ctrl+c"
                try:
                    pyautogui.hotkey(*keys)
                    return f"⌨️ Hotkey: {'+'.join(keys)}"
                except Exception as e:
                    return f"❌ {e}"
        return "❓ Format: hotkey ctrl+c"

    def _take_screenshot(self) -> str:
        img  = pyautogui.screenshot()
        path = Path("screenshots") / f"ss_{datetime.now():%Y%m%d_%H%M%S}.png"
        path.parent.mkdir(exist_ok=True)
        img.save(path)
        return f"📸 Screenshot saved: {path}"

    def take_screenshot_bytes(self) -> bytes:
        img = pyautogui.screenshot()
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ── WEB / VOLUME ──────────────────────────────────────────────────────────

    def _open_url(self, command: str) -> str:
        cmd = command.lower()
        for t in ["go to","open website","browse","navigate to"]:
            if t in cmd:
                url = cmd.split(t, 1)[1].strip()
                if not url:
                    return "❓ Specify URL"
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                webbrowser.open(url)
                return f"🌐 Opened: {url}"
        return "❓ Specify URL"

    def _search_web(self, command: str) -> str:
        cmd = command.lower()
        for t in ["search for","google ","look up"]:
            if t in cmd:
                q = cmd.split(t, 1)[1].strip()
                if not q:
                    return "❓ Specify query"
                webbrowser.open(f"https://www.google.com/search?q={requests.utils.quote(q)}")
                return f"🔎 Searching: {q}"
        return "❓ Specify query"

    def _control_volume(self, command: str) -> str:
        cmd = command.lower()
        if "mute" in cmd:
            pyautogui.press("volumemute")
            return "🔇 Muted"
        elif "up" in cmd or "increase" in cmd:
            for _ in range(5):
                pyautogui.press("volumeup")
            return "🔊 Volume up"
        elif "down" in cmd or "decrease" in cmd or "lower" in cmd:
            for _ in range(5):
                pyautogui.press("volumedown")
            return "🔉 Volume down"
        return "❓ volume up / down / mute"

    def _control_brightness(self, command: str) -> str:
        cmd = command.lower()
        try:
            import screen_brightness_control as sbc
        except ImportError:
            return "⚠️ Install screen-brightness-control: pip install screen-brightness-control"
        try:
            if "up" in cmd or "increase" in cmd:
                cur = sbc.get_brightness(display=0)[0]
                new = min(100, cur + 10)
                sbc.set_brightness(new, display=0)
                return f"☀️ Brightness: {new}%"
            elif "down" in cmd or "decrease" in cmd or "lower" in cmd:
                cur = sbc.get_brightness(display=0)[0]
                new = max(0, cur - 10)
                sbc.set_brightness(new, display=0)
                return f"🌑 Brightness: {new}%"
            import re
            m = re.search(r'(\d+)%?', cmd)
            if m:
                val = max(0, min(100, int(m.group(1))))
                sbc.set_brightness(val, display=0)
                return f"☀️ Brightness: {val}%"
        except Exception as e:
            return f"❌ {e}"
        return "❓ brightness up / down / 50%"

    # ── POWER ─────────────────────────────────────────────────────────────────

    def _shutdown_pc(self) -> str:
        subprocess.Popen("shutdown /s /t 30", shell=True)
        return "⚠️ Shutting down in 30s.\n  Cancel: 'cancel shutdown' or run: shutdown /a"

    def _restart_pc(self) -> str:
        subprocess.Popen("shutdown /r /t 30", shell=True)
        return "🔄 Restarting in 30s.\n  Cancel: 'cancel shutdown'"

    def _sleep_pc(self) -> str:
        subprocess.Popen("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True)
        return "💤 Going to sleep..."

    def _lock_pc(self) -> str:
        subprocess.Popen("rundll32.exe user32.dll,LockWorkStation", shell=True)
        return "🔒 PC locked"


# ============= IMAGE GENERATOR =============
class AgentImageGenerator:
    def __init__(self, persona: AgentPersona):
        self.persona = persona

    def generate_image(self, prompt: str, style: str = "") -> Optional[bytes]:
        try:
            ps    = self.persona.get_current_persona()['image_style']
            full  = f"{prompt}, {style or ps}, high quality, detailed"
            url   = f"https://image.pollinations.ai/prompt/{requests.utils.quote(full)}?width=1024&height=1024&model=flux&nologo=true"
            r     = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 5000:
                head = r.content[:8]
                if head[:3] == b'\xff\xd8\xff' or head[:8] == b'\x89PNG\r\n\x1a\n':
                    return r.content
        except Exception as e:
            print(f"Image error: {e}")
        return None

    def save_image(self, data: bytes, prompt: str) -> Path:
        p = Path("agent_images")
        p.mkdir(exist_ok=True)
        safe = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ','-','_')).rstrip()
        path = p / f"img_{safe}_{datetime.now():%Y%m%d_%H%M%S}.png"
        path.write_bytes(data)
        return path


# ============= VOICE TRANSCRIBER =============
class VoiceTranscriber:
    def __init__(self, log_cb=None):
        self.log_cb = log_cb or print
        self.model  = None

    def load(self):
        self.log_cb("🎙️ Loading Whisper model...")
        self.model = whisper.load_model("base")
        self.log_cb("✅ Whisper ready!")

    def transcribe(self, audio_bytes: bytes, suffix=".ogg") -> str:
        if not self.model:
            self.load()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            result = self.model.transcribe(tmp_path)
            text   = result["text"].strip()
            self.log_cb(f"🎙️ Transcribed: {text}")
            return text
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ============= TELEGRAM BOT =============
class TelegramBotManager:
    def __init__(self, token, allowed_id, llm_mgr, persona, task_mgr,
                 pc_agent, voice, memory, plugin_mgr, log_cb=None):
        self.token      = token
        self.allowed_id = allowed_id
        self.llm_mgr    = llm_mgr
        self.persona    = persona
        self.task_mgr   = task_mgr
        self.pc_agent   = pc_agent
        self.voice      = voice
        self.memory     = memory
        self.plugin_mgr = plugin_mgr
        self.log        = log_cb or print
        self.app        = None
        self._thread    = None
        self.running    = False
        self._loop      = None
        self._last_chat_id = None

    def _allowed(self, update: Update) -> bool:
        return self.allowed_id == 0 or update.effective_user.id == self.allowed_id

    # ── Commands ─────────────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx):
        if not self._allowed(update):
            return
        self._last_chat_id = update.effective_chat.id
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("⚡ System Info", callback_data="cmd:system info"),
             InlineKeyboardButton("💾 Disk Info", callback_data="cmd:disk info")],
            [InlineKeyboardButton("⚙️ Processes", callback_data="cmd:running"),
             InlineKeyboardButton("📸 Screenshot", callback_data="cmd:screenshot")],
            [InlineKeyboardButton("📋 Plugins", callback_data="cmd:plugins"),
             InlineKeyboardButton("🔋 Battery", callback_data="cmd:battery info")],
        ])
        await update.message.reply_text(
            "👋 AI Agent XP v2 online!\n\nUse the buttons or type any command.",
            reply_markup=kb
        )

    async def _cmd_screenshot(self, update: Update, ctx):
        if not self._allowed(update):
            return
        self._last_chat_id = update.effective_chat.id
        await update.message.reply_text("📸 Taking screenshot...")
        data = self.pc_agent.take_screenshot_bytes()
        await update.message.reply_photo(photo=data, caption="📸 Current screen")

    async def _cmd_models(self, update: Update, ctx):
        if not self._allowed(update):
            return
        self._last_chat_id = update.effective_chat.id
        models = self.llm_mgr.available_models()
        current = self.llm_mgr.current_key
        lines = ["🤖 Available models (free):\n"]
        kb_rows = []
        for m in models:
            mark = " ✅" if m == current else ""
            lines.append(f"  • {m}{mark}")
            kb_rows.append([InlineKeyboardButton(
                f"{'✅ ' if m == current else ''}{m}",
                callback_data=f"model:{m}"
            )])
        await update.message.reply_text(
            "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(kb_rows)
        )

    async def _cmd_plugins(self, update: Update, ctx):
        if not self._allowed(update):
            return
        self._last_chat_id = update.effective_chat.id
        await update.message.reply_text(self.plugin_mgr.list_plugins())

    async def _cmd_memory(self, update: Update, ctx):
        if not self._allowed(update):
            return
        self._last_chat_id = update.effective_chat.id
        ctx_text = self.memory.get_context()
        if len(ctx_text) > 3000:
            ctx_text = ctx_text[-3000:]
        await update.message.reply_text(f"🧠 Memory:\n{ctx_text or '(empty)'}")

    async def _cmd_clear(self, update: Update, ctx):
        if not self._allowed(update):
            return
        self.memory.clear()
        await update.message.reply_text("🧹 Memory cleared.")

    async def _callback(self, update: Update, ctx):
        q = update.callback_query
        data = q.data or ""
        await q.answer()
        self._last_chat_id = update.effective_chat.id

        if data.startswith("model:"):
            model = data[6:]
            if self.llm_mgr.switch_model(model):
                await q.edit_message_text(f"✅ Switched to: {model}")
            return

        if data.startswith("rem_dismiss:"):
            rid = data.split(":", 1)[1]
            await q.edit_message_text(f"✅ Reminder #{rid} dismissed.")
            return

        if data.startswith("rem_snooze:"):
            parts = data.split(":", 2)
            rid  = parts[1] if len(parts) > 1 else "?"
            task = parts[2] if len(parts) > 2 else "reminder"
            loop = self._loop
            chat_id = update.effective_chat.id

            def _resnooze():
                time.sleep(300)
                if loop and self.app:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.app.bot.send_message(
                                chat_id,
                                f"📞 *SNOOZED REMINDER*\n━━━━━━━━━━━━━━━━━━━━\n🔔  *{task}*\n━━━━━━━━━━━━━━━━━━━━",
                                parse_mode="Markdown"
                            ),
                            loop
                        )
                    except Exception:
                        pass

            threading.Thread(target=_resnooze, daemon=True).start()
            await q.edit_message_text(f"⏰ Snoozed 5 min: {task}")
            return

        if data.startswith("cmd:"):
            cmd = data[4:]
            if cmd == "screenshot":
                await q.message.reply_text("📸 Taking screenshot...")
                img = self.pc_agent.take_screenshot_bytes()
                await q.message.reply_photo(photo=img, caption="📸 Current screen")
                return
            result = self.plugin_mgr.handle(cmd) or self.pc_agent.handle(cmd) or "❓ Unknown"
            await q.message.reply_text(result[:4000])

    # ── Text / voice / document handlers ─────────────────────────────────────

    async def _handle_text(self, update: Update, ctx):
        if not self._allowed(update):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        self._last_chat_id = update.effective_chat.id
        text = (update.message.text or "").strip()
        if not text:
            return
        self.log(f"📱 TG: {text}")

        # Try plugin (sync)
        plugin_result = self.plugin_mgr.handle(text)
        if plugin_result is not None:
            await update.message.reply_text(plugin_result[:4000])
            return

        # Try PC command (sync)
        pc_result = self.pc_agent.handle(text)
        if pc_result is not None and pc_result != "→ plugin_list":
            await update.message.reply_text(pc_result[:4000])
            return
        if pc_result == "→ plugin_list":
            await update.message.reply_text(self.plugin_mgr.list_plugins()[:4000])
            return

        # AI chat — run LLM in a thread so we don't block the event loop
        await update.message.reply_text("💭 Thinking...")
        try:
            p = self.persona.get_current_persona()
            system = self.persona.get_persona_prompt()
            ctx_text = self.memory.get_context()
            messages = [
                {"role": "system", "content": system + f"\n\nContext:\n{ctx_text}"},
                {"role": "user", "content": text},
            ]
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(None, self.llm_mgr.chat, messages)
            self.memory.add("user", text)
            self.memory.add(p["name"].lower(), answer)
            await update.message.reply_text(f"{p['emoji']} {answer[:4000]}")
        except Exception as e:
            await update.message.reply_text(f"❌ AI Error: {e}")

    async def _handle_voice(self, update: Update, ctx):
        if not self._allowed(update):
            return
        self._last_chat_id = update.effective_chat.id
        await update.message.reply_text("🎙️ Transcribing...")

        try:
            vf = await update.message.voice.get_file()
            audio = await vf.download_as_bytearray()
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, self.voice.transcribe, bytes(audio), ".ogg"
            )
            await update.message.reply_text(f'🎙️ Heard: "{text}"\n⚡ Processing...')

            plugin_result = self.plugin_mgr.handle(text)
            if plugin_result is not None:
                await update.message.reply_text(plugin_result[:4000])
                return

            pc_result = self.pc_agent.handle(text)
            if pc_result is not None and pc_result != "→ plugin_list":
                await update.message.reply_text(pc_result[:4000])
                return

            p = self.persona.get_current_persona()
            ctx_text = self.memory.get_context()
            messages = [
                {"role": "system", "content": self.persona.get_persona_prompt() + f"\nContext:\n{ctx_text}"},
                {"role": "user", "content": text},
            ]
            answer = await loop.run_in_executor(None, self.llm_mgr.chat, messages)
            self.memory.add("user", text)
            self.memory.add(p["name"].lower(), answer)
            await update.message.reply_text(f"{p['emoji']} {answer[:4000]}")
        except Exception as e:
            try:
                await update.message.reply_text(f"❌ {e}")
            except Exception:
                pass

    async def _handle_document(self, update: Update, ctx):
        if not self._allowed(update):
            return
        self._last_chat_id = update.effective_chat.id
        await update.message.reply_text("📎 Received file — downloading...")
        try:
            doc = update.message.document
            if doc:
                f = await ctx.bot.get_file(doc.file_id)
                data = await f.download_as_bytearray()
                save_dir = Path.home() / "Downloads"
                save_dir.mkdir(parents=True, exist_ok=True)
                save = save_dir / doc.file_name
                save.write_bytes(bytes(data))
                await update.message.reply_text(f"✅ Saved to Downloads: {doc.file_name}")
        except Exception as e:
            await update.message.reply_text(f"❌ {e}")

    def send_message(self, text: str):
        if not (self.app and self.running and self._loop and self._last_chat_id):
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.app.bot.send_message(self._last_chat_id, text),
                self._loop
            )
        except Exception:
            pass

    def send_file(self, file_path: str, caption: str = ""):
        if not (self.app and self.running and self._loop and self._last_chat_id):
            return
        chat_id = self._last_chat_id

        async def _send():
            with open(file_path, "rb") as f:
                await self.app.bot.send_document(chat_id, document=f, caption=caption)

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._loop)
        except Exception:
            pass

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if not self.token:
            self.log("⚠️ No TELEGRAM_TOKEN")
            return

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            self.app = ApplicationBuilder().token(self.token).build()
            self.app.add_handler(CommandHandler("start",      self._cmd_start))
            self.app.add_handler(CommandHandler("screenshot", self._cmd_screenshot))
            self.app.add_handler(CommandHandler("models",     self._cmd_models))
            self.app.add_handler(CommandHandler("plugins",    self._cmd_plugins))
            self.app.add_handler(CommandHandler("memory",     self._cmd_memory))
            self.app.add_handler(CommandHandler("clear",      self._cmd_clear))
            self.app.add_handler(CallbackQueryHandler(self._callback))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
            self.app.add_handler(MessageHandler(filters.VOICE,    self._handle_voice))
            self.app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
            self.log("📱 Telegram bot started!")
            self.running = True
            try:
                self.app.run_polling(stop_signals=None, close_loop=False)
            except Exception as e:
                self.log(f"❌ Telegram polling error: {e}")
            finally:
                self.running = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        if self.app and self._loop:
            try:
                asyncio.run_coroutine_threadsafe(self.app.stop(), self._loop)
            except Exception:
                pass
        self.running = False


# ============= XP BUTTON HELPER =============
def xp_button(parent, text, command, width=12, bg=None):
    return tk.Button(
        parent, text=text, command=command,
        font=(XP["font"], XP["font_size"]),
        bg=bg or XP["btn"], fg="#000000",
        relief="raised", bd=2,
        activebackground=XP["highlight"],
        activeforeground=XP["highlight_fg"],
        cursor="hand2", width=width,
    )


# ============= MAIN XP APPLICATION v2 =============
class AgentXPApp:
    def __init__(self, root: tk.Tk):
        self.root         = root
        self.root.title("AI Agent XP v2 — Multi-Model + Plugins + Streaming")
        self.root.configure(bg=XP["bg"])
        self.root.geometry("960x680")
        self.root.minsize(760, 520)

        style = ttk.Style()
        try:
            style.theme_use("classic")
        except Exception:
            pass

        # Core state
        self.persona      = AgentPersona()
        self.task_manager = TaskManager()
        self.memory       = PersistentMemory()
        self.llm_mgr      = LLMManager(
            log_cb=lambda m: self.root.after(0, lambda msg=m: self._system_msg(msg))
        )
        self.image_gen    = AgentImageGenerator(self.persona)
        self.is_thinking  = False
        self._stream_has_token = False

        self.pc_agent   = LocalPCAgent(
            log_cb=lambda m: self.root.after(0, lambda msg=m: self._system_msg(msg, "pc"))
        )
        self.voice      = VoiceTranscriber(
            log_cb=lambda m: self.root.after(0, lambda msg=m: self._system_msg(msg))
        )
        self.plugin_mgr = PluginManager(
            PLUGINS_DIR, self,
            log_cb=lambda m: self.root.after(0, lambda msg=m: self._system_msg(msg, "plugin"))
        )
        self.tg_bot     = None

        self._build_ui()
        self._init_agent()

    # ─── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = tk.Frame(self.root, bg="#808080", bd=2, relief="raised")
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        # Title bar
        tb = tk.Frame(outer, height=30, bg=XP["title_bar"])
        tb.pack(fill="x")
        tb.pack_propagate(False)
        tk.Label(tb, text="  🖥️  AI Agent XP v2  —  Multi-Model · Plugins · Streaming",
                 font=(XP["font"], 9, "bold"),
                 bg=XP["title_bar"], fg=XP["title_fg"]).pack(side="left", pady=4)
        for txt, cmd, color in [
            ("_", lambda: self.root.iconify(), "#5A8FE0"),
            ("✕", self.root.destroy,           "#C0302C"),
        ]:
            tk.Button(tb, text=txt, font=("Tahoma", 8, "bold"),
                      bg=color, fg="white", relief="raised", bd=1,
                      width=3, cursor="hand2", command=cmd).pack(side="right", pady=3, padx=1)

        self._build_menu(outer)

        main = tk.Frame(outer, bg=XP["bg"])
        main.pack(fill="both", expand=True)

        self._build_left_panel(main)
        self._build_chat_panel(main)
        self._build_taskbar(outer)

    def _build_menu(self, parent):
        bar = tk.Frame(parent, bg=XP["bg_dark"], relief="raised", bd=1)
        bar.pack(fill="x")

        menus = [
            ("File", [
                ("📂 Open File...",    self._menu_open_file),
                ("💾 Save Chat Log",  self._save_chat_log),
                ("─────────",         None),
                ("Exit",              self.root.destroy),
            ]),
            ("View", [
                ("Clear Chat",        self._clear_chat),
                ("Status",            self._show_status),
                ("Memory Browser",    self._show_memory_browser),
                ("Task List",         lambda: self._system_msg(self.task_manager.list_tasks())),
            ]),
            ("AI Model", [(m, lambda k=m: self._switch_model(k))
                           for m in FREE_MODELS.keys()]),
            ("Personas", [(f"{v['emoji']} {v['name']}", lambda k=k: self._switch_persona(k))
                           for k, v in AgentPersona.PERSONAS.items()]),
            ("PC", [
                ("System Info",       lambda: self._local_cmd("system info")),
                ("Disk Info",         lambda: self._local_cmd("disk info")),
                ("Running Processes", lambda: self._local_cmd("running")),
                ("Installed Apps",    lambda: self._local_cmd("list apps")),
                ("Network Info",      lambda: self._local_cmd("network info")),
                ("Battery Info",      lambda: self._local_cmd("battery info")),
                ("Startup Programs",  lambda: self._local_cmd("startup apps")),
                ("Environment Vars",  lambda: self._local_cmd("environment variables")),
                ("Take Screenshot",   lambda: self._local_cmd("screenshot")),
                ("Lock PC",           lambda: self._local_cmd("lock pc")),
                ("Shut Down PC",      lambda: self._local_cmd("shutdown")),
                ("Restart PC",        lambda: self._local_cmd("restart")),
                ("Sleep",             lambda: self._local_cmd("sleep")),
            ]),
            ("Plugins", [
                ("List Plugins",      lambda: self._system_msg(self.plugin_mgr.list_plugins(), "plugin")),
                ("Reload All",        self._reload_plugins),
                ("Open Plugin Folder",lambda: subprocess.Popen(f'explorer "{PLUGINS_DIR.resolve()}"')),
            ]),
            ("Telegram", [
                ("Start Bot",         self._start_telegram),
                ("Stop Bot",          self._stop_telegram),
                ("Bot Status",        self._telegram_status),
                ("Send File to TG",   self._send_file_to_telegram),
            ]),
            ("Memory", [
                ("Show Context",      lambda: self._system_msg(self.memory.get_context()[-2000:])),
                ("Clear Memory",      lambda: (self.memory.clear(), self._system_msg("🧹 Memory cleared"))),
                ("Search Memory",     self._search_memory_dialog),
            ]),
            ("Help", [("Commands", self._show_help)]),
        ]

        for label, items in menus:
            mb   = tk.Menubutton(bar, text=label, font=(XP["font"], 9),
                                  bg=XP["bg_dark"], fg="#000000", relief="flat",
                                  padx=8, pady=2,
                                  activebackground=XP["highlight"],
                                  activeforeground="white")
            mb.pack(side="left")
            menu = tk.Menu(mb, tearoff=False, font=(XP["font"], 9),
                           bg=XP["bg"], fg="#000000",
                           activebackground=XP["highlight"],
                           activeforeground="white")
            for item_label, item_cmd in items:
                if item_cmd is None:
                    menu.add_separator()
                else:
                    menu.add_command(label=item_label, command=item_cmd)
            mb["menu"] = menu

    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=XP["bg"], width=195)
        left.pack(side="left", fill="y", padx=(6,3), pady=6)
        left.pack_propagate(False)

        # Persona panel
        pf = tk.LabelFrame(left, text=" 🤖 Persona ", font=(XP["font"],9,"bold"),
                            bg=XP["bg"], fg="#000", relief="groove", bd=2)
        pf.pack(fill="x", pady=(0,4))
        self.persona_label = tk.Label(pf, text="", font=(XP["font"],10,"bold"),
                                       bg=XP["bg"], fg=XP["user_fg"], wraplength=165, justify="center")
        self.persona_label.pack(pady=3)
        self.persona_desc  = tk.Label(pf, text="", font=(XP["font"],8),
                                       bg=XP["bg"], fg="#555", wraplength=165, justify="center")
        self.persona_desc.pack(pady=(0,3))

        # Model panel
        mf = tk.LabelFrame(left, text=" 🤖 AI Model ", font=(XP["font"],9,"bold"),
                            bg=XP["bg"], fg="#000", relief="groove", bd=2)
        mf.pack(fill="x", pady=(0,4))
        self.model_var = tk.StringVar(value=self.llm_mgr.current_key)
        self.model_menu = ttk.Combobox(mf, textvariable=self.model_var,
                                        values=self.llm_mgr.available_models(),
                                        state="readonly", font=(XP["font"],7), width=22)
        self.model_menu.pack(padx=4, pady=4)
        self.model_menu.bind("<<ComboboxSelected>>",
                             lambda e: self._switch_model(self.model_var.get()))

        # Persona switcher
        sf = tk.LabelFrame(left, text=" Switch Persona ", font=(XP["font"],9,"bold"),
                            bg=XP["bg"], fg="#000", relief="groove", bd=2)
        sf.pack(fill="x", pady=(0,4))
        for key, val in AgentPersona.PERSONAS.items():
            tk.Button(sf, text=f"{val['emoji']} {val['name']}", font=(XP["font"],8),
                      bg=XP["btn"], fg="#000", relief="raised", bd=2, anchor="w",
                      activebackground=XP["highlight"], activeforeground="white",
                      cursor="hand2", command=lambda k=key: self._switch_persona(k)
                      ).pack(fill="x", padx=4, pady=1)

        # Quick PC
        qf = tk.LabelFrame(left, text=" ⚡ Quick PC ", font=(XP["font"],9,"bold"),
                            bg=XP["bg"], fg="#000", relief="groove", bd=2)
        qf.pack(fill="x", pady=(0,4))
        for label, cmd in [
            ("📊 System Info",    "system info"),
            ("💾 Disk Info",      "disk info"),
            ("⚙️ Processes",     "running"),
            ("📋 Apps",          "list apps"),
            ("🚀 Startup",       "startup apps"),
            ("📸 Screenshot",    "screenshot"),
            ("🔒 Lock PC",       "lock pc"),
            ("⚠️ Shut Down",     "shutdown"),
        ]:
            tk.Button(qf, text=label, font=(XP["font"],8),
                      bg=XP["btn"], fg="#000", relief="raised", bd=2, anchor="w",
                      activebackground=XP["highlight"], activeforeground="white",
                      cursor="hand2", command=lambda c=cmd: self._local_cmd(c)
                      ).pack(fill="x", padx=4, pady=1)

        # Telegram
        tgf = tk.LabelFrame(left, text=" 📱 Telegram ", font=(XP["font"],9,"bold"),
                             bg=XP["bg"], fg="#000", relief="groove", bd=2)
        tgf.pack(fill="x", pady=(0,4))
        self.tg_status_label = tk.Label(tgf, text="⬤ Offline", font=(XP["font"],9,"bold"),
                                         bg=XP["bg"], fg="#CC0000")
        self.tg_status_label.pack(pady=3)
        xp_button(tgf, "▶ Start Bot", self._start_telegram, width=16, bg="#4CAF50").pack(padx=4, pady=2)
        xp_button(tgf, "■ Stop Bot",  self._stop_telegram,  width=16, bg="#D32F2F").pack(padx=4, pady=(2,4))
        xp_button(tgf, "📤 Send File", self._send_file_to_telegram, width=16).pack(padx=4, pady=(0,4))

        # Image Gen
        igf = tk.LabelFrame(left, text=" 🎨 Image Gen ", font=(XP["font"],9,"bold"),
                             bg=XP["bg"], fg="#000", relief="groove", bd=2)
        igf.pack(fill="x")
        self.img_entry = tk.Entry(igf, font=(XP["font"],8), relief="sunken", bd=2)
        self.img_entry.pack(fill="x", padx=4, pady=(4,2))
        self.img_entry.insert(0, "describe an image...")
        self.img_entry.bind("<FocusIn>",
            lambda e: self.img_entry.delete(0, "end")
                      if self.img_entry.get() == "describe an image..." else None)
        self.img_entry.bind("<Return>", lambda e: self._generate_image())
        xp_button(igf, "🎨 Generate", self._generate_image, width=16).pack(padx=4, pady=(2,4))

        self._update_persona_panel()

    def _build_chat_panel(self, parent):
        right = tk.Frame(parent, bg=XP["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(3,6), pady=6)

        cf = tk.LabelFrame(right, text=" 💬 Conversation ", font=(XP["font"],9,"bold"),
                            bg=XP["bg"], fg="#000", relief="groove", bd=2)
        cf.pack(fill="both", expand=True)

        self.chat_area = tk.Text(cf, state="disabled", wrap="word",
                                  bg=XP["chat_bg"], fg="#000",
                                  font=(XP["font"],9), relief="sunken",
                                  bd=2, padx=6, pady=6, cursor="arrow")
        sb = ttk.Scrollbar(cf, command=self.chat_area.yview)
        self.chat_area.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.chat_area.pack(fill="both", expand=True, padx=4, pady=4)

        for tag, fg, style in [
            ("user",     XP["user_fg"],   "bold"),
            ("bot",      XP["bot_fg"],    "normal"),
            ("telegram", XP["tg_fg"],     "bold"),
            ("system",   XP["system_fg"], "italic"),
            ("error",    XP["error_fg"],  "bold"),
            ("thinking", "#B8860B",       "italic"),
            ("pc",       "#8B4513",       "normal"),
            ("plugin",   XP["plugin_fg"], "bold"),
            ("stream",   XP["bot_fg"],    "normal"),
        ]:
            self.chat_area.tag_configure(
                tag, foreground=fg,
                font=(XP["font"],9,style)
            )

        # Input bar
        inf = tk.Frame(right, bg=XP["bg"])
        inf.pack(fill="x", pady=(4,0))

        self.input_entry = tk.Entry(inf, font=(XP["font"],10), relief="sunken", bd=2,
                                     bg=XP["chat_bg"])
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0,4))
        self.input_entry.bind("<Return>", lambda e: self._send_message())

        xp_button(inf, "Send ➤", self._send_message, width=10,
                  bg=XP["highlight"]).pack(side="right")

    def _build_taskbar(self, parent):
        tb = tk.Frame(parent, bg=XP["taskbar_bg"], height=30)
        tb.pack(fill="x", side="bottom")
        tb.pack_propagate(False)
        tk.Label(tb, text="  🪟 start ", font=(XP["font"],9,"bold"),
                 bg="#4CAF50", fg="white", relief="raised", bd=2).pack(side="left", padx=4, pady=3)
        tk.Frame(tb, bg=XP["taskbar_btn"], width=2).pack(side="left", fill="y", pady=3)
        tk.Label(tb, text="  🖥️ AI Agent XP v2  ",
                 font=(XP["font"],9), bg=XP["taskbar_btn"],
                 fg="white", relief="raised", bd=2).pack(side="left", padx=4, pady=3)
        self.clock_label = tk.Label(tb, text="", font=(XP["font"],9),
                                     bg=XP["taskbar_bg"], fg="white")
        self.clock_label.pack(side="right", padx=8)
        self._update_clock()
        self.status_var = tk.StringVar(value="Initializing...")
        tk.Label(tb, textvariable=self.status_var, font=(XP["font"],8),
                 bg=XP["taskbar_bg"], fg="#AACCFF").pack(side="right", padx=16)

    def _update_clock(self):
        self.clock_label.config(text=datetime.now().strftime("%I:%M %p"))
        self.root.after(10000, self._update_clock)

    # ─── INIT ────────────────────────────────────────────────────────────────

    def _init_agent(self):
        def _load():
            # Load plugins
            loaded = self.plugin_mgr.load_all()
            if loaded:
                self.root.after(0, lambda: self._system_msg(
                    f"🔌 Plugins: {', '.join(loaded)}", "plugin"
                ))
            # Load PDFs for RAG
            self.root.after(0, lambda: self._system_msg("📚 Loading documents..."))
            try:
                DATA_DIR.mkdir(exist_ok=True)
                docs = load_pdf_documents(DATA_DIR)
                chunks = split_documents(docs)
                create_vectorstore(chunks)
                self.root.after(0, lambda: self._system_msg(f"✅ RAG ready — {len(docs)} pages"))
            except FileNotFoundError:
                self.root.after(0, lambda: self._system_msg("⚠️ No PDFs in /data — RAG disabled", "system"))
            except Exception as e:
                self.root.after(0, lambda err=e: self._system_msg(f"⚠️ RAG init failed: {err}", "system"))

            self.root.after(0, lambda: self._system_msg(
                f"✅ Agent ready! Model: {self.llm_mgr.current_info()}"
            ))
            self.root.after(0, lambda: self.status_var.set("Ready"))

        threading.Thread(target=_load, daemon=True).start()

        self._system_msg("🖥️ AI Agent XP v2 — Multi-Model · Plugins · Streaming · Persistent Memory")
        self._system_msg("⚡ PC & Plugin commands are instant. AI chat uses the selected free model.", "system")
        self._system_msg("💡 delete/rename/move/copy/read file  |  lock pc  |  cancel shutdown", "system")
        self._system_msg("💡 Plugins: weather in Paris  |  clipboard read  |  remind me in 5 minutes to call John", "plugin")
        self._system_msg("Initializing...", "thinking")

    # ─── CHAT ────────────────────────────────────────────────────────────────

    def _append_chat(self, prefix, text, tag="bot"):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", f"{prefix} ", tag)
        self.chat_area.insert("end", f"{text}\n\n", tag)
        self.chat_area.see("end")
        self.chat_area.config(state="disabled")

    def _system_msg(self, text, tag="system"):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", f"  {text}\n", tag)
        self.chat_area.see("end")
        self.chat_area.config(state="disabled")

    def _clear_chat(self):
        self.chat_area.config(state="normal")
        self.chat_area.delete("1.0", "end")
        self.chat_area.config(state="disabled")
        self._system_msg("🧹 Chat cleared.")

    def _show_status(self):
        p  = self.persona.get_current_persona()
        tg = "Online ✅" if (self.tg_bot and self.tg_bot.running) else "Offline ❌"
        self._system_msg(
            f"📊 {p['emoji']} {p['name']} | "
            f"Model: {self.llm_mgr.current_key} | "
            f"Memory: {len(self.memory.history)} turns | "
            f"Plugins: {len(self.plugin_mgr.plugins)} | "
            f"Telegram: {tg}"
        )

    def _show_help(self):
        self._system_msg(
            "💡 COMMANDS (⚡=instant, 🤖=AI):\n\n"
            "  ⚡ FILES:\n"
            "    create file hello.py  |  create file notes.txt in documents\n"
            "    create file app.js with content: console.log('hi')\n"
            "    open file report.pdf  |  open notes.txt from desktop\n"
            "    find file *.pdf  |  list files in downloads\n"
            "    read file notes.txt\n"
            "    delete file old.txt  |  rename file a.txt to b.txt\n"
            "    move file x.txt to documents  |  copy file x.txt to downloads\n\n"
            "  ⚡ PC:\n"
            "    system info | disk info | battery info | network info\n"
            "    running processes | kill process chrome\n"
            "    startup apps | environment variables\n"
            "    open chrome | close chrome\n"
            "    go to youtube.com | search for AI tutorials\n"
            "    volume up/down/mute | screenshot\n"
            "    lock pc | shutdown | restart | sleep | cancel shutdown\n"
            "    type Hello World | hotkey ctrl+c\n\n"
            "  🔌 PLUGINS:\n"
            "    weather in Paris\n"
            "    clipboard read | clipboard write hello\n"
            "    remind me in 10 minutes to drink water\n\n"
            "  🤖 AI (current model):\n"
            "    Just chat! | task: buy milk | generate image: a sunset\n"
            "    Switch model via AI Model menu or dropdown\n"
        )

    # ─── SEND WITH STREAMING ─────────────────────────────────────────────────

    def _send_message(self):
        if self.is_thinking:
            return
        user_input = self.input_entry.get().strip()
        if not user_input:
            return
        self.input_entry.delete(0, "end")
        self._append_chat("You:", user_input, "user")

        # Task shortcut
        if user_input.lower().startswith("task:"):
            self._system_msg(self.task_manager.add_task(user_input[5:].strip()))
            return

        # Image generation
        if user_input.lower().startswith("generate image:"):
            self._run_image_gen(user_input[15:].strip())
            return

        # Memory search
        if user_input.lower().startswith("search memory:"):
            q = user_input[14:].strip()
            hits = self.memory.search(q)
            self._system_msg("🧠 Memory search:\n" + ("\n".join(hits) if hits else "(no results)"))
            return

        # Plugin
        plugin_result = self.plugin_mgr.handle(user_input)
        if plugin_result is not None:
            self._append_chat("🔌 Plugin:", plugin_result, "plugin")
            return

        # Local PC command
        local_result = self.pc_agent.handle(user_input)
        if local_result == "→ plugin_list":
            self._system_msg(self.plugin_mgr.list_plugins(), "plugin")
            return
        if local_result is not None:
            if "screenshot saved:" in local_result.lower():
                path = local_result.split(": ", 1)[-1].strip()
                try:
                    img_bytes = Path(path).read_bytes()
                    self.root.after(0, lambda: self._show_image_window(img_bytes, "Screenshot"))
                except Exception:
                    pass
            self._append_chat("⚡ PC:", local_result, "pc")
            self.status_var.set("Ready")
            return

        # Validate API keys for current provider
        cfg = FREE_MODELS[self.llm_mgr.current_key]
        if cfg["provider"] == "groq" and not os.getenv("GROQ_API_KEY", ""):
            self._system_msg("⚠️ GROQ_API_KEY missing in .env", "error")
            return
        if cfg["provider"] == "openrouter" and not os.getenv("OPENROUTER_API_KEY", ""):
            self._system_msg("⚠️ OPENROUTER_API_KEY missing in .env", "error")
            return

        self.is_thinking = True
        self._stream_has_token = False
        self.status_var.set("Thinking...")

        # Insert persona prefix (will be followed by streamed tokens)
        p = self.persona.get_current_persona()
        prefix = f"{p['emoji']} {p['name']}: "
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", prefix, "bot")
        self._stream_prefix_start = self.chat_area.index(f"end-{len(prefix)+1}c")
        self.chat_area.config(state="disabled")

        def _stream_token(token):
            self._stream_has_token = True
            self.root.after(0, lambda t=token: self._insert_stream_token(t))

        def _think():
            answer = ""
            err = None
            try:
                ctx_text = self.memory.get_context()
                system   = (
                    self.persona.get_persona_prompt()
                    + f"\n\nContext:\n{ctx_text}\n\n"
                    + f"Active tasks:{self.task_manager.get_active_context()}"
                )
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_input},
                ]
                answer = self.llm_mgr.chat(messages, stream_cb=_stream_token)
                if answer:
                    self.memory.add("user", user_input)
                    self.memory.add(p["name"].lower(), answer)
            except Exception as e:
                err = e
            finally:
                self.root.after(0, lambda: self._finish_stream(err))
                self.root.after(0, lambda: self.status_var.set("Ready"))
                self.is_thinking = False

        threading.Thread(target=_think, daemon=True).start()

    def _insert_stream_token(self, token: str):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", token, "stream")
        self.chat_area.see("end")
        self.chat_area.config(state="disabled")

    def _finish_stream(self, err=None):
        self.chat_area.config(state="normal")
        if err is not None:
            if not self._stream_has_token:
                self.chat_area.insert("end", f"⚠️ Error: {err}", "error")
            else:
                self.chat_area.insert("end", f"\n⚠️ Error: {err}", "error")
        self.chat_area.insert("end", "\n\n", "bot")
        self.chat_area.see("end")
        self.chat_area.config(state="disabled")

    # ─── LOCAL CMD RUNNER ────────────────────────────────────────────────────

    def _local_cmd(self, cmd: str):
        self.status_var.set("⚡ Running...")

        def _run():
            result = self.plugin_mgr.handle(cmd) or self.pc_agent.handle(cmd) or "❓ Not recognized."
            if result == "→ plugin_list":
                result = self.plugin_mgr.list_plugins()
            if "screenshot saved:" in result.lower():
                path = result.split(": ", 1)[-1].strip()
                try:
                    img_bytes = Path(path).read_bytes()
                    self.root.after(0, lambda: self._show_image_window(img_bytes, "Screenshot"))
                except Exception:
                    pass
            self.root.after(0, lambda: self._append_chat("⚡ PC:", result, "pc"))
            self.root.after(0, lambda: self.status_var.set("Ready"))

        threading.Thread(target=_run, daemon=True).start()

    # ─── PERSONA / MODEL ─────────────────────────────────────────────────────

    def _switch_persona(self, key):
        if self.persona.switch_persona(key):
            p = self.persona.get_current_persona()
            self._system_msg(f"✨ Switched to: {p['emoji']} {p['name']}")
            self._update_persona_panel()

    def _update_persona_panel(self):
        p = self.persona.get_current_persona()
        self.persona_label.config(text=f"{p['emoji']} {p['name']}")
        self.persona_desc.config(text=p["description"])

    def _switch_model(self, key):
        if self.llm_mgr.switch_model(key):
            self.model_var.set(key)
            self._system_msg(f"🤖 Model: {self.llm_mgr.current_info()}", "system")

    # ─── PLUGINS ─────────────────────────────────────────────────────────────

    def _reload_plugins(self):
        for name in list(self.plugin_mgr.plugins.keys()):
            self.plugin_mgr.unload(name)
        loaded = self.plugin_mgr.load_all()
        self._system_msg(f"🔌 Reloaded: {', '.join(loaded) if loaded else 'none'}", "plugin")

    # ─── MEMORY BROWSER ──────────────────────────────────────────────────────

    def _show_memory_browser(self):
        win = tk.Toplevel(self.root)
        win.title("🧠 Memory Browser")
        win.geometry("600x400")
        win.configure(bg=XP["bg"])

        tb = tk.Frame(win, bg=XP["title_bar"], height=28)
        tb.pack(fill="x")
        tb.pack_propagate(False)
        tk.Label(tb, text="  🧠 Memory Browser", font=(XP["font"],9,"bold"),
                 bg=XP["title_bar"], fg=XP["title_fg"]).pack(side="left", pady=4)
        tk.Button(tb, text="✕", font=("Tahoma",8,"bold"), bg="#C0302C", fg="white",
                  relief="raised", bd=1, width=3, cursor="hand2",
                  command=win.destroy).pack(side="right", pady=3, padx=2)

        txt = tk.Text(win, wrap="word", font=(XP["font"],9), bg=XP["chat_bg"],
                      relief="sunken", bd=2, padx=6, pady=6)
        sb  = ttk.Scrollbar(win, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True, padx=6, pady=6)

        ctx = self.memory.get_context()
        txt.insert("end", ctx or "(memory is empty)")
        txt.config(state="disabled")

        bf = tk.Frame(win, bg=XP["bg"])
        bf.pack(fill="x", padx=6, pady=4)
        xp_button(bf, "🧹 Clear", lambda: (self.memory.clear(), win.destroy()), width=10).pack(side="left")
        xp_button(bf, "Close",   win.destroy, width=10).pack(side="right")

    def _search_memory_dialog(self):
        try:
            q = simpledialog.askstring("Search Memory", "Enter keyword:", parent=self.root)
        except Exception:
            return
        if q:
            hits = self.memory.search(q)
            self._system_msg(
                f"🧠 Memory search '{q}':\n" +
                ("\n".join(f"  • {h}" for h in hits) if hits else "  (no results)")
            )

    # ─── MENU ACTIONS ────────────────────────────────────────────────────────

    def _menu_open_file(self):
        path = filedialog.askopenfilename(title="Open File")
        if path:
            try:
                os.startfile(path)
                self._system_msg(f"✅ Opened: {Path(path).name}")
            except Exception as e:
                self._system_msg(f"❌ {e}", "error")

    def _save_chat_log(self):
        path = filedialog.asksaveasfilename(
            title="Save Chat Log", defaultextension=".txt",
            filetypes=[("Text files","*.txt"),("All files","*.*")]
        )
        if path:
            self.chat_area.config(state="normal")
            content = self.chat_area.get("1.0","end")
            self.chat_area.config(state="disabled")
            try:
                Path(path).write_text(content, encoding="utf-8")
                self._system_msg(f"💾 Chat saved: {path}")
            except Exception as e:
                self._system_msg(f"❌ {e}", "error")

    # ─── TELEGRAM ────────────────────────────────────────────────────────────

    def _start_telegram(self):
        if not TELEGRAM_TOKEN:
            messagebox.showerror("Error", "TELEGRAM_TOKEN not in .env!")
            return
        if self.tg_bot and self.tg_bot.running:
            self._system_msg("📱 Bot already running!", "system")
            return
        self.tg_bot = TelegramBotManager(
            token=TELEGRAM_TOKEN, allowed_id=ALLOWED_USER_ID,
            llm_mgr=self.llm_mgr, persona=self.persona,
            task_mgr=self.task_manager, pc_agent=self.pc_agent,
            voice=self.voice, memory=self.memory,
            plugin_mgr=self.plugin_mgr,
            log_cb=lambda m: self.root.after(0, lambda: self._system_msg(f"📱 {m}", "telegram")),
        )
        self.tg_bot.start()
        self.tg_status_label.config(text="⬤ Online", fg="#4CAF50")
        self._system_msg("📱 Telegram bot started! Send /start in Telegram.", "telegram")

    def _stop_telegram(self):
        if self.tg_bot:
            self.tg_bot.stop()
            self.tg_status_label.config(text="⬤ Offline", fg="#CC0000")
            self._system_msg("📱 Telegram bot stopped.", "system")

    def _telegram_status(self):
        if self.tg_bot and self.tg_bot.running:
            self._system_msg("📱 Telegram: ONLINE ✅", "telegram")
        else:
            self._system_msg("📱 Telegram: OFFLINE ❌", "system")

    def _send_file_to_telegram(self):
        if not (self.tg_bot and self.tg_bot.running):
            messagebox.showwarning("Telegram", "Start the Telegram bot first!")
            return
        path = filedialog.askopenfilename(title="Select file to send via Telegram")
        if path:
            self.tg_bot.send_file(path, caption=f"📤 {Path(path).name}")
            self._system_msg(f"📤 Sending to Telegram: {Path(path).name}", "telegram")

    # ─── IMAGE ───────────────────────────────────────────────────────────────

    def _generate_image(self):
        prompt = self.img_entry.get().strip()
        if not prompt or prompt == "describe an image...":
            self._system_msg("⚠️ Enter a description first.", "error")
            return
        self.img_entry.delete(0, "end")
        self._run_image_gen(prompt)

    def _run_image_gen(self, prompt):
        self.status_var.set("Generating image...")
        self._system_msg(f"🎨 Generating: {prompt}")

        def _gen():
            data = self.image_gen.generate_image(prompt)
            if data:
                try:
                    path = self.image_gen.save_image(data, prompt)
                    self.root.after(0, lambda p=path: self._system_msg(f"✅ Saved: {p}"))
                    self.root.after(0, lambda d=data: self._show_image_window(d, prompt))
                except Exception as e:
                    self.root.after(0, lambda err=e: self._system_msg(f"❌ {err}", "error"))
            else:
                self.root.after(0, lambda: self._system_msg("❌ Image generation failed.", "error"))
            self.root.after(0, lambda: self.status_var.set("Ready"))

        threading.Thread(target=_gen, daemon=True).start()

    def _show_image_window(self, image_data: bytes, title: str):
        win = tk.Toplevel(self.root)
        win.title(title[:50])
        win.configure(bg=XP["bg"])
        tb = tk.Frame(win, bg=XP["title_bar"], height=28)
        tb.pack(fill="x")
        tb.pack_propagate(False)
        tk.Label(tb, text=f"  🖼️  {title[:50]}", font=(XP["font"],9,"bold"),
                 bg=XP["title_bar"], fg=XP["title_fg"]).pack(side="left", pady=4)
        tk.Button(tb, text="✕", font=("Tahoma",8,"bold"), bg="#C0302C", fg="white",
                  relief="raised", bd=1, width=3, cursor="hand2",
                  command=win.destroy).pack(side="right", pady=3, padx=2)
        try:
            img   = Image.open(BytesIO(image_data))
            img.thumbnail((640, 520))
            photo = ImageTk.PhotoImage(img)
            lbl   = tk.Label(win, image=photo, bg=XP["bg"])
            lbl.image = photo  # keep reference
            lbl.pack(padx=10, pady=10)
        except Exception as e:
            tk.Label(win, text=f"Could not display: {e}",
                     bg=XP["bg"], font=(XP["font"],9)).pack(pady=20)


# ============= RAG HELPERS =============
def load_pdf_documents(data_dir: Path):
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError("No PDFs")
    docs = []
    for p in pdfs:
        loader = PyPDFLoader(str(p))
        pages  = loader.load()
        for pg in pages:
            pg.metadata["source"] = p.name
        docs.extend(pages)
    return docs

def split_documents(docs):
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)

def create_vectorstore(chunks):
    return FAISS.from_documents(
        chunks, HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )


# ============= ENTRY POINT =============
def main():
    root = tk.Tk()
    root.withdraw()
    app = AgentXPApp(root)
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()