import os
import sys
import time
import json
import base64
import threading
import subprocess
import tempfile
import winreg
import webbrowser
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

import psutil
import requests
import pyautogui
import pygetwindow as gw
from PIL import Image, ImageDraw, ImageFont, ImageTk
from dotenv import load_dotenv

import tkinter as tk
from tkinter import ttk, messagebox

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)

import whisper

# ============= CONFIGURATION =============
load_dotenv()
DATA_DIR        = Path("data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.1-8b-instant"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
TOP_K           = 3

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "")
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0"))

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
    "font":         "Tahoma",
    "font_size":    9,
    "taskbar_bg":   "#245EDC",
    "taskbar_btn":  "#3A7FDE",
}

# ============= AGENT PERSONAS =============
class AgentPersona:
    PERSONAS = {
        "assistant": {
            "name": "Assistant", "emoji": "🤖",
            "description": "Helpful, efficient, and professional AI assistant",
            "traits": "Be helpful, concise, and efficient. Focus on getting tasks done. Provide clear answers and solutions.",
            "response_style": "professional, clear, solution-oriented",
            "image_style": "professional, clean, minimalist"
        },
        "friend": {
            "name": "Friend", "emoji": "👋",
            "description": "Casual, friendly conversational companion",
            "traits": "Be warm, empathetic, and conversational. Show genuine interest. Use casual language.",
            "response_style": "conversational, warm, engaging",
            "image_style": "friendly, warm, casual"
        },
        "mentor": {
            "name": "Mentor", "emoji": "🎓",
            "description": "Wise, experienced guide and teacher",
            "traits": "Be knowledgeable, patient, and encouraging. Provide guidance and wisdom.",
            "response_style": "wise, instructive, supportive",
            "image_style": "professional, authoritative, calm"
        },
        "creative": {
            "name": "Creative Partner", "emoji": "🎨",
            "description": "Creative brainstorming partner",
            "traits": "Be imaginative, open-minded, and playful. Generate creative ideas and solutions.",
            "response_style": "creative, inspiring, imaginative",
            "image_style": "artistic, colorful, creative"
        },
        "analyst": {
            "name": "Analyst", "emoji": "📊",
            "description": "Data-driven analytical thinker",
            "traits": "Be logical, precise, and analytical. Break down complex problems.",
            "response_style": "analytical, structured, data-focused",
            "image_style": "clean, organized, minimal"
        },
    }

    def __init__(self):
        self.current_persona = "assistant"
        self.persona_history = []

    def switch_persona(self, persona_name: str) -> bool:
        if persona_name in self.PERSONAS:
            self.persona_history.append(self.current_persona)
            self.current_persona = persona_name
            return True
        return False

    def get_current_persona(self) -> Dict:
        return self.PERSONAS[self.current_persona]

    def get_persona_prompt(self) -> str:
        persona = self.get_current_persona()
        return f"""
You are {persona['name']}, an AI {persona['description']}.
PERSONALITY TRAITS: {persona['traits']}
RESPONSE STYLE: {persona['response_style']}
CRITICAL RULES:
- Stay in character as {persona['name']}
- Adapt your responses to match the persona
- Be consistent with the persona's traits
"""

    def list_personas(self) -> str:
        result = "\n📋 Available Personas:\n"
        for key, value in self.PERSONAS.items():
            current = " ✅ CURRENT" if key == self.current_persona else ""
            result += f"   • {value['emoji']} {value['name']} - {value['description']}{current}\n"
        return result


# ============= TASK MANAGEMENT SYSTEM =============
class TaskManager:
    def __init__(self):
        self.tasks = []
        self.context = {}

    def add_task(self, task: str, priority: str = "normal"):
        self.tasks.append({
            "task": task,
            "priority": priority,
            "created": datetime.now(),
            "status": "pending"
        })
        return f"✅ Task added: {task}"

    def get_active_context(self) -> str:
        if self.tasks:
            pending = [t for t in self.tasks if t['status'] == 'pending']
            if pending:
                return f"\nCurrent pending tasks: {', '.join([t['task'] for t in pending[:3]])}"
        return ""

    def clear_tasks(self):
        self.tasks = []
        return "🧹 All tasks cleared"


# ============= LOCAL PC AGENT (INSTANT — NO API CALLS) =============
class LocalPCAgent:
    """
    Handles PC tasks locally and instantly without any LLM API calls.
    Uses keyword parsing to route commands.
    """

    APP_MAP = {
        "notepad":       "notepad.exe",
        "calculator":    "calc.exe",
        "calc":          "calc.exe",
        "paint":         "mspaint.exe",
        "explorer":      "explorer.exe",
        "chrome":        "chrome.exe",
        "firefox":       "firefox.exe",
        "edge":          "msedge.exe",
        "word":          "WINWORD.EXE",
        "excel":         "EXCEL.EXE",
        "powerpoint":    "POWERPNT.EXE",
        "vlc":           "vlc.exe",
        "spotify":       "Spotify.exe",
        "discord":       "Discord.exe",
        "vscode":        "code.exe",
        "pycharm":       "pycharm64.exe",
        "terminal":      "cmd.exe",
        "cmd":           "cmd.exe",
        "powershell":    "powershell.exe",
        "task manager":  "taskmgr.exe",
        "control panel": "control.exe",
        "settings":      "ms-settings:",
    }

    def __init__(self, log_cb=None):
        self.log_cb = log_cb or print
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE    = 0.3

    def handle(self, command: str) -> str:
        """Route command to the right local handler instantly."""
        cmd = command.lower().strip()

        # Files & Folders
        if any(w in cmd for w in ["find file", "search file", "locate file"]):
            return self._find_files(command)
        if any(w in cmd for w in ["list files", "show files", "files in", "what's in", "whats in"]):
            return self._list_folder(command)
        if any(w in cmd for w in ["find folder", "search folder"]):
            return self._find_folders(command)
        if any(w in cmd for w in ["open desktop", "open documents", "open downloads",
                                   "open pictures", "open videos", "open music"]):
            return self._open_location(command)

        # Apps
        if any(w in cmd for w in ["list apps", "installed apps", "what apps", "show apps", "my apps"]):
            return self._list_installed_apps()
        if any(w in cmd for w in ["open ", "launch ", "start ", "run "]):
            return self._open_app(command)
        if any(w in cmd for w in ["close ", "kill app", "stop app"]):
            return self._close_app(command)

        # System Stats
        if any(w in cmd for w in ["cpu", "ram", "memory usage"]):
            return self._system_stats()
        if any(w in cmd for w in ["disk", "storage", "hard drive", "free space"]):
            return self._disk_stats()
        if any(w in cmd for w in ["battery", "power level"]):
            return self._battery_info()
        if any(w in cmd for w in ["system info", "pc info", "computer info", "about my pc"]):
            return self._full_system_info()
        if any(w in cmd for w in ["network", "wifi", "ip address", "internet info"]):
            return self._network_info()

        # Processes
        if any(w in cmd for w in ["running", "processes", "active apps", "what is running"]):
            return self._running_processes()
        if any(w in cmd for w in ["kill process", "end process", "stop process"]):
            return self._kill_process(command)

        # Screenshot
        if any(w in cmd for w in ["screenshot", "screen capture", "capture screen"]):
            return self._take_screenshot()

        # Web
        if any(w in cmd for w in ["go to", "open website", "browse", "navigate to"]):
            return self._open_url(command)
        if any(w in cmd for w in ["search for", "google ", "look up", "search google"]):
            return self._search_web(command)

        # Volume
        if "volume" in cmd:
            return self._control_volume(command)

        # ── POWER COMMANDS ────────────────────────────────────────────────────
        if any(w in cmd for w in ["shut down", "shutdown", "turn off pc", "power off"]):
            return self._shutdown_pc()
        if any(w in cmd for w in ["restart", "reboot"]):
            return self._restart_pc()
        if any(w in cmd for w in ["sleep", "hibernate"]):
            return self._sleep_pc()

        return None  # None means: not a local command, pass to AI

    # ── FILES ────────────────────────────────────────────────────────────────
    def _find_files(self, command: str) -> str:
        keywords = ["find file", "search file", "locate file"]
        query = command.lower()
        for kw in keywords:
            query = query.replace(kw, "").strip()
        query = query.strip(": ").strip()

        if not query:
            return "❓ Specify file name: 'find file report.pdf'"

        self.log_cb(f"🔍 Searching: {query}")
        results = []
        search_dirs = [Path.home(), Path.home()/"Desktop",
                       Path.home()/"Documents", Path.home()/"Downloads",
                       Path.home()/"Pictures", Path.home()/"Videos"]

        for d in search_dirs:
            if d.exists():
                try:
                    for match in d.rglob(f"*{query}*"):
                        results.append(str(match))
                        if len(results) >= 20:
                            break
                except PermissionError:
                    pass
            if len(results) >= 20:
                break

        if results:
            out = f"🔍 Found {len(results)} result(s) for '{query}':\n"
            for r in results:
                out += f"  📄 {r}\n"
            return out
        return f"❌ No files found matching '{query}'"

    def _list_folder(self, command: str) -> str:
        cmd = command.lower()
        folder = Path.home()
        if "desktop" in cmd:      folder = Path.home() / "Desktop"
        elif "documents" in cmd:  folder = Path.home() / "Documents"
        elif "downloads" in cmd:  folder = Path.home() / "Downloads"
        elif "pictures" in cmd:   folder = Path.home() / "Pictures"
        elif "videos" in cmd:     folder = Path.home() / "Videos"
        elif "music" in cmd:      folder = Path.home() / "Music"

        if not folder.exists():
            return f"❌ Folder not found: {folder}"

        items   = list(folder.iterdir())
        files   = [i for i in items if i.is_file()]
        folders = [i for i in items if i.is_dir()]

        out  = f"📁 {folder}\n  📂 {len(folders)} folders, 📄 {len(files)} files\n\n"
        if folders:
            out += "📂 Folders:\n"
            for f in sorted(folders)[:15]:
                out += f"  • {f.name}\n"
        if files:
            out += "\n📄 Files:\n"
            for f in sorted(files)[:20]:
                size = f.stat().st_size
                size_str = f"{size//1024}KB" if size > 1024 else f"{size}B"
                out += f"  • {f.name} ({size_str})\n"
        return out

    def _find_folders(self, command: str) -> str:
        query = command.lower()
        for kw in ["find folder", "search folder"]:
            query = query.replace(kw, "").strip()
        query = query.strip(": ").strip()
        results = []
        try:
            for match in Path.home().rglob(f"*{query}*"):
                if match.is_dir():
                    results.append(str(match))
                    if len(results) >= 15:
                        break
        except PermissionError:
            pass
        if results:
            out = f"📂 Found {len(results)} folder(s) matching '{query}':\n"
            for r in results:
                out += f"  📂 {r}\n"
            return out
        return f"❌ No folders found matching '{query}'"

    def _open_location(self, command: str) -> str:
        cmd = command.lower()
        if "desktop" in cmd:      path = Path.home() / "Desktop"
        elif "documents" in cmd:  path = Path.home() / "Documents"
        elif "downloads" in cmd:  path = Path.home() / "Downloads"
        elif "pictures" in cmd:   path = Path.home() / "Pictures"
        elif "videos" in cmd:     path = Path.home() / "Videos"
        elif "music" in cmd:      path = Path.home() / "Music"
        else:                     path = Path.home()
        subprocess.Popen(f'explorer "{path}"')
        return f"📁 Opened: {path}"

    # ── APPS ─────────────────────────────────────────────────────────────────
    def _list_installed_apps(self) -> str:
        self.log_cb("📋 Scanning installed apps...")
        apps = []
        reg_paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        ]
        for reg_path in reg_paths:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
                for i in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        subkey = winreg.OpenKey(key, winreg.EnumKey(key, i))
                        try:
                            name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                            if name and name.strip():
                                apps.append(name.strip())
                        except FileNotFoundError:
                            pass
                    except:
                        pass
            except:
                pass
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                  r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")
            for i in range(winreg.QueryInfoKey(key)[0]):
                try:
                    subkey = winreg.OpenKey(key, winreg.EnumKey(key, i))
                    try:
                        name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                        if name and name.strip():
                            apps.append(name.strip())
                    except FileNotFoundError:
                        pass
                except:
                    pass
        except:
            pass

        apps = sorted(set(apps))
        if apps:
            out = f"📋 Found {len(apps)} installed apps:\n"
            for app in apps[:50]:
                out += f"  • {app}\n"
            if len(apps) > 50:
                out += f"  ... and {len(apps)-50} more\n"
            return out
        return "❌ Could not retrieve installed apps"

    def _open_app(self, command: str) -> str:
        cmd = command.lower()

        SEARCH_PATHS = [
            Path(os.environ.get("ProgramFiles",      "C:/Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")),
            Path(os.environ.get("LocalAppData",      Path.home() / "AppData" / "Local")),
            Path(os.environ.get("AppData",           Path.home() / "AppData" / "Roaming")),
        ]

        for keyword, exe in self.APP_MAP.items():
            if keyword in cmd:
                self.log_cb(f"🖥️ Opening: {exe}")

                if exe.startswith("ms-"):
                    try:
                        subprocess.Popen(f"start {exe}", shell=True)
                        return f"✅ Opened: {keyword.title()}"
                    except Exception as e:
                        return f"❌ Failed to open {keyword}: {e}"

                try:
                    result = subprocess.run(
                        f'where "{exe}"',
                        shell=True, capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        path = result.stdout.strip().splitlines()[0]
                        subprocess.Popen(f'"{path}"', shell=True)
                        return f"✅ Opened: {keyword.title()}"
                except Exception:
                    pass

                found_path = None
                for search_dir in SEARCH_PATHS:
                    if not search_dir.exists():
                        continue
                    try:
                        matches = list(search_dir.rglob(exe))
                        if matches:
                            found_path = matches[0]
                            break
                    except (PermissionError, OSError):
                        continue

                if found_path:
                    subprocess.Popen(f'"{found_path}"', shell=True)
                    return f"✅ Opened: {keyword.title()} ({found_path})"

                try:
                    r = subprocess.run(
                        f'powershell -Command "Start-Process \'{exe}\'"',
                        shell=True, capture_output=True, text=True, timeout=8
                    )
                    if r.returncode == 0:
                        return f"✅ Opened: {keyword.title()}"
                    else:
                        err = r.stderr.strip() or "not found"
                        return f"❌ Could not open '{keyword}': {err}"
                except subprocess.TimeoutExpired:
                    return f"⚠️ Timed out trying to open '{keyword}'. It may have launched anyway."
                except Exception as e:
                    return f"❌ Failed to open '{keyword}': {e}"

        for trigger in ["open ", "launch ", "start ", "run "]:
            if trigger in cmd:
                app_name = cmd.split(trigger, 1)[1].strip()
                if not app_name:
                    continue

                try:
                    result = subprocess.run(
                        f'where "{app_name}"',
                        shell=True, capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        path = result.stdout.strip().splitlines()[0]
                        subprocess.Popen(f'"{path}"', shell=True)
                        return f"✅ Opened: {app_name}"
                except Exception:
                    pass

                try:
                    r = subprocess.run(
                        f'powershell -Command "Start-Process \'{app_name}\'"',
                        shell=True, capture_output=True, text=True, timeout=8
                    )
                    if r.returncode == 0:
                        return f"✅ Opened: {app_name}"
                    else:
                        return (
                            f"❌ Could not open '{app_name}'.\n"
                            f"   Make sure the app is installed.\n"
                            f"   Try: open chrome / open notepad / open calc"
                        )
                except subprocess.TimeoutExpired:
                    return f"⚠️ Timed out opening '{app_name}'. It may have launched anyway."
                except Exception as e:
                    return f"❌ Error opening '{app_name}': {e}"

        return "❓ App not recognized. Try: open chrome / open notepad / open calc"

    def _close_app(self, command: str) -> str:
        cmd = command.lower()
        for keyword in self.APP_MAP:
            if keyword in cmd:
                exe = self.APP_MAP[keyword].replace(".exe", "").lower()
                killed = 0
                for proc in psutil.process_iter(['name', 'pid']):
                    if exe in proc.info['name'].lower():
                        proc.kill()
                        killed += 1
                return f"✅ Closed {killed} instance(s) of {keyword}" if killed else f"⚠️ {keyword} was not running"
        return "❓ Specify app: 'close chrome'"

    # ── SYSTEM STATS ─────────────────────────────────────────────────────────
    def _system_stats(self) -> str:
        cpu   = psutil.cpu_percent(interval=1)
        ram   = psutil.virtual_memory()
        out   = "📊 SYSTEM STATS\n"
        out  += f"  🖥️  CPU Usage: {cpu}% ({psutil.cpu_count()} cores)\n"
        out  += f"  🧠 RAM Total: {ram.total//(1024**3)} GB\n"
        out  += f"  🧠 RAM Used:  {ram.used//(1024**3)} GB ({ram.percent}%)\n"
        out  += f"  🧠 RAM Free:  {ram.available//(1024**3)} GB\n"
        return out

    def _disk_stats(self) -> str:
        out = "💾 DISK STORAGE\n"
        for part in psutil.disk_partitions():
            try:
                u    = psutil.disk_usage(part.mountpoint)
                pct  = u.percent
                bar  = "█" * int(pct/10) + "░" * (10 - int(pct/10))
                out += f"\n  Drive {part.device}\n"
                out += f"  [{bar}] {pct}%\n"
                out += f"  Total: {u.total//(1024**3)}GB | Used: {u.used//(1024**3)}GB | Free: {u.free//(1024**3)}GB\n"
            except PermissionError:
                pass
        return out

    def _battery_info(self) -> str:
        b = psutil.sensors_battery()
        if not b:
            return "🔌 No battery detected (desktop PC)"
        status = "Charging 🔌" if b.power_plugged else "On Battery 🔋"
        secs   = b.secsleft
        tl     = f"{secs//3600}h {(secs%3600)//60}m" if secs > 0 else "Calculating..."
        return f"🔋 Battery: {b.percent:.1f}%\n  Status: {status}\n  Time left: {tl}"

    def _full_system_info(self) -> str:
        import platform
        out  = "🖥️ SYSTEM INFORMATION\n"
        out += f"  OS:        {platform.system()} {platform.release()}\n"
        out += f"  Version:   {platform.version()[:50]}\n"
        out += f"  Machine:   {platform.machine()}\n"
        out += f"  Processor: {platform.processor()[:50]}\n"
        out += f"  CPU Cores: {psutil.cpu_count()} ({psutil.cpu_count(logical=False)} physical)\n"
        out += f"  RAM:       {psutil.virtual_memory().total//(1024**3)} GB\n"
        out += f"  Python:    {platform.python_version()}\n"
        out += f"  Hostname:  {platform.node()}\n"
        boot = datetime.fromtimestamp(psutil.boot_time())
        out += f"  Boot time: {boot.strftime('%Y-%m-%d %H:%M')}\n"
        return out

    def _network_info(self) -> str:
        import socket
        out = "🌐 NETWORK INFO\n"
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            out += f"  Hostname: {hostname}\n  Local IP: {local_ip}\n"
        except:
            pass
        stats = psutil.net_io_counters()
        out  += f"  Sent:     {stats.bytes_sent//(1024**2)} MB\n"
        out  += f"  Received: {stats.bytes_recv//(1024**2)} MB\n"
        for iface, addrs in list(psutil.net_if_addrs().items())[:5]:
            for addr in addrs:
                if addr.family == 2:
                    out += f"  • {iface}: {addr.address}\n"
        return out

    # ── PROCESSES ────────────────────────────────────────────────────────────
    def _running_processes(self) -> str:
        procs = []
        for proc in psutil.process_iter(['name', 'pid', 'memory_info']):
            try:
                mem = proc.info['memory_info'].rss // (1024**2)
                procs.append((proc.info['name'], proc.info['pid'], mem))
            except:
                pass
        procs = sorted(procs, key=lambda x: x[2], reverse=True)
        out   = f"⚙️ TOP RUNNING PROCESSES ({len(procs)} total)\n"
        out  += f"  {'Name':<30} {'PID':<8} RAM\n  " + "-"*45 + "\n"
        for name, pid, mem in procs[:20]:
            out += f"  {name:<30} {pid:<8} {mem} MB\n"
        return out

    def _kill_process(self, command: str) -> str:
        cmd = command.lower()
        for trigger in ["kill process", "end process", "stop process"]:
            if trigger in cmd:
                target = cmd.split(trigger, 1)[1].strip()
                killed = 0
                for proc in psutil.process_iter(['name', 'pid']):
                    if target in proc.info['name'].lower():
                        proc.kill()
                        killed += 1
                return f"✅ Killed {killed} process(es) matching '{target}'" if killed else f"⚠️ No process: '{target}'"
        return "❓ Specify: 'kill process chrome'"

    # ── WEB / VOLUME / SCREENSHOT ────────────────────────────────────────────
    def _open_url(self, command: str) -> str:
        cmd = command.lower()
        for trigger in ["go to", "open website", "browse", "navigate to"]:
            if trigger in cmd:
                url = cmd.split(trigger, 1)[1].strip()
                if not url.startswith("http"):
                    url = "https://" + url
                webbrowser.open(url)
                return f"🌐 Opened: {url}"
        return "❓ Specify URL: 'go to youtube.com'"

    def _search_web(self, command: str) -> str:
        cmd = command.lower()
        for trigger in ["search for", "google ", "look up", "search google"]:
            if trigger in cmd:
                query = cmd.split(trigger, 1)[1].strip()
                webbrowser.open(f"https://www.google.com/search?q={requests.utils.quote(query)}")
                return f"🔎 Searching Google for: {query}"
        return "❓ Specify: 'search for python tutorials'"

    def _control_volume(self, command: str) -> str:
        cmd = command.lower()
        if "mute" in cmd:
            pyautogui.press("volumemute")
            return "🔇 Volume muted"
        elif "up" in cmd or "increase" in cmd:
            for _ in range(5): pyautogui.press("volumeup")
            return "🔊 Volume increased"
        elif "down" in cmd or "decrease" in cmd or "lower" in cmd:
            for _ in range(5): pyautogui.press("volumedown")
            return "🔉 Volume decreased"
        return "❓ Try: volume up / down / mute"

    def _take_screenshot(self) -> str:
        img  = pyautogui.screenshot()
        path = Path("screenshots") / f"ss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path.parent.mkdir(exist_ok=True)
        img.save(path)
        return f"📸 Screenshot saved: {path}"

    def take_screenshot_bytes(self) -> bytes:
        img = pyautogui.screenshot()
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ── POWER COMMANDS ───────────────────────────────────────────────────────
    def _shutdown_pc(self) -> str:
        self.log_cb("⚠️ Shutting down PC in 10 seconds...")
        subprocess.Popen("shutdown /s /t 10", shell=True)
        return "⚠️ PC will shut down in 10 seconds.\nTo cancel: run 'shutdown /a' in CMD."

    def _restart_pc(self) -> str:
        self.log_cb("🔄 Restarting PC in 10 seconds...")
        subprocess.Popen("shutdown /r /t 10", shell=True)
        return "🔄 PC will restart in 10 seconds.\nTo cancel: run 'shutdown /a' in CMD."

    def _sleep_pc(self) -> str:
        self.log_cb("💤 Putting PC to sleep...")
        subprocess.Popen("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True)
        return "💤 PC going to sleep..."


# ============= IMAGE GENERATOR =============
class AgentImageGenerator:
    def __init__(self, persona: AgentPersona):
        self.persona = persona

    def generate_image(self, prompt: str, style: str = "") -> Optional[bytes]:
        try:
            persona_style = self.persona.get_current_persona()['image_style']
            enhanced      = f"{prompt}, {style or persona_style}, high quality, detailed"
            encoded       = requests.utils.quote(enhanced)
            url           = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true"
            print("   📡 Trying Pollinations.ai...")
            r = requests.get(url, timeout=45)
            if r.status_code == 200 and len(r.content) > 5000:
                if r.content.startswith(b'\xff\xd8') or r.content.startswith(b'\x89PNG'):
                    print("   ✅ Image generated!")
                    return r.content
        except Exception as e:
            print(f"   ⚠️ Image error: {e}")
        return None

    def save_image(self, image_bytes: bytes, prompt: str) -> Path:
        img_dir = Path("agent_images")
        img_dir.mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        path = img_dir / f"image_{safe}_{ts}.png"
        with open(path, "wb") as f:
            f.write(image_bytes)
        return path


# ============= WHISPER VOICE TRANSCRIPTION =============
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
            os.unlink(tmp_path)


# ============= RAG FUNCTIONS =============
def load_pdf_documents(data_dir: Path):
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("No PDF files found")
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
        documents.extend(pages)
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)

def answer_question(question, llm, prompt, memory):
    history      = "\n".join(memory[-10:])
    final_prompt = prompt.format(history=history, question=question)
    response     = llm.invoke(final_prompt).content
    return response

def build_agent_prompt(persona: AgentPersona, task_manager: TaskManager):
    persona_info = persona.get_current_persona()
    task_context = task_manager.get_active_context()
    template = f"""
{persona.get_persona_prompt()}

CURRENT CAPABILITIES:
- Answer questions and provide information
- Help with tasks and problem-solving
- Generate images when asked
- Control the PC locally

{task_context}

IMPORTANT RULES:
1. Always respond in character as {persona_info['name']}
2. Be helpful and engaging

Conversation history:
{{history}}

User: {{question}}

{persona_info['emoji']} {persona_info['name']}:"""
    return PromptTemplate.from_template(template)


# ============= TELEGRAM BOT =============
class TelegramBotManager:
    def __init__(self, token: str, allowed_id: int,
                 llm_ref, persona_ref, task_ref,
                 pc_agent_ref, voice_ref, memory_ref,
                 log_cb=None):
        self.token      = token
        self.allowed_id = allowed_id
        self.llm        = llm_ref
        self.persona    = persona_ref
        self.task_mgr   = task_ref
        self.pc_agent   = pc_agent_ref
        self.voice      = voice_ref
        self.memory     = memory_ref
        self.log_cb     = log_cb or print
        self.app        = None
        self._thread    = None
        self.running    = False

    def _is_allowed(self, update: Update) -> bool:
        if self.allowed_id == 0:
            return True
        return update.effective_user.id == self.allowed_id

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            await update.message.reply_text("⛔ Unauthorized.")
            return
        await update.message.reply_text(
            "👋 Hi! I'm your AI Agent.\n\n"
            "Send me any command:\n"
            "• find file report.pdf\n"
            "• list apps\n"
            "• open chrome\n"
            "• system info / disk info\n"
            "• running processes\n"
            "• go to youtube.com\n"
            "• shut down / restart / sleep\n"
            "• Or just chat normally!\n\n"
            "Send voice messages too 🎙️\n"
            "/screenshot /status /help"
        )

    async def _cmd_screenshot(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update): return
        self.log_cb("📸 Screenshot via Telegram")
        img_bytes = self.pc_agent.take_screenshot_bytes()
        await update.message.reply_photo(photo=img_bytes, caption="📸 Current screen")

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update): return
        current = self.persona.get_current_persona()
        msg = (f"📊 Agent Status\n"
               f"Persona: {current['emoji']} {current['name']}\n"
               f"Tasks: {len(self.task_mgr.tasks)}\n"
               f"Memory: {len(self.memory)//2} exchanges")
        await update.message.reply_text(msg)

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update): return
        await update.message.reply_text(
            "📋 COMMANDS:\n"
            "• find file [name]\n"
            "• list files in [folder]\n"
            "• list apps / open [app]\n"
            "• system info / disk info\n"
            "• running processes\n"
            "• kill process [name]\n"
            "• go to [url]\n"
            "• search for [query]\n"
            "• volume up/down/mute\n"
            "• battery info / network info\n"
            "• shut down / restart / sleep\n"
            "• Or just chat with the AI!\n"
            "/screenshot /status"
        )

    async def _handle_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        text = update.message.text.strip()
        self.log_cb(f"📱 Telegram: {text}")

        def _run():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = self.pc_agent.handle(text)
            if result is not None:
                loop.run_until_complete(update.message.reply_text(result))
            else:
                loop.run_until_complete(update.message.reply_text("💭 Thinking..."))
                try:
                    prompt  = build_agent_prompt(self.persona, self.task_mgr)
                    answer  = answer_question(text, self.llm, prompt, self.memory)
                    current = self.persona.get_current_persona()
                    self.memory.append(f"User: {text}")
                    self.memory.append(f"{current['name']}: {answer}")
                    loop.run_until_complete(
                        update.message.reply_text(f"{current['emoji']} {answer}")
                    )
                except Exception as e:
                    loop.run_until_complete(
                        update.message.reply_text(f"❌ AI Error: {str(e)}")
                    )
            loop.close()

        threading.Thread(target=_run, daemon=True).start()

    async def _handle_voice(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update): return
        self.log_cb("🎙️ Voice message via Telegram")
        await update.message.reply_text("🎙️ Transcribing...")

        def _run():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                vf    = loop.run_until_complete(update.message.voice.get_file())
                audio = loop.run_until_complete(vf.download_as_bytearray())
                text  = self.voice.transcribe(bytes(audio), ".ogg")
                loop.run_until_complete(
                    update.message.reply_text(f"🎙️ Heard: \"{text}\"\n⚡ Processing...")
                )

                result = self.pc_agent.handle(text)
                if result is not None:
                    loop.run_until_complete(update.message.reply_text(result))
                else:
                    prompt  = build_agent_prompt(self.persona, self.task_mgr)
                    answer  = answer_question(text, self.llm, prompt, self.memory)
                    current = self.persona.get_current_persona()
                    self.memory.append(f"User: {text}")
                    self.memory.append(f"{current['name']}: {answer}")
                    loop.run_until_complete(
                        update.message.reply_text(f"{current['emoji']} {answer}")
                    )
            except Exception as e:
                loop.run_until_complete(
                    update.message.reply_text(f"❌ Error: {str(e)}")
                )
            finally:
                loop.close()

        threading.Thread(target=_run, daemon=True).start()

    def start(self):
        if not self.token:
            self.log_cb("⚠️ No TELEGRAM_TOKEN — bot disabled")
            return

        def _run():
            import asyncio
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.app = ApplicationBuilder().token(self.token).build()
            self.app.add_handler(CommandHandler("start",      self._cmd_start))
            self.app.add_handler(CommandHandler("screenshot", self._cmd_screenshot))
            self.app.add_handler(CommandHandler("status",     self._cmd_status))
            self.app.add_handler(CommandHandler("help",       self._cmd_help))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
            self.app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))
            self.log_cb("📱 Telegram bot started!")
            self.running = True
            self.app.run_polling(stop_signals=None)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        if self.app:
            try: self.app.stop()
            except: pass
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


# ============= MAIN XP APPLICATION =============
class AgentXPApp:
    def __init__(self, root: tk.Tk):
        self.root        = root
        self.root.title("AI Agent — XP Edition + Telegram")
        self.root.configure(bg=XP["bg"])
        self.root.geometry("900x660")
        self.root.minsize(700, 520)

        style = ttk.Style()
        style.theme_use("classic")

        # Agent state
        self.persona      = AgentPersona()
        self.task_manager = TaskManager()
        self.image_gen    = AgentImageGenerator(self.persona)
        self.memory       = []
        self.llm          = None
        self.is_thinking  = False
        self.pc_agent     = LocalPCAgent(log_cb=lambda m: self.root.after(0, lambda msg=m: self._system_msg(msg, "pc")))
        self.voice        = VoiceTranscriber(log_cb=lambda m: self.root.after(0, lambda msg=m: self._system_msg(msg)))
        self.tg_bot       = None

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
        tk.Label(tb, text="  🖥️  AI Agent — XP + Telegram Edition",
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
            ("File",    [("Exit", self.root.destroy)]),
            ("View",    [("Clear Chat", self._clear_chat), ("Status", self._show_status)]),
            ("Personas",[( f"{v['emoji']} {v['name']}", lambda k=k: self._switch_persona(k))
                          for k, v in AgentPersona.PERSONAS.items()]),
            ("PC",      [("System Info",      lambda: self._local_cmd("system info")),
                         ("Disk Info",         lambda: self._local_cmd("disk info")),
                         ("Running Processes", lambda: self._local_cmd("running processes")),
                         ("Installed Apps",    lambda: self._local_cmd("list apps")),
                         ("Network Info",      lambda: self._local_cmd("network info")),
                         ("Battery Info",      lambda: self._local_cmd("battery info")),
                         ("Take Screenshot",   lambda: self._local_cmd("screenshot")),
                         ("Shut Down PC",      lambda: self._local_cmd("shutdown")),
                         ("Restart PC",        lambda: self._local_cmd("restart")),
                         ("Sleep",             lambda: self._local_cmd("sleep"))]),
            ("Telegram",[("Start Bot",  self._start_telegram),
                         ("Stop Bot",   self._stop_telegram),
                         ("Bot Status", self._telegram_status)]),
            ("Help",    [("Commands", self._show_help)]),
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
                menu.add_command(label=item_label, command=item_cmd)
            mb["menu"] = menu

    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=XP["bg"], width=195)
        left.pack(side="left", fill="y", padx=(6, 3), pady=6)
        left.pack_propagate(False)

        # Persona
        pf = tk.LabelFrame(left, text=" 🤖 Persona ", font=(XP["font"], 9, "bold"),
                            bg=XP["bg"], fg="#000000", relief="groove", bd=2)
        pf.pack(fill="x", pady=(0, 5))
        self.persona_label = tk.Label(pf, text="", font=(XP["font"], 10, "bold"),
                                       bg=XP["bg"], fg=XP["user_fg"], wraplength=165, justify="center")
        self.persona_label.pack(pady=4)
        self.persona_desc = tk.Label(pf, text="", font=(XP["font"], 8),
                                      bg=XP["bg"], fg="#555", wraplength=165, justify="center")
        self.persona_desc.pack(pady=(0, 4))

        # Switch
        sf = tk.LabelFrame(left, text=" Switch Persona ", font=(XP["font"], 9, "bold"),
                            bg=XP["bg"], fg="#000000", relief="groove", bd=2)
        sf.pack(fill="x", pady=(0, 5))
        for key, val in AgentPersona.PERSONAS.items():
            tk.Button(sf, text=f"{val['emoji']} {val['name']}", font=(XP["font"], 8),
                      bg=XP["btn"], fg="#000", relief="raised", bd=2, anchor="w",
                      activebackground=XP["highlight"], activeforeground="white",
                      cursor="hand2", command=lambda k=key: self._switch_persona(k)
                      ).pack(fill="x", padx=4, pady=2)

        # Quick PC
        qf = tk.LabelFrame(left, text=" ⚡ Quick PC ", font=(XP["font"], 9, "bold"),
                            bg=XP["bg"], fg="#000000", relief="groove", bd=2)
        qf.pack(fill="x", pady=(0, 5))
        for label, cmd in [
            ("📊 System Info",  "system info"),
            ("💾 Disk Info",    "disk info"),
            ("⚙️ Processes",   "running processes"),
            ("📋 Apps List",    "list apps"),
            ("📸 Screenshot",   "screenshot"),
            ("⚠️ Shut Down",   "shutdown"),
            ("🔄 Restart",      "restart"),
            ("💤 Sleep",        "sleep"),
        ]:
            tk.Button(qf, text=label, font=(XP["font"], 8),
                      bg=XP["btn"], fg="#000", relief="raised", bd=2, anchor="w",
                      activebackground=XP["highlight"], activeforeground="white",
                      cursor="hand2", command=lambda c=cmd: self._local_cmd(c)
                      ).pack(fill="x", padx=4, pady=1)

        # Telegram
        tgf = tk.LabelFrame(left, text=" 📱 Telegram ", font=(XP["font"], 9, "bold"),
                             bg=XP["bg"], fg="#000000", relief="groove", bd=2)
        tgf.pack(fill="x", pady=(0, 5))
        self.tg_status_label = tk.Label(tgf, text="⬤ Offline", font=(XP["font"], 9, "bold"),
                                         bg=XP["bg"], fg="#CC0000")
        self.tg_status_label.pack(pady=4)
        xp_button(tgf, "▶ Start Bot", self._start_telegram, width=16, bg="#4CAF50").pack(padx=4, pady=2)
        xp_button(tgf, "■ Stop Bot",  self._stop_telegram,  width=16, bg="#D32F2F").pack(padx=4, pady=(2, 6))

        # Image Gen
        igf = tk.LabelFrame(left, text=" 🎨 Image Gen ", font=(XP["font"], 9, "bold"),
                             bg=XP["bg"], fg="#000000", relief="groove", bd=2)
        igf.pack(fill="x")
        self.img_entry = tk.Entry(igf, font=(XP["font"], 8), relief="sunken", bd=2)
        self.img_entry.pack(fill="x", padx=4, pady=(4, 2))
        self.img_entry.insert(0, "describe an image...")
        self.img_entry.bind("<FocusIn>", lambda e: self.img_entry.delete(0, "end")
                             if self.img_entry.get() == "describe an image..." else None)
        self.img_entry.bind("<Return>", lambda e: self._generate_image())
        xp_button(igf, "🎨 Generate", self._generate_image, width=16).pack(padx=4, pady=(2, 4))

        self._update_persona_panel()

    def _build_chat_panel(self, parent):
        right = tk.Frame(parent, bg=XP["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(3, 6), pady=6)

        cf = tk.LabelFrame(right, text=" 💬 Conversation ", font=(XP["font"], 9, "bold"),
                            bg=XP["bg"], fg="#000000", relief="groove", bd=2)
        cf.pack(fill="both", expand=True)

        self.chat_area = tk.Text(cf, state="disabled", wrap="word",
                                  bg=XP["chat_bg"], fg="#000",
                                  font=(XP["font"], 9), relief="sunken",
                                  bd=2, padx=6, pady=6, cursor="arrow")
        sb = ttk.Scrollbar(cf, command=self.chat_area.yview)
        self.chat_area.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.chat_area.pack(fill="both", expand=True, padx=4, pady=4)

        self.chat_area.tag_configure("user",     foreground=XP["user_fg"],  font=(XP["font"], 9, "bold"))
        self.chat_area.tag_configure("bot",      foreground=XP["bot_fg"],   font=(XP["font"], 9))
        self.chat_area.tag_configure("telegram", foreground=XP["tg_fg"],    font=(XP["font"], 9, "bold"))
        self.chat_area.tag_configure("system",   foreground=XP["system_fg"],font=(XP["font"], 8, "italic"))
        self.chat_area.tag_configure("error",    foreground=XP["error_fg"], font=(XP["font"], 9, "bold"))
        self.chat_area.tag_configure("thinking", foreground="#B8860B",      font=(XP["font"], 8, "italic"))
        self.chat_area.tag_configure("pc",       foreground="#8B4513",      font=(XP["font"], 9))

        # Input
        inf = tk.Frame(right, bg=XP["bg"])
        inf.pack(fill="x", pady=(4, 0))
        self.input_entry = tk.Entry(inf, font=(XP["font"], 10),
                                     relief="sunken", bd=2, bg=XP["chat_bg"])
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.input_entry.bind("<Return>", lambda e: self._send_message())
        xp_button(inf, "Send ➤", self._send_message, width=10,
                  bg=XP["highlight"]).pack(side="right")
        self.input_entry.configure(fg="#000")

    def _build_taskbar(self, parent):
        tb = tk.Frame(parent, bg=XP["taskbar_bg"], height=30)
        tb.pack(fill="x", side="bottom")
        tb.pack_propagate(False)
        tk.Label(tb, text="  🪟 start ", font=(XP["font"], 9, "bold"),
                 bg="#4CAF50", fg="white", relief="raised", bd=2).pack(side="left", padx=4, pady=3)
        tk.Frame(tb, bg=XP["taskbar_btn"], width=2).pack(side="left", fill="y", pady=3)
        tk.Label(tb, text="  🖥️ AI Agent + Telegram  ",
                 font=(XP["font"], 9), bg=XP["taskbar_btn"],
                 fg="white", relief="raised", bd=2).pack(side="left", padx=4, pady=3)
        self.clock_label = tk.Label(tb, text="", font=(XP["font"], 9),
                                     bg=XP["taskbar_bg"], fg="white")
        self.clock_label.pack(side="right", padx=8)
        self._update_clock()
        self.status_var = tk.StringVar(value="Initializing...")
        tk.Label(tb, textvariable=self.status_var, font=(XP["font"], 8),
                 bg=XP["taskbar_bg"], fg="#AACCFF").pack(side="right", padx=16)

    def _update_clock(self):
        self.clock_label.config(text=datetime.now().strftime("%I:%M %p"))
        self.root.after(10000, self._update_clock)

    # ─── AGENT INIT ─────────────────────────────────────────────────────────

    def _init_agent(self):
        def _load():
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                self.root.after(0, lambda: self._system_msg("❌ GROQ_API_KEY missing in .env!", "error"))
                return
            self.root.after(0, lambda: self._system_msg("📚 Loading documents..."))
            try:
                DATA_DIR.mkdir(exist_ok=True)
                docs   = load_pdf_documents(DATA_DIR)
                chunks = split_documents(docs)
                create_vectorstore(chunks)
                self.root.after(0, lambda: self._system_msg(f"✅ RAG ready — {len(docs)} pages"))
            except FileNotFoundError:
                self.root.after(0, lambda: self._system_msg("⚠️ No PDFs — RAG disabled", "system"))
            self.llm = ChatGroq(model=GROQ_MODEL, api_key=groq_key, temperature=0.7)
            self.root.after(0, lambda: self._system_msg("✅ Agent ready! Type a message or start Telegram bot."))
            self.root.after(0, lambda: self.status_var.set("Ready"))

        threading.Thread(target=_load, daemon=True).start()
        self._system_msg("🖥️ Welcome to AI Agent XP + Telegram Edition!")
        self._system_msg("⚡ PC commands are instant — AI chat uses Groq online.", "system")
        self._system_msg("Initializing...", "thinking")

    # ─── CHAT HELPERS ────────────────────────────────────────────────────────

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
        current = self.persona.get_current_persona()
        tg_stat = "Online ✅" if (self.tg_bot and self.tg_bot.running) else "Offline ❌"
        self._system_msg(
            f"📊 Persona: {current['emoji']} {current['name']} | "
            f"Interactions: {len(self.memory)//2} | Telegram: {tg_stat}"
        )

    def _show_help(self):
        self._system_msg(
            "💡 COMMANDS:\n"
            "  ⚡ INSTANT (local, no API):\n"
            "  • find file report.pdf\n"
            "  • list files in downloads\n"
            "  • list apps / open chrome\n"
            "  • system info / disk info / battery info\n"
            "  • running processes / kill process chrome\n"
            "  • go to youtube.com\n"
            "  • search for python tutorials\n"
            "  • volume up / down / mute\n"
            "  • screenshot\n"
            "  • shut down / restart / sleep\n\n"
            "  🌐 ONLINE features:\n"
            "  • Just chat → AI responds (Groq)\n"
            "  • generate image: a sunset → Pollinations\n"
            "  • task: buy milk → task added\n"
        )

    # ─── ACTIONS ─────────────────────────────────────────────────────────────

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

        # Image generation (online)
        if user_input.lower().startswith("generate image:"):
            self._run_image_gen(user_input[15:].strip())
            return

        # Try local PC command first (instant ⚡)
        local_result = self.pc_agent.handle(user_input)
        if local_result is not None:
            self._append_chat("⚡ PC:", local_result, "pc")
            self.status_var.set("Ready")
            return

        # Fall back to AI chat (online)
        if not self.llm:
            self._system_msg("⚠️ AI not ready yet...", "error")
            return

        self.is_thinking = True
        self.status_var.set("Thinking...")
        self._system_msg("💭 Thinking...", "thinking")

        def _think():
            try:
                prompt  = build_agent_prompt(self.persona, self.task_manager)
                answer  = answer_question(user_input, self.llm, prompt, self.memory)
                current = self.persona.get_current_persona()
                self.memory.append(f"User: {user_input}")
                self.memory.append(f"{current['name']}: {answer}")
                if len(self.memory) > 50:
                    self.memory = self.memory[-50:]
                self.root.after(0, lambda: self._append_chat(f"{current['emoji']} {current['name']}:", answer, "bot"))
                self.root.after(0, lambda: self.status_var.set("Ready"))
            except Exception as e:
                self.root.after(0, lambda: self._system_msg(f"⚠️ Error: {e}", "error"))
            finally:
                self.is_thinking = False

        threading.Thread(target=_think, daemon=True).start()

    def _local_cmd(self, cmd: str):
        """Run a local PC command and show result."""
        self.status_var.set("⚡ Running locally...")

        def _run():
            result = self.pc_agent.handle(cmd)
            if result is None:
                result = "❓ Command not recognized locally."
            if "screenshot saved:" in result.lower():
                path = result.split(": ", 1)[-1].strip()
                try:
                    img_bytes = Path(path).read_bytes()
                    self.root.after(0, lambda: self._show_image_window(img_bytes, "Screenshot"))
                except:
                    pass
            self.root.after(0, lambda: self._append_chat("⚡ PC:", result, "pc"))
            self.root.after(0, lambda: self.status_var.set("Ready"))

        threading.Thread(target=_run, daemon=True).start()

    def _switch_persona(self, key):
        if self.persona.switch_persona(key):
            current = self.persona.get_current_persona()
            self._system_msg(f"✨ Switched to: {current['emoji']} {current['name']}")
            self._update_persona_panel()

    def _update_persona_panel(self):
        current = self.persona.get_current_persona()
        self.persona_label.config(text=f"{current['emoji']} {current['name']}")
        self.persona_desc.config(text=current['description'])

    # ─── TELEGRAM ────────────────────────────────────────────────────────────

    def _start_telegram(self):
        if not TELEGRAM_TOKEN:
            messagebox.showerror("Error", "TELEGRAM_TOKEN not found in .env file!\nAdd it and restart.")
            return
        if self.tg_bot and self.tg_bot.running:
            self._system_msg("📱 Telegram bot already running!", "system")
            return
        if not self.llm:
            self._system_msg("⚠️ Wait for agent to initialize first.", "error")
            return

        self.tg_bot = TelegramBotManager(
            token=TELEGRAM_TOKEN,
            allowed_id=ALLOWED_USER_ID,
            llm_ref=self.llm,
            persona_ref=self.persona,
            task_ref=self.task_manager,
            pc_agent_ref=self.pc_agent,
            voice_ref=self.voice,
            memory_ref=self.memory,
            log_cb=lambda m: self.root.after(0, lambda: self._system_msg(f"📱 {m}", "telegram")),
        )
        self.tg_bot.start()
        self.tg_status_label.config(text="⬤ Online", fg="#4CAF50")
        self._system_msg("📱 Telegram bot started! Open Telegram and send /start to your bot.", "telegram")

    def _stop_telegram(self):
        if self.tg_bot:
            self.tg_bot.stop()
            self.tg_status_label.config(text="⬤ Offline", fg="#CC0000")
            self._system_msg("📱 Telegram bot stopped.", "system")

    def _telegram_status(self):
        if self.tg_bot and self.tg_bot.running:
            self._system_msg("📱 Telegram bot is ONLINE ✅", "telegram")
        else:
            self._system_msg("📱 Telegram bot is OFFLINE ❌", "system")

    # ─── IMAGE ───────────────────────────────────────────────────────────────

    def _generate_image(self):
        prompt = self.img_entry.get().strip()
        if not prompt or prompt == "describe an image...":
            self._system_msg("⚠️ Enter an image description first.", "error")
            return
        self.img_entry.delete(0, "end")
        self._run_image_gen(prompt)

    def _run_image_gen(self, prompt):
        self.status_var.set("Generating image...")
        self._system_msg(f"🎨 Generating: {prompt}")

        def _gen():
            data = self.image_gen.generate_image(prompt)
            if data:
                path = self.image_gen.save_image(data, prompt)
                self.root.after(0, lambda: self._system_msg(f"✅ Saved: {path}"))
                self.root.after(0, lambda: self._show_image_window(data, prompt))
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
        tk.Label(tb, text=f"  🖼️  {title[:50]}", font=(XP["font"], 9, "bold"),
                 bg=XP["title_bar"], fg=XP["title_fg"]).pack(side="left", pady=4)
        tk.Button(tb, text="✕", font=("Tahoma", 8, "bold"), bg="#C0302C",
                  fg="white", relief="raised", bd=1, width=3,
                  cursor="hand2", command=win.destroy).pack(side="right", pady=3, padx=2)
        try:
            img   = Image.open(BytesIO(image_data))
            img.thumbnail((600, 500))
            photo = ImageTk.PhotoImage(img)
            lbl   = tk.Label(win, image=photo, bg=XP["bg"])
            lbl.image = photo
            lbl.pack(padx=10, pady=10)
        except Exception as e:
            tk.Label(win, text=f"Could not display: {e}", bg=XP["bg"],
                     font=(XP["font"], 9)).pack(pady=20)


# ============= ENTRY POINT =============
def main():
    root = tk.Tk()
    root.withdraw()
    app = AgentXPApp(root)
    root.deiconify()
    root.mainloop()

if __name__ == "__main__":
    main()