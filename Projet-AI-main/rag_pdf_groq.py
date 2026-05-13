import os
from pathlib import Path
from typing import Optional, Dict, List
import requests
from dotenv import load_dotenv
from datetime import datetime
import random
import time
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageTk
import json
import base64
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============= CONFIGURATION =============
DATA_DIR = Path("data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3

# ============= XP COLOR PALETTE =============
XP = {
    "bg":           "#ECE9D8",
    "bg_dark":      "#D4D0C8",
    "btn":          "#D4D0C8",
    "btn_active":   "#316AC5",
    "title_bar":    "#0A246A",
    "title_fg":     "#FFFFFF",
    "border_light": "#FFFFFF",
    "border_dark":  "#808080",
    "border_darker":"#404040",
    "highlight":    "#316AC5",
    "highlight_fg": "#FFFFFF",
    "chat_bg":      "#FFFFFF",
    "user_fg":      "#0A246A",
    "bot_fg":       "#1A5C1A",
    "system_fg":    "#808080",
    "error_fg":     "#CC0000",
    "font":         "Tahoma",
    "font_size":    9,
    "taskbar_bg":   "#245EDC",
    "taskbar_btn":  "#3A7FDE",
    "green_btn":    "#4CAF50",
    "red_btn":      "#D32F2F",
}

# ============= AGENT PERSONAS =============
class AgentPersona:
    PERSONAS = {
        "assistant": {
            "name": "Assistant",
            "emoji": "🤖",
            "description": "Helpful, efficient, and professional AI assistant",
            "traits": "Be helpful, concise, and efficient. Focus on getting tasks done. Provide clear answers and solutions.",
            "response_style": "professional, clear, solution-oriented",
            "image_style": "professional, clean, minimalist"
        },
        "friend": {
            "name": "Friend",
            "emoji": "👋",
            "description": "Casual, friendly conversational companion",
            "traits": "Be warm, empathetic, and conversational. Show genuine interest. Use casual language.",
            "response_style": "conversational, warm, engaging",
            "image_style": "friendly, warm, casual"
        },
        "mentor": {
            "name": "Mentor",
            "emoji": "🎓",
            "description": "Wise, experienced guide and teacher",
            "traits": "Be knowledgeable, patient, and encouraging. Provide guidance and wisdom.",
            "response_style": "wise, instructive, supportive",
            "image_style": "professional, authoritative, calm"
        },
        "creative": {
            "name": "Creative Partner",
            "emoji": "🎨",
            "description": "Creative brainstorming partner",
            "traits": "Be imaginative, open-minded, and playful. Generate creative ideas and solutions.",
            "response_style": "creative, inspiring, imaginative",
            "image_style": "artistic, colorful, creative"
        },
        "analyst": {
            "name": "Analyst",
            "emoji": "📊",
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

PERSONALITY TRAITS:
{persona['traits']}

RESPONSE STYLE:
{persona['response_style']}

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

# ============= FIXED IMAGE GENERATOR WITH MULTIPLE SOURCES =============
class AgentImageGenerator:
    def __init__(self, persona: AgentPersona):
        self.persona = persona

    def generate_image(self, prompt: str, style: str = "") -> Optional[bytes]:
        print(f"   🎨 Attempting to generate: {prompt}")
        methods = [
            self._try_pollinations,
            self._try_lexica,
            self._try_placeholder,
        ]
        for method in methods:
            result = method(prompt, style)
            if result:
                return result
            time.sleep(1)
        return None

    def _try_pollinations(self, prompt: str, style: str = "") -> Optional[bytes]:
        try:
            persona_style = self.persona.get_current_persona()['image_style']
            enhanced_prompt = f"{prompt}, {style or persona_style}, high quality, detailed"
            encoded_prompt = requests.utils.quote(enhanced_prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true"
            print("   📡 Trying Pollinations.ai...")
            response = requests.get(url, timeout=45)
            if response.status_code == 200 and len(response.content) > 5000:
                if response.content.startswith(b'\xff\xd8') or response.content.startswith(b'\x89PNG'):
                    print("   ✅ Image generated via Pollinations!")
                    return response.content
        except Exception as e:
            print(f"   ⚠️ Pollinations error: {str(e)[:50]}")
        return None

    def _try_lexica(self, prompt: str, style: str = "") -> Optional[bytes]:
        try:
            search_url = f"https://lexica.art/api/v1/search?q={requests.utils.quote(prompt)}"
            print("   📡 Trying Lexica.art...")
            response = requests.get(search_url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('images') and len(data['images']) > 0:
                    image_url = data['images'][0]['src']
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        print("   ✅ Image found via Lexica!")
                        return img_response.content
        except Exception as e:
            print(f"   ⚠️ Lexica error: {str(e)[:50]}")
        return None

    def _try_placeholder(self, prompt: str, style: str = "") -> Optional[bytes]:
        try:
            img = Image.new('RGB', (1024, 1024), color=(20, 25, 45))
            draw = ImageDraw.Draw(img)
            try:
                font_large = ImageFont.truetype("arial.ttf", 36)
                font_medium = ImageFont.truetype("arial.ttf", 24)
                font_small = ImageFont.truetype("arial.ttf", 18)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
            draw.rectangle([10, 10, 1014, 1014], outline=(100, 100, 150), width=3)
            title = "IMAGE GENERATION REQUEST"
            draw.text((200, 100), title, fill=(255, 200, 100), font=font_large)
            draw.text((100, 250), f"Prompt: {prompt}", fill=(200, 200, 255), font=font_medium)
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            return img_buffer.getvalue()
        except Exception as e:
            print(f"   ❌ Placeholder creation failed: {e}")
        return None

    def save_image(self, image_bytes: bytes, prompt: str) -> Path:
        img_dir = Path("agent_images")
        img_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = img_dir / f"image_{safe_prompt}_{timestamp}.png"
        with open(filename, "wb") as f:
            f.write(image_bytes)
        return filename

# ============= RAG FUNCTIONS =============
def load_pdf_documents(data_dir: Path):
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("No PDF files found")
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
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
    history = "\n".join(memory[-10:])
    final_prompt = prompt.format(history=history, question=question)
    response = llm.invoke(final_prompt).content
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
- Switch between different personas

{task_context}

IMPORTANT RULES:
1. Always respond in character as {persona_info['name']}
2. Be helpful and engaging
3. For image generation, user will type 'generate image: [description]'

Conversation history:
{{history}}

User: {{question}}

{persona_info['emoji']} {persona_info['name']}:"""
    return PromptTemplate.from_template(template)


# ============= XP WIDGET HELPERS =============
def xp_relief_frame(parent, **kwargs):
    """A sunken beveled XP-style frame"""
    f = tk.Frame(parent, relief="sunken", bd=2,
                 bg=kwargs.pop("bg", XP["bg"]), **kwargs)
    return f

def xp_button(parent, text, command, width=12, bg=None):
    """Classic XP-style button"""
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        font=(XP["font"], XP["font_size"]),
        bg=bg or XP["btn"],
        fg="#000000",
        relief="raised",
        bd=2,
        activebackground=XP["highlight"],
        activeforeground=XP["highlight_fg"],
        cursor="hand2",
        width=width,
    )
    return btn


# ============= WINDOWS XP TITLE BAR =============
class XPTitleBar(tk.Frame):
    def __init__(self, parent, title, icon="🖥️", **kwargs):
        super().__init__(parent, bg=XP["title_bar"], height=28, **kwargs)
        self.pack_propagate(False)

        # Gradient effect using a canvas
        self.canvas = tk.Canvas(self, bg=XP["title_bar"], highlightthickness=0, height=28)
        self.canvas.pack(fill="both", expand=True)

        # Draw gradient manually
        self.canvas.bind("<Configure>", self._draw_gradient)

        # Icon + Title
        self.title_label = tk.Label(
            self.canvas,
            text=f" {icon} {title}",
            font=(XP["font"], 9, "bold"),
            bg=XP["title_bar"],
            fg=XP["title_fg"],
        )
        self.canvas.create_window(5, 14, window=self.title_label, anchor="w")

        # Window control buttons frame
        btn_frame = tk.Frame(self.canvas, bg=XP["title_bar"])
        self.canvas.create_window(0, 14, window=btn_frame, anchor="e", tags="btns")

        self._min_btn = tk.Button(btn_frame, text="—", font=("Tahoma", 7, "bold"),
                                   bg="#4A80D0", fg="white", relief="raised", bd=1,
                                   width=2, height=1, cursor="hand2",
                                   command=lambda: parent.winfo_toplevel().iconify())
        self._min_btn.pack(side="left", padx=1, pady=2)

        self._close_btn = tk.Button(btn_frame, text="✕", font=("Tahoma", 7, "bold"),
                                     bg="#C0302C", fg="white", relief="raised", bd=1,
                                     width=2, height=1, cursor="hand2",
                                     command=parent.winfo_toplevel().destroy)
        self._close_btn.pack(side="left", padx=1, pady=2)

        self.canvas.bind("<Configure>", self._on_resize)

    def _draw_gradient(self, event=None):
        pass

    def _on_resize(self, event):
        self.canvas.coords("btns") if self.canvas.find_withtag("btns") else None
        self.canvas.itemconfig("btns", tags="btns")
        # Reposition the buttons to right side
        w = event.width
        self.canvas.delete("btn_win")
        self.canvas.create_window(w - 5, 14, window=self.canvas.winfo_children()[-1]
                                   if self.canvas.winfo_children() else None, anchor="e", tags="btn_win")


# ============= TASK DIALOG (XP STYLE) =============
class XPTaskDialog(tk.Toplevel):
    def __init__(self, parent, task_manager: TaskManager, refresh_cb):
        super().__init__(parent)
        self.task_manager = task_manager
        self.refresh_cb = refresh_cb
        self.title("")
        self.resizable(False, False)
        self.configure(bg=XP["bg"])
        self.grab_set()

        # XP title bar simulation via built-in title
        self.title("📋 Task Manager")

        tk.Label(self, text="📋 Task Manager", font=(XP["font"], 10, "bold"),
                 bg=XP["bg"]).pack(pady=(10, 5), padx=15, anchor="w")

        # Task list
        list_frame = xp_relief_frame(self, bg=XP["chat_bg"])
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.task_list = tk.Text(list_frame, width=40, height=8,
                                  font=(XP["font"], 9), bg=XP["chat_bg"],
                                  fg="#000000", state="disabled", wrap="word")
        self.task_list.pack(fill="both", expand=True, padx=4, pady=4)

        # Add task
        tk.Label(self, text="New Task:", font=(XP["font"], 9),
                 bg=XP["bg"]).pack(padx=10, anchor="w")

        entry_frame = tk.Frame(self, bg=XP["bg"])
        entry_frame.pack(fill="x", padx=10, pady=5)

        self.task_entry = tk.Entry(entry_frame, font=(XP["font"], 9),
                                    relief="sunken", bd=2)
        self.task_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.task_entry.bind("<Return>", lambda e: self._add_task())

        xp_button(entry_frame, "Add", self._add_task, width=8).pack(side="right")

        # Buttons
        btn_frame = tk.Frame(self, bg=XP["bg"])
        btn_frame.pack(pady=8)
        xp_button(btn_frame, "Clear All", self._clear_tasks, width=10).pack(side="left", padx=5)
        xp_button(btn_frame, "Close", self.destroy, width=8).pack(side="left", padx=5)

        self._refresh_list()

    def _refresh_list(self):
        self.task_list.config(state="normal")
        self.task_list.delete("1.0", "end")
        if self.task_manager.tasks:
            for i, t in enumerate(self.task_manager.tasks, 1):
                self.task_list.insert("end", f"{i}. [{t['priority']}] {t['task']} — {t['status']}\n")
        else:
            self.task_list.insert("end", "No tasks yet.")
        self.task_list.config(state="disabled")

    def _add_task(self):
        text = self.task_entry.get().strip()
        if text:
            self.task_manager.add_task(text)
            self.task_entry.delete(0, "end")
            self._refresh_list()
            self.refresh_cb(f"✅ Task added: {text}")

    def _clear_tasks(self):
        self.task_manager.clear_tasks()
        self._refresh_list()
        self.refresh_cb("🧹 All tasks cleared")


# ============= MAIN XP CHAT APPLICATION =============
class AgentXPApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Agent — Windows XP Edition")
        self.root.configure(bg=XP["bg"])
        self.root.geometry("780x620")
        self.root.minsize(650, 500)

        # Apply classic theme
        style = ttk.Style()
        style.theme_use("classic")
        style.configure("TCombobox", font=(XP["font"], XP["font_size"]))

        # Agent state
        self.persona = AgentPersona()
        self.task_manager = TaskManager()
        self.image_gen = AgentImageGenerator(self.persona)
        self.memory = []
        self.llm = None
        self.is_thinking = False

        self._build_ui()
        self._init_agent()

    # ─── UI CONSTRUCTION ────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Outer window chrome (XP border) ──
        outer = tk.Frame(self.root, bg=XP["border_dark"], bd=2, relief="raised")
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Fake XP title bar ──
        title_bar = tk.Frame(outer, height=30, bg=XP["title_bar"])
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)

        # Gradient-ish left section
        left_title = tk.Frame(title_bar, bg=XP["title_bar"])
        left_title.pack(side="left", fill="y")

        tk.Label(left_title, text="  🖥️  AI Agent — XP Edition",
                 font=(XP["font"], 9, "bold"),
                 bg=XP["title_bar"], fg=XP["title_fg"]).pack(side="left", pady=4)

        # Window buttons
        for txt, cmd, color in [
            ("_", lambda: self.root.iconify(), "#5A8FE0"),
            ("□", lambda: None, "#5A8FE0"),
            ("✕", self.root.destroy, "#C0302C"),
        ]:
            tk.Button(title_bar, text=txt, font=("Tahoma", 8, "bold"),
                      bg=color, fg="white", relief="raised", bd=1,
                      width=3, cursor="hand2", command=cmd).pack(side="right", pady=3, padx=1)

        # ── Menu bar ──
        self._build_menu_bar(outer)

        # ── Main content area ──
        main = tk.Frame(outer, bg=XP["bg"])
        main.pack(fill="both", expand=True)

        # ── Left panel (persona + status) ──
        self._build_left_panel(main)

        # ── Right panel (chat) ──
        self._build_chat_panel(main)

        # ── Taskbar at bottom ──
        self._build_taskbar(outer)

    def _build_menu_bar(self, parent):
        menubar_frame = tk.Frame(parent, bg=XP["bg_dark"], relief="raised", bd=1)
        menubar_frame.pack(fill="x")

        menus = [
            ("File", [("Exit", self.root.destroy)]),
            ("View", [("Clear Chat", self._clear_chat), ("Show Status", self._show_status)]),
            ("Personas", [(f"{v['emoji']} {v['name']}", lambda k=k: self._switch_persona(k))
                          for k, v in AgentPersona.PERSONAS.items()]),
            ("Tasks", [("Open Task Manager", self._open_task_manager),
                       ("Clear All Tasks", lambda: self._system_msg(self.task_manager.clear_tasks()))]),
            ("Help", [("Commands", self._show_help)]),
        ]

        for label, items in menus:
            mb = tk.Menubutton(menubar_frame, text=label,
                               font=(XP["font"], 9),
                               bg=XP["bg_dark"], fg="#000000",
                               relief="flat", padx=8, pady=2,
                               activebackground=XP["highlight"],
                               activeforeground="white")
            mb.pack(side="left")
            menu = tk.Menu(mb, tearoff=False,
                           font=(XP["font"], 9),
                           bg=XP["bg"], fg="#000000",
                           activebackground=XP["highlight"],
                           activeforeground="white")
            for item_label, item_cmd in items:
                menu.add_command(label=item_label, command=item_cmd)
            mb["menu"] = menu

    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=XP["bg"], width=185)
        left.pack(side="left", fill="y", padx=(6, 3), pady=6)
        left.pack_propagate(False)

        # Persona box
        persona_frame = tk.LabelFrame(left, text=" 🤖 Persona ",
                                       font=(XP["font"], 9, "bold"),
                                       bg=XP["bg"], fg="#000000",
                                       relief="groove", bd=2)
        persona_frame.pack(fill="x", pady=(0, 6))

        self.persona_label = tk.Label(persona_frame,
                                       text="",
                                       font=(XP["font"], 10, "bold"),
                                       bg=XP["bg"], fg=XP["user_fg"],
                                       wraplength=160, justify="center")
        self.persona_label.pack(pady=6, padx=4)

        self.persona_desc = tk.Label(persona_frame, text="",
                                      font=(XP["font"], 8),
                                      bg=XP["bg"], fg="#555555",
                                      wraplength=160, justify="center")
        self.persona_desc.pack(pady=(0, 6), padx=4)

        # Persona buttons
        switch_frame = tk.LabelFrame(left, text=" Switch Persona ",
                                      font=(XP["font"], 9, "bold"),
                                      bg=XP["bg"], fg="#000000",
                                      relief="groove", bd=2)
        switch_frame.pack(fill="x", pady=(0, 6))

        for key, val in AgentPersona.PERSONAS.items():
            btn = tk.Button(switch_frame,
                            text=f"{val['emoji']} {val['name']}",
                            font=(XP["font"], 8),
                            bg=XP["btn"], fg="#000000",
                            relief="raised", bd=2,
                            activebackground=XP["highlight"],
                            activeforeground="white",
                            cursor="hand2",
                            anchor="w",
                            command=lambda k=key: self._switch_persona(k))
            btn.pack(fill="x", padx=4, pady=2)

        # Tasks box
        tasks_frame = tk.LabelFrame(left, text=" 📋 Tasks ",
                                     font=(XP["font"], 9, "bold"),
                                     bg=XP["bg"], fg="#000000",
                                     relief="groove", bd=2)
        tasks_frame.pack(fill="x", pady=(0, 6))

        xp_button(tasks_frame, "📋 Task Manager",
                  self._open_task_manager, width=16).pack(padx=4, pady=4)

        self.task_count_label = tk.Label(tasks_frame,
                                          text="Tasks: 0",
                                          font=(XP["font"], 8),
                                          bg=XP["bg"], fg="#555555")
        self.task_count_label.pack(pady=(0, 4))

        # Image gen box
        img_frame = tk.LabelFrame(left, text=" 🎨 Image Gen ",
                                   font=(XP["font"], 9, "bold"),
                                   bg=XP["bg"], fg="#000000",
                                   relief="groove", bd=2)
        img_frame.pack(fill="x")

        self.img_entry = tk.Entry(img_frame, font=(XP["font"], 8),
                                   relief="sunken", bd=2)
        self.img_entry.pack(fill="x", padx=4, pady=(4, 2))
        self.img_entry.insert(0, "describe an image...")
        self.img_entry.bind("<FocusIn>", lambda e: self.img_entry.delete(0, "end")
                             if self.img_entry.get() == "describe an image..." else None)
        self.img_entry.bind("<Return>", lambda e: self._generate_image())

        xp_button(img_frame, "🎨 Generate", self._generate_image, width=16).pack(padx=4, pady=(2, 4))

        self._update_persona_panel()

    def _build_chat_panel(self, parent):
        right = tk.Frame(parent, bg=XP["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(3, 6), pady=6)

        # Chat area
        chat_frame = tk.LabelFrame(right, text=" 💬 Conversation ",
                                    font=(XP["font"], 9, "bold"),
                                    bg=XP["bg"], fg="#000000",
                                    relief="groove", bd=2)
        chat_frame.pack(fill="both", expand=True)

        self.chat_area = tk.Text(
            chat_frame,
            state="disabled",
            wrap="word",
            bg=XP["chat_bg"],
            fg="#000000",
            font=(XP["font"], 9),
            relief="sunken",
            bd=2,
            padx=6,
            pady=6,
            cursor="arrow",
        )
        scrollbar = ttk.Scrollbar(chat_frame, command=self.chat_area.yview)
        self.chat_area.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.chat_area.pack(fill="both", expand=True, padx=4, pady=4)

        # Text tags for colors
        self.chat_area.tag_configure("user",   foreground=XP["user_fg"],  font=(XP["font"], 9, "bold"))
        self.chat_area.tag_configure("bot",    foreground=XP["bot_fg"],   font=(XP["font"], 9))
        self.chat_area.tag_configure("system", foreground=XP["system_fg"],font=(XP["font"], 8, "italic"))
        self.chat_area.tag_configure("error",  foreground=XP["error_fg"], font=(XP["font"], 9, "bold"))
        self.chat_area.tag_configure("thinking",foreground="#B8860B",     font=(XP["font"], 8, "italic"))

        # Input area
        input_frame = tk.Frame(right, bg=XP["bg"])
        input_frame.pack(fill="x", pady=(4, 0))

        self.input_entry = tk.Entry(
            input_frame,
            font=(XP["font"], 10),
            relief="sunken",
            bd=2,
            bg=XP["chat_bg"],
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.input_entry.bind("<Return>", lambda e: self._send_message())

        send_btn = xp_button(input_frame, "Send ➤", self._send_message, width=10)
        send_btn.configure(bg="#316AC5", fg="white",
                           activebackground="#1A4A9A", activeforeground="white")
        send_btn.pack(side="right")

    def _build_taskbar(self, parent):
        taskbar = tk.Frame(parent, bg=XP["taskbar_bg"], height=30)
        taskbar.pack(fill="x", side="bottom")
        taskbar.pack_propagate(False)

        # Start button-ish
        tk.Label(taskbar, text="  🪟 start ",
                 font=(XP["font"], 9, "bold"),
                 bg="#4CAF50", fg="white",
                 relief="raised", bd=2,
                 cursor="hand2").pack(side="left", padx=4, pady=3)

        # Separator
        tk.Frame(taskbar, bg=XP["taskbar_btn"], width=2).pack(side="left", fill="y", pady=3)

        # Active window button
        self.taskbar_label = tk.Label(taskbar,
                                       text="  🤖 AI Agent — XP Edition  ",
                                       font=(XP["font"], 9),
                                       bg=XP["taskbar_btn"], fg="white",
                                       relief="raised", bd=2)
        self.taskbar_label.pack(side="left", padx=4, pady=3)

        # Clock on the right
        self.clock_label = tk.Label(taskbar, text="",
                                     font=(XP["font"], 9),
                                     bg=XP["taskbar_bg"], fg="white")
        self.clock_label.pack(side="right", padx=8)
        self._update_clock()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(taskbar, textvariable=self.status_var,
                               font=(XP["font"], 8),
                               bg=XP["taskbar_bg"], fg="#AACCFF")
        status_bar.pack(side="right", padx=16)

    # ─── CLOCK ──────────────────────────────────────────────────────────────

    def _update_clock(self):
        now = datetime.now().strftime("%I:%M %p")
        self.clock_label.config(text=now)
        self.root.after(10000, self._update_clock)

    # ─── AGENT INIT ─────────────────────────────────────────────────────────

    def _init_agent(self):
        def _load():
            load_dotenv()
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                self.root.after(0, lambda: self._system_msg(
                    "❌ GROQ_API_KEY not found in .env file!", tag="error"))
                return

            self.root.after(0, lambda: self._system_msg("📚 Loading documents..."))
            try:
                DATA_DIR.mkdir(exist_ok=True)
                documents = load_pdf_documents(DATA_DIR)
                chunks = split_documents(documents)
                self.vectorstore = create_vectorstore(chunks)
                self.root.after(0, lambda: self._system_msg(
                    f"✅ Loaded {len(documents)} pages, {len(chunks)} chunks — RAG ready!"))
            except FileNotFoundError:
                self.root.after(0, lambda: self._system_msg(
                    "⚠️ No PDFs found in 'data/' — RAG disabled", tag="system"))

            self.llm = ChatGroq(model=GROQ_MODEL, api_key=groq_api_key, temperature=0.7)
            self.root.after(0, lambda: self._system_msg("✅ AI Agent ready! Type a message below."))
            self.root.after(0, lambda: self.status_var.set("Agent Ready"))

        threading.Thread(target=_load, daemon=True).start()
        self._system_msg("🖥️ Welcome to AI Agent — Windows XP Edition!")
        self._system_msg("Initializing... please wait.", tag="thinking")

    # ─── CHAT HELPERS ────────────────────────────────────────────────────────

    def _append_chat(self, prefix: str, text: str, tag: str = "bot"):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", f"{prefix} ", tag)
        self.chat_area.insert("end", f"{text}\n\n", tag)
        self.chat_area.see("end")
        self.chat_area.config(state="disabled")

    def _system_msg(self, text: str, tag: str = "system"):
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
        msg = (f"📊 STATUS — Persona: {current['emoji']} {current['name']} | "
               f"Interactions: {len(self.memory)//2} | Tasks: {len(self.task_manager.tasks)}")
        self._system_msg(msg)

    def _show_help(self):
        help_text = (
            "💡 COMMANDS:\n"
            "  • Type normally to chat\n"
            "  • 'generate image: ...' to create an image\n"
            "  • 'task: ...' to add a task\n"
            "  • Use Personas menu or left panel to switch roles\n"
            "  • Use Tasks menu to manage tasks\n"
        )
        self._system_msg(help_text)

    # ─── ACTIONS ─────────────────────────────────────────────────────────────

    def _send_message(self):
        if self.is_thinking:
            return

        user_input = self.input_entry.get().strip()
        if not user_input:
            return
        self.input_entry.delete(0, "end")

        # Task shortcut
        if user_input.lower().startswith("task:"):
            task_desc = user_input[5:].strip()
            result = self.task_manager.add_task(task_desc)
            self._append_chat("You:", user_input, "user")
            self._system_msg(result)
            self._update_task_count()
            return

        # Image generation shortcut
        if user_input.lower().startswith("generate image:"):
            prompt = user_input[15:].strip()
            self._append_chat("You:", user_input, "user")
            self._run_image_gen(prompt)
            return

        # Normal chat
        if not self.llm:
            self._system_msg("⚠️ Agent not ready yet, please wait...", tag="error")
            return

        self._append_chat("You:", user_input, "user")
        self.is_thinking = True
        self.status_var.set("Thinking...")
        self._system_msg(f"💭 {self.persona.get_current_persona()['name']} is thinking...", tag="thinking")

        def _think():
            try:
                prompt = build_agent_prompt(self.persona, self.task_manager)
                answer = answer_question(user_input, self.llm, prompt, self.memory)
                current = self.persona.get_current_persona()
                self.memory.append(f"User: {user_input}")
                self.memory.append(f"{current['name']}: {answer}")
                if len(self.memory) > 50:
                    self.memory = self.memory[-50:]
                self.root.after(0, lambda: self._append_chat(
                    f"{current['emoji']} {current['name']}:", answer, "bot"))
                self.root.after(0, lambda: self.status_var.set("Ready"))
            except Exception as e:
                self.root.after(0, lambda: self._system_msg(f"⚠️ Error: {e}", tag="error"))
                self.root.after(0, lambda: self.status_var.set("Error"))
            finally:
                self.is_thinking = False

        threading.Thread(target=_think, daemon=True).start()

    def _switch_persona(self, key: str):
        if self.persona.switch_persona(key):
            current = self.persona.get_current_persona()
            self._system_msg(f"✨ Switched to: {current['emoji']} {current['name']} — {current['description']}")
            self._update_persona_panel()
            self.status_var.set(f"Persona: {current['name']}")

    def _update_persona_panel(self):
        current = self.persona.get_current_persona()
        self.persona_label.config(text=f"{current['emoji']} {current['name']}")
        self.persona_desc.config(text=current['description'])

    def _update_task_count(self):
        count = len(self.task_manager.tasks)
        self.task_count_label.config(text=f"Tasks: {count}")

    def _open_task_manager(self):
        def refresh_cb(msg):
            self._system_msg(msg)
            self._update_task_count()
        XPTaskDialog(self.root, self.task_manager, refresh_cb)

    def _generate_image(self):
        prompt = self.img_entry.get().strip()
        if not prompt or prompt == "describe an image...":
            self._system_msg("⚠️ Please enter an image description in the left panel.", tag="error")
            return
        self.img_entry.delete(0, "end")
        self._system_msg(f"🎨 Generating image: '{prompt}'...")
        self._run_image_gen(prompt)

    def _run_image_gen(self, prompt: str):
        self.status_var.set("Generating image...")

        def _gen():
            image_data = self.image_gen.generate_image(prompt)
            if image_data:
                filename = self.image_gen.save_image(image_data, prompt)
                self.root.after(0, lambda: self._system_msg(
                    f"✅ Image saved: {filename}"))
                self.root.after(0, lambda: self._show_image_window(image_data, prompt))
            else:
                self.root.after(0, lambda: self._system_msg(
                    "❌ Image generation failed.", tag="error"))
            self.root.after(0, lambda: self.status_var.set("Ready"))

        threading.Thread(target=_gen, daemon=True).start()

    def _show_image_window(self, image_data: bytes, prompt: str):
        """Show generated image in an XP-style popup window"""
        win = tk.Toplevel(self.root)
        win.title(f"🖼️ {prompt[:40]}")
        win.configure(bg=XP["bg"])
        win.resizable(True, True)

        # Title bar
        tb = tk.Frame(win, bg=XP["title_bar"], height=28)
        tb.pack(fill="x")
        tb.pack_propagate(False)
        tk.Label(tb, text=f"  🖼️  {prompt[:50]}",
                 font=(XP["font"], 9, "bold"),
                 bg=XP["title_bar"], fg=XP["title_fg"]).pack(side="left", pady=4)
        tk.Button(tb, text="✕", font=("Tahoma", 8, "bold"),
                  bg="#C0302C", fg="white", relief="raised", bd=1,
                  width=3, cursor="hand2", command=win.destroy).pack(side="right", pady=3, padx=2)

        try:
            img = Image.open(BytesIO(image_data))
            img.thumbnail((500, 500))
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(win, image=photo, bg=XP["bg"])
            label.image = photo
            label.pack(padx=10, pady=10)
        except Exception as e:
            tk.Label(win, text=f"Could not display image:\n{e}",
                     bg=XP["bg"], font=(XP["font"], 9)).pack(pady=20)

        tk.Label(win, text=f"Prompt: {prompt}",
                 font=(XP["font"], 8), bg=XP["bg"],
                 fg="#555555", wraplength=480).pack(pady=(0, 6))


# ============= ENTRY POINT =============
def main():
    root = tk.Tk()
    root.withdraw()  # hide while building

    app = AgentXPApp(root)

    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()