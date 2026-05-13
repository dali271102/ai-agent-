import os
from pathlib import Path
from typing import Optional, Dict, List
import requests
from dotenv import load_dotenv
from datetime import datetime
import random
import time
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import json
import base64

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

# ============= AGENT PERSONAS =============
class AgentPersona:
    """Different personas the AI agent can adopt"""
    
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
        """Try multiple image generation sources"""
        
        print(f"   🎨 Attempting to generate: {prompt}")
        
        # Try multiple methods in order
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
        """Try Pollinations.ai (free, no API key)"""
        try:
            persona_style = self.persona.get_current_persona()['image_style']
            enhanced_prompt = f"{prompt}, {style or persona_style}, high quality, detailed"
            
            encoded_prompt = requests.utils.quote(enhanced_prompt)
            # Using different parameters for better success rate
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true"
            
            print("   📡 Trying Pollinations.ai...")
            response = requests.get(url, timeout=45)
            
            if response.status_code == 200 and len(response.content) > 5000:
                # Verify it's actually an image
                if response.content.startswith(b'\xff\xd8') or response.content.startswith(b'\x89PNG'):
                    print("   ✅ Image generated via Pollinations!")
                    return response.content
                else:
                    print("   ⚠️ Invalid image data received")
            else:
                print(f"   ⚠️ Pollinations returned {response.status_code}")
                
        except Exception as e:
            print(f"   ⚠️ Pollinations error: {str(e)[:50]}")
        
        return None
    
    def _try_lexica(self, prompt: str, style: str = "") -> Optional[bytes]:
        """Try Lexica.art API (free, no key required)"""
        try:
            search_url = f"https://lexica.art/api/v1/search?q={requests.utils.quote(prompt)}"
            print("   📡 Trying Lexica.art...")
            
            response = requests.get(search_url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('images') and len(data['images']) > 0:
                    # Get the first image URL
                    image_url = data['images'][0]['src']
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        print("   ✅ Image found via Lexica!")
                        return img_response.content
                        
        except Exception as e:
            print(f"   ⚠️ Lexica error: {str(e)[:50]}")
        
        return None
    
    def _try_placeholder(self, prompt: str, style: str = "") -> Optional[bytes]:
        """Create a text-based placeholder image with the prompt"""
        try:
            print("   📝 Creating information image...")
            
            # Create a larger, more informative image
            img = Image.new('RGB', (1024, 1024), color=(20, 25, 45))
            draw = ImageDraw.Draw(img)
            
            # Try to load a font
            try:
                # Try different font sizes
                font_large = ImageFont.truetype("arial.ttf", 36)
                font_medium = ImageFont.truetype("arial.ttf", 24)
                font_small = ImageFont.truetype("arial.ttf", 18)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Draw a border
            draw.rectangle([10, 10, 1014, 1014], outline=(100, 100, 150), width=3)
            
            # Title
            title = "🎨 IMAGE GENERATION REQUEST"
            bbox = draw.textbbox((0, 0), title, font=font_large)
            title_width = bbox[2] - bbox[0]
            draw.text((512 - title_width//2, 100), title, fill=(255, 200, 100), font=font_large)
            
            # Prompt
            prompt_text = f"Prompt: {prompt}"
            bbox = draw.textbbox((0, 0), prompt_text, font=font_medium)
            prompt_width = bbox[2] - bbox[0]
            draw.text((512 - prompt_width//2, 250), prompt_text, fill=(200, 200, 255), font=font_medium)
            
            # Style info
            persona_style = self.persona.get_current_persona()['image_style']
            style_text = f"Style: {persona_style}"
            bbox = draw.textbbox((0, 0), style_text, font=font_small)
            style_width = bbox[2] - bbox[0]
            draw.text((512 - style_width//2, 350), style_text, fill=(150, 150, 200), font=font_small)
            
            # Instructions
            instructions = [
                "⚠️ Image generation API temporarily unavailable",
                "Try again later or use a different prompt",
                "",
                "💡 Tips for better results:",
                "• Be more specific in your prompt",
                "• Try: 'generate image: a realistic photo of...'",
                "• Check your internet connection"
            ]
            
            y = 500
            for instruction in instructions:
                bbox = draw.textbbox((0, 0), instruction, font=font_small)
                text_width = bbox[2] - bbox[0]
                draw.text((512 - text_width//2, y), instruction, fill=(180, 180, 220), font=font_small)
                y += 35
            
            # Save to bytes
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            print("   ℹ️ Created information image (API unavailable)")
            return img_buffer.getvalue()
            
        except Exception as e:
            print(f"   ❌ Placeholder creation failed: {e}")
        
        return None
    
    def save_image(self, image_bytes: bytes, prompt: str) -> Path:
        """Save generated image"""
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)

def answer_question(question, llm, prompt, memory):
    history = "\n".join(memory[-10:])
    final_prompt = prompt.format(history=history, question=question)
    response = llm.invoke(final_prompt).content
    return response

# ============= ENHANCED AGENT PROMPT =============
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

# ============= MAIN AGENT FUNCTION =============
def main():
    load_dotenv()
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found")
        print("Please add GROQ_API_KEY to your .env file")
        return
    
    # Initialize agent systems
    persona = AgentPersona()
    task_manager = TaskManager()
    image_gen = AgentImageGenerator(persona)
    
    # Load PDF documents (optional)
    print("\n📚 Loading documents...")
    try:
        DATA_DIR.mkdir(exist_ok=True)
        documents = load_pdf_documents(DATA_DIR)
        print(f"✅ Loaded {len(documents)} pages")
        
        chunks = split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        vectorstore = create_vectorstore(chunks)
        print("✅ Vector store ready")
        
    except FileNotFoundError:
        print("⚠️ No PDF files found in 'data/' folder - RAG disabled")
    
    # Initialize LLM
    print("\n🤖 Initializing AI Agent...")
    llm = ChatGroq(model=GROQ_MODEL, api_key=groq_api_key, temperature=0.7)
    
    print("\n" + "="*70)
    print("🤖 AI AGENT - Multi-Persona Assistant 🤖")
    print("="*70)
    
    # Show initial persona
    current = persona.get_current_persona()
    print(f"✨ Current Persona: {current['emoji']} {current['name']} - {current['description']}")
    
    print("\n💡 COMMANDS:")
    print("   • 'help' - Show all commands")
    print("   • 'switch to [persona]' - Change my role")
    print("   • 'personas' - List all personas")
    print("   • 'generate image: [description]' - Create an image")
    print("   • 'task: [description]' - Add a task")
    print("   • 'status' - Show agent status")
    print("   • 'quit' - Exit")
    print("="*70 + "\n")
    
    memory = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == "quit":
                print(f"\n{persona.get_current_persona()['emoji']} Goodbye! Come back anytime!")
                break
            
            # Help command
            if user_input.lower() == "help":
                print("\n📋 AVAILABLE COMMANDS:")
                print("   • 'switch to assistant' - Professional helper")
                print("   • 'switch to friend' - Casual conversation")
                print("   • 'switch to mentor' - Wise guide")
                print("   • 'switch to creative' - Brainstorming partner")
                print("   • 'switch to analyst' - Data-driven thinker")
                print("   • 'personas' - List all personas")
                print("   • 'task: [description]' - Add a task")
                print("   • 'tasks' - Show current tasks")
                print("   • 'clear tasks' - Clear all tasks")
                print("   • 'generate image: [description]' - Create an image")
                print("   • 'status' - Show current agent status")
                continue
            
            # List personas
            if user_input.lower() == "personas":
                print(persona.list_personas())
                continue
            
            # Status command
            if user_input.lower() == "status":
                current = persona.get_current_persona()
                print(f"\n📊 AGENT STATUS:")
                print(f"   Persona: {current['emoji']} {current['name']}")
                print(f"   Description: {current['description']}")
                print(f"   Interactions: {len(memory)}")
                print(f"   Tasks: {len(task_manager.tasks)}")
                continue
            
            # Tasks commands
            if user_input.lower() == "tasks":
                if task_manager.tasks:
                    print("\n📋 Current Tasks:")
                    for i, task in enumerate(task_manager.tasks, 1):
                        print(f"   {i}. [{task['priority']}] {task['task']} - {task['status']}")
                else:
                    print("📋 No active tasks")
                continue
            
            if user_input.lower() == "clear tasks":
                print(task_manager.clear_tasks())
                continue
            
            # Switch persona
            if user_input.lower().startswith("switch to "):
                persona_name = user_input.lower().replace("switch to ", "").strip()
                if persona.switch_persona(persona_name):
                    current = persona.get_current_persona()
                    print(f"\n✨ Switched to: {current['emoji']} {current['name']}")
                    print(f"   {current['description']}")
                else:
                    print(f"❌ Persona '{persona_name}' not found. Type 'personas' to see options.")
                continue
            
            # Add task
            if user_input.lower().startswith("task:"):
                task_desc = user_input[5:].strip()
                print(task_manager.add_task(task_desc))
                continue
            
            # ============= IMAGE GENERATION (FIXED) =============
            if user_input.lower().startswith("generate image:"):
                image_prompt = user_input[15:].strip()
                if not image_prompt:
                    print("❌ Please provide an image description.")
                    print("   Example: 'generate image: a futuristic city with flying cars'")
                    continue
                
                print(f"\n🎨 Generating image for: '{image_prompt}'")
                print("⏳ This may take 15-30 seconds...")
                
                # Show progress animation
                start_time = time.time()
                
                image_data = image_gen.generate_image(image_prompt)
                
                if image_data:
                    filename = image_gen.save_image(image_data, image_prompt)
                    elapsed = time.time() - start_time
                    print(f"\n✅ Image generated and saved in {elapsed:.1f} seconds!")
                    print(f"📁 Location: {filename}")
                    
                    # Try to open the image
                    try:
                        os.startfile(filename)
                        print("🖼️ Image opened automatically!")
                    except:
                        print(f"💡 You can find your image in the 'agent_images' folder")
                else:
                    print("\n❌ Failed to generate image. The placeholder image shows your request.")
                    print("💡 Tips for better results:")
                    print("   • Be more specific in your description")
                    print("   • Try: 'generate image: realistic photo of...'")
                    print("   • Check your internet connection")
                continue
            
            # Normal conversation
            print(f"\n💭 {persona.get_current_persona()['name']} is thinking...")
            prompt = build_agent_prompt(persona, task_manager)
            answer = answer_question(user_input, llm, prompt, memory)
            
            current_persona = persona.get_current_persona()
            print(f"\n{current_persona['emoji']} {answer}\n")
            
            # Update memory
            memory.append(f"User: {user_input}")
            memory.append(f"{current_persona['name']}: {answer}")
            
            if len(memory) > 50:
                memory = memory[-50:]
                
        except KeyboardInterrupt:
            print("\n\n🤖 Agent shutting down. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            print("Continuing...")

if __name__ == "__main__":
    main()