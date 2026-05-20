"""
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
