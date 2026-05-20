"""
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
                r = subprocess.run("powershell Get-Clipboard", shell=True,
                                   capture_output=True, text=True, timeout=5)
                text = r.stdout.strip()
                return f"📋 Clipboard:\n{text}" if text else "📋 Clipboard is empty"
            except Exception as e:
                return f"❌ {e}"
        if "write" in cmd or "copy" in cmd:
            for trigger in ["clipboard write ", "clipboard copy "]:
                if trigger in cmd:
                    text = command[command.lower().index(trigger) + len(trigger):]
                    escaped = text.replace("'", "\'")
                    subprocess.run(
                        f"powershell Set-Clipboard -Value \'{escaped}\'",
                        shell=True, timeout=5
                    )
                    return f"📋 Copied to clipboard: {text[:60]}"
        if "clear" in cmd:
            subprocess.run("powershell Set-Clipboard -Value $null", shell=True, timeout=5)
            return "📋 Clipboard cleared"
        return None
