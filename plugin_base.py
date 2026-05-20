from typing import Optional, Dict

class Plugin:
    name = 'unnamed'
    description = ''
    version = '1.0'
    author = ''
    def __init__(self, agent_ref): self.agent = agent_ref
    def on_load(self): pass
    def on_unload(self): pass
    def handle(self, command: str) -> Optional[str]: return None
    def get_commands(self) -> Dict[str, str]: return {}