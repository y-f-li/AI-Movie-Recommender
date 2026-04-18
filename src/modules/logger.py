import time
from pathlib import Path


class Logger:
    def __init__(self, log_name: str = "agent.log"):
        self.base_dir = Path(__file__).resolve().parents[2]
        self.log_dir = self.base_dir / "evaluation_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / log_name

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def log(self, event_type: str, content: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{self._timestamp()}] [{event_type.upper()}] {content}\n")

    def log_message(self, message: str):
        self.log("MESSAGE", f"Incoming: {message}")

    def log_response(self, response: str):
        self.log("RESPONSE", f"Response: {response}")

    def log_reaction(self, reaction: str, message_ordinal: int):
        self.log("REACTION", f"Reaction: '{reaction}' on message #{message_ordinal}")
