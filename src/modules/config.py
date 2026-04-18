from dataclasses import dataclass
from pathlib import Path
import os

SRC_DIR = os.path.dirname(__file__)
PRIVATE_DATA_DIR = os.path.join(SRC_DIR, "private_data")
os.makedirs(PRIVATE_DATA_DIR, exist_ok=True)

@dataclass
class Config:
    """Holds runtime configuration for the agent."""
    kb_path: str
    ratings_path: str
    host_url: str
    username: str
    password: str
    private_data_dir: str = PRIVATE_DATA_DIR
    artifacts_dir: str | None = None
