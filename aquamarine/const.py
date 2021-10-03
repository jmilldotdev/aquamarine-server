import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
DOTENV_PATH = BASE_DIR / ".env"

load_dotenv(DOTENV_PATH)

OBSIDIAN_VAULT_PATH = os.environ.get("OBSIDIAN_VAULT_PATH")
IMAGE_VAULT_PATH = os.environ.get("IMAGE_VAULT_PATH")
