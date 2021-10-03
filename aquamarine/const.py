import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
DOTENV_PATH = BASE_DIR / ".env"

load_dotenv(DOTENV_PATH)

OBSIDIAN_VAULT_PATH = os.environ.get("OBSIDIAN_VAULT_PATH")
IMAGE_VAULT_PATH = os.environ.get("IMAGE_VAULT_PATH")

TEXT_EXTENSIONS = [
    ".txt",
    ".md",
]

IMAGE_EXTENSIONS = [
    ".bmp",
    ".gif",
    ".ief",
    ".jpg",
    ".jpe",
    ".jpeg",
    ".png",
    ".svg",
    ".tiff",
    ".tif",
    ".ico",
    ".ras",
    ".pnm",
    ".pbm",
    ".pgm",
    ".ppm",
    ".rgb",
    ".xbm",
    ".xpm",
    ".xwd",
    ".cgm",
    ".g3",
    ".jp2",
    ".ktx",
    ".pict",
    ".pic",
    ".pct",
    ".btif",
    ".sgi",
    ".svgz",
    ".psd",
    ".uvi",
    ".uvvi",
    ".uvg",
    ".uvvg",
    ".djvu",
    ".djv",
    ".dwg",
    ".dxf",
    ".fbs",
    ".fpx",
    ".fst",
    ".mmr",
    ".rlc",
    ".mdi",
    ".wdp",
    ".npx",
    ".wbmp",
    ".xif",
    ".webp",
    ".3ds",
    ".cmx",
    ".fh",
    ".fhc",
    ".fh4",
    ".fh5",
    ".fh7",
    ".pntg",
    ".pnt",
    ".mac",
    ".sid",
    ".pcx",
    ".qtif",
    ".qti",
    ".tga",
]
