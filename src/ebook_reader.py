
from pathlib import Path

def read_ebook(ebook_path: Path) -> str:
    """Reads the content of an ebook file."""
    return ebook_path.read_text(encoding="utf-8")
