import io
from pathlib import Path

from pypdf import PdfReader


class UnsupportedFileType(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    """Return plain text from a .pdf or .txt upload."""
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        return data.decode("utf-8", errors="ignore")
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise UnsupportedFileType(f"Unsupported file type: {ext or 'unknown'}")
