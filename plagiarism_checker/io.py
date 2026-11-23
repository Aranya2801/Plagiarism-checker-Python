"""Reading documents from disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable

log = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".tex"}
SUPPORTED = TEXT_EXTENSIONS | {".pdf", ".docx"}


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):          # utf-8 BOM (Notepad likes these)
        return raw[3:].decode("utf-8", errors="replace")
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):     # utf-16 with BOM
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # Most likely a Windows-1252 file. cp1252 can fail on a few bytes,
        # latin-1 never does.
        try:
            return raw.decode("cp1252")
        except UnicodeDecodeError:
            return raw.decode("latin-1")


def read_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
        with fitz.open(str(path)) as doc:
            return "\n".join(page.get_text() for page in doc)
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    except ImportError:
        raise ImportError(f"can't read {path.name}: pip install pymupdf (or pypdf)") from None


def read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        raise ImportError(f"can't read {path.name}: pip install python-docx") from None
    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)


def read_document(path: Path | str) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    return read_text(path)


def iter_files(folder: Path, recursive: bool = False) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in sorted(folder.glob(pattern)):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            yield p


def load_directory(folder: Path | str, recursive: bool = False) -> Dict[str, str]:
    """{name: text} for every supported file. With recursive=True names are
    relative paths, so essays/a.txt and drafts/a.txt don't collide."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    out = {}
    for p in iter_files(folder, recursive):
        name = p.relative_to(folder).as_posix() if recursive else p.name
        try:
            text = read_document(p)
        except (ImportError, OSError, ValueError) as e:
            log.warning("skipping %s: %s", p, e)
            continue
        if text.strip():
            out[name] = text
        else:
            log.warning("skipping %s: no text", p)
    return out
