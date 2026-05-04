"""
╔══════════════════════════════════════════════════════════════╗
║                    UTILITY FUNCTIONS                         ║
║   File reading · URL fetching · PDF extraction · Batch IO   ║
╚══════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── File Reading ──────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".log", ".tex"}


def read_text_file(filepath: str | Path, encoding: str = "utf-8") -> str:
    """Read any text file with automatic encoding fallback."""
    fp = Path(filepath)
    for enc in (encoding, "latin-1", "cp1252", "utf-16"):
        try:
            return fp.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # Last resort: ignore errors
    return fp.read_bytes().decode("utf-8", errors="replace")


def read_pdf(filepath: str | Path) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz) if available,
    falling back to pdfplumber, then pypdf.
    """
    fp = Path(filepath)
    text = ""

    # Try PyMuPDF (fastest)
    try:
        import fitz  # PyMuPDF
        with fitz.open(str(fp)) as doc:
            text = "\n".join(page.get_text() for page in doc)
        logger.info(f"PDF read via PyMuPDF: {fp.name}")
        return text
    except ImportError:
        pass

    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(str(fp)) as pdf:
            text = "\n".join(
                (page.extract_text() or "") for page in pdf.pages
            )
        logger.info(f"PDF read via pdfplumber: {fp.name}")
        return text
    except ImportError:
        pass

    # Try pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(fp))
        text = "\n".join(
            (page.extract_text() or "") for page in reader.pages
        )
        logger.info(f"PDF read via pypdf: {fp.name}")
        return text
    except ImportError:
        pass

    logger.warning(f"No PDF library found. Install: pip install PyMuPDF")
    return ""


def load_documents_from_directory(
    directory: str | Path,
    extensions: Tuple[str, ...] = (".txt", ".md"),
    recursive: bool = False,
    include_pdf: bool = False,
) -> Dict[str, str]:
    """
    Load all documents from a directory.
    Returns {filename: content} dict.
    """
    d = Path(directory)
    docs: Dict[str, str] = {}

    if include_pdf:
        extensions = extensions + (".pdf",)

    pattern = "**/*" if recursive else "*"
    for fp in sorted(d.glob(pattern)):
        if fp.is_file() and fp.suffix.lower() in extensions:
            try:
                if fp.suffix.lower() == ".pdf":
                    content = read_pdf(fp)
                else:
                    content = read_text_file(fp)
                if content.strip():
                    docs[fp.name] = content
                    logger.debug(f"Loaded: {fp.name} ({len(content)} chars)")
            except Exception as e:
                logger.warning(f"Failed to read {fp}: {e}")

    return docs


# ─── URL Fetching ──────────────────────────────────────────────────

def fetch_url_text(url: str, timeout: int = 10) -> str:
    """
    Download and extract plain text from a URL.
    Strips HTML tags for web pages.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "PlagiarismChecker/2.0 (Python)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            encoding = resp.headers.get_content_charset("utf-8")
            html = raw.decode(encoding, errors="replace")
    except urllib.error.URLError as e:
        logger.error(f"URL fetch failed: {url} — {e}")
        return ""

    # Strip HTML tags
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>",  " ", text,  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Hashing / Fingerprinting ──────────────────────────────────────

def document_fingerprint(text: str) -> str:
    """SHA-256 fingerprint of a document (for caching / deduplication)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def are_duplicates(text1: str, text2: str) -> bool:
    """Exact duplicate check via hashing."""
    return document_fingerprint(text1) == document_fingerprint(text2)


# ─── Statistics ────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?]+", text))


def document_stats(text: str) -> dict:
    """Return basic statistics about a document."""
    return {
        "characters":       len(text),
        "words":            word_count(text),
        "sentences":        sentence_count(text),
        "unique_words":     len(set(text.lower().split())),
        "avg_word_length":  round(
            sum(len(w) for w in text.split()) / max(word_count(text), 1), 2
        ),
        "fingerprint":      document_fingerprint(text)[:12] + "…",
    }


# ─── JSON Helpers ──────────────────────────────────────────────────

def save_json(data: dict, path: str) -> None:
    """Save data as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    """Load JSON from file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Sample Document Generator ─────────────────────────────────────

SAMPLE_DOCUMENTS = {
    "sample_original.txt": """
Artificial intelligence is transforming the modern world in profound ways.
Machine learning algorithms can now recognize patterns in data that were
previously invisible to human analysts. From healthcare diagnostics to
financial forecasting, AI systems are being deployed across every major
industry. The development of large language models has opened new frontiers
in natural language processing, enabling computers to understand and generate
human-like text with remarkable fluency.
""",
    "sample_paraphrased.txt": """
AI is changing our world significantly. Algorithms based on machine learning
are now capable of identifying data patterns that human analysts could not
previously detect. AI systems are deployed in healthcare, finance, and many
other sectors. Large language models have created new possibilities for NLP,
allowing machines to read and write text that sounds surprisingly human.
""",
    "sample_unrelated.txt": """
The history of ancient Rome spans over a thousand years of political, cultural,
and military evolution. Beginning as a small city-state on the Italian peninsula,
Rome grew to encompass much of Europe, North Africa, and the Near East.
The Roman Republic gave way to the Roman Empire under Augustus Caesar, whose
reign ushered in a long period of relative peace known as the Pax Romana.
Roman engineering achievements such as aqueducts, roads, and amphitheaters
still stand today as testaments to their civilizational genius.
""",
    "sample_partial_copy.txt": """
There is no doubt that artificial intelligence is transforming the modern world
in profound ways. The development of large language models has opened new
frontiers in natural language processing. However, the ethical implications of
these technologies must also be carefully considered. Issues of bias, privacy,
transparency, and accountability are increasingly central to public discourse
about the responsible deployment of AI systems.
""",
}


def create_sample_documents(directory: str = "./samples") -> None:
    """Write sample documents for testing."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for name, content in SAMPLE_DOCUMENTS.items():
        (d / name).write_text(content.strip(), encoding="utf-8")
    print(f"✅  Sample documents created in: {d.resolve()}")
