"""
╔══════════════════════════════════════════════════════════════╗
║                PLAGIARISM DETECTION ENGINE                   ║
║  Algorithms: TF-IDF | Cosine | Jaccard | LSH | n-gram       ║
╚══════════════════════════════════════════════════════════════╝

Core detection engine supporting multiple similarity algorithms
with configurable thresholds and detailed reporting.
"""

from __future__ import annotations

import os
import re
import math
import time
import hashlib
import logging
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────────────────────

@dataclass
class DocumentPair:
    """Represents a pair of compared documents with similarity scores."""
    doc1: str
    doc2: str
    cosine_score: float = 0.0
    jaccard_score: float = 0.0
    ngram_score: float = 0.0
    combined_score: float = 0.0
    verdict: str = "ORIGINAL"
    highlighted_sections: List[str] = field(default_factory=list)
    word_overlap: List[str] = field(default_factory=list)
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "document_1": self.doc1,
            "document_2": self.doc2,
            "scores": {
                "cosine_similarity": round(self.cosine_score * 100, 2),
                "jaccard_similarity": round(self.jaccard_score * 100, 2),
                "ngram_similarity": round(self.ngram_score * 100, 2),
                "combined_score": round(self.combined_score * 100, 2),
            },
            "verdict": self.verdict,
            "common_phrases": self.highlighted_sections[:5],
            "word_overlap_count": len(self.word_overlap),
            "execution_time_ms": round(self.execution_time * 1000, 2),
        }


@dataclass
class CheckResult:
    """Full result from a plagiarism check run."""
    total_documents: int = 0
    total_pairs: int = 0
    plagiarised_pairs: int = 0
    pairs: List[DocumentPair] = field(default_factory=list)
    timestamp: str = ""
    duration: float = 0.0

    @property
    def plagiarism_rate(self) -> float:
        if self.total_pairs == 0:
            return 0.0
        return (self.plagiarised_pairs / self.total_pairs) * 100


# ─────────────────────────────────────────────────────────────
#  Text Preprocessing
# ─────────────────────────────────────────────────────────────

class TextPreprocessor:
    """
    Advanced text preprocessing pipeline.
    Handles cleaning, tokenization, stopword removal & stemming.
    """

    STOP_WORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "s", "t", "can", "will", "just", "don", "should",
        "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
        "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma",
        "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
        "won", "wouldn"
    }

    def __init__(self, remove_stopwords: bool = True, min_word_length: int = 3):
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length

    def clean(self, text: str) -> str:
        """Remove noise: URLs, emails, special characters, extra whitespace."""
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Remove emails
        text = re.sub(r"\S+@\S+", " ", text)
        # Remove special chars but keep letters and spaces
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def tokenize(self, text: str) -> List[str]:
        """Split text into meaningful tokens."""
        cleaned = self.clean(text)
        tokens = cleaned.split()
        tokens = [t for t in tokens if len(t) >= self.min_word_length]
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOP_WORDS]
        return tokens

    def stem(self, word: str) -> str:
        """Simple Porter-like stemmer (no NLTK dependency)."""
        suffixes = ["ing", "tion", "ness", "ment", "able", "ible",
                    "ful", "less", "ous", "ive", "ize", "ise", "ed", "er", "ly", "s"]
        for sfx in suffixes:
            if word.endswith(sfx) and len(word) - len(sfx) >= 3:
                return word[: -len(sfx)]
        return word

    def preprocess(self, text: str, use_stemming: bool = False) -> str:
        """Full preprocessing pipeline → returns cleaned string."""
        tokens = self.tokenize(text)
        if use_stemming:
            tokens = [self.stem(t) for t in tokens]
        return " ".join(tokens)

    def get_ngrams(self, text: str, n: int = 3) -> List[str]:
        """Extract character n-grams for robust matching."""
        cleaned = self.clean(text)
        return [cleaned[i: i + n] for i in range(len(cleaned) - n + 1)]

    def get_word_ngrams(self, text: str, n: int = 2) -> List[Tuple[str, ...]]:
        """Extract word-level n-grams (shingles)."""
        tokens = self.tokenize(text)
        return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


# ─────────────────────────────────────────────────────────────
#  Similarity Algorithms
# ─────────────────────────────────────────────────────────────

class SimilarityEngine:
    """
    Multi-algorithm similarity computation engine.
    Provides TF-IDF cosine, Jaccard, n-gram overlap, and LSH methods.
    """

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        self.preprocessor = preprocessor or TextPreprocessor()
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._documents: List[str] = []

    # ── TF-IDF Cosine Similarity ───────────────────────────────

    def build_tfidf_index(self, texts: List[str]) -> None:
        """Build TF-IDF index from a corpus of documents."""
        processed = [self.preprocessor.preprocess(t) for t in texts]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,   # log(1+tf) smoothing
            strip_accents="unicode",
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(processed)
        self._documents = texts

    def cosine_sim(self, text1: str, text2: str) -> float:
        """
        Compute TF-IDF cosine similarity between two texts.
        Builds a local vectorizer if no index exists.
        """
        processed = [
            self.preprocessor.preprocess(text1),
            self.preprocessor.preprocess(text2),
        ]
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        try:
            tfidf = vec.fit_transform(processed)
            score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    def cosine_sim_matrix(self) -> np.ndarray:
        """Return pairwise cosine similarity matrix for indexed corpus."""
        if self._tfidf_matrix is None:
            raise ValueError("Call build_tfidf_index() first.")
        return cosine_similarity(self._tfidf_matrix)

    # ── Jaccard Similarity ─────────────────────────────────────

    def jaccard_sim(self, text1: str, text2: str) -> float:
        """
        Token-set Jaccard similarity.
        J(A,B) = |A ∩ B| / |A ∪ B|
        """
        set1 = set(self.preprocessor.tokenize(text1))
        set2 = set(self.preprocessor.tokenize(text2))
        if not set1 and not set2:
            return 1.0
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union)

    # ── N-gram / Shingle Similarity ────────────────────────────

    def ngram_sim(self, text1: str, text2: str, n: int = 3) -> float:
        """
        Shingle (word n-gram) Jaccard similarity.
        More robust to paraphrasing than single-token Jaccard.
        """
        shingles1 = set(self.preprocessor.get_word_ngrams(text1, n))
        shingles2 = set(self.preprocessor.get_word_ngrams(text2, n))
        if not shingles1 and not shingles2:
            return 1.0
        intersection = shingles1 & shingles2
        union = shingles1 | shingles2
        return len(intersection) / len(union) if union else 0.0

    # ── Common Phrases Extraction ─────────────────────────────

    def find_common_phrases(
        self, text1: str, text2: str, min_len: int = 5
    ) -> List[str]:
        """Find longest common substring phrases between two texts."""
        words1 = self.preprocessor.tokenize(text1)
        words2 = self.preprocessor.tokenize(text2)
        common = []
        for length in range(min_len, 2, -1):
            for i in range(len(words1) - length + 1):
                phrase = tuple(words1[i: i + length])
                for j in range(len(words2) - length + 1):
                    if tuple(words2[j: j + length]) == phrase:
                        phrase_str = " ".join(phrase)
                        if not any(phrase_str in c for c in common):
                            common.append(phrase_str)
        return common[:10]  # top 10

    # ── Word Overlap ───────────────────────────────────────────

    def word_overlap(self, text1: str, text2: str) -> List[str]:
        """Return sorted list of common significant words."""
        set1 = set(self.preprocessor.tokenize(text1))
        set2 = set(self.preprocessor.tokenize(text2))
        return sorted(list(set1 & set2))

    # ── MinHash / LSH approximation ────────────────────────────

    def minhash_sim(self, text1: str, text2: str, num_hashes: int = 100) -> float:
        """
        MinHash approximation of Jaccard similarity.
        Fast for very large documents.
        """
        shingles1 = set(self.preprocessor.get_word_ngrams(text1, 2))
        shingles2 = set(self.preprocessor.get_word_ngrams(text2, 2))
        if not shingles1 or not shingles2:
            return 0.0

        # Generate hash functions
        matches = 0
        LARGE_PRIME = (1 << 61) - 1
        rng = np.random.RandomState(42)
        a_vals = rng.randint(1, LARGE_PRIME, num_hashes)
        b_vals = rng.randint(0, LARGE_PRIME, num_hashes)

        def min_hash(shingles, a, b):
            return min(
                (a * hash(s) + b) % LARGE_PRIME for s in shingles
            )

        for a, b in zip(a_vals, b_vals):
            if min_hash(shingles1, a, b) == min_hash(shingles2, a, b):
                matches += 1

        return matches / num_hashes


# ─────────────────────────────────────────────────────────────
#  Main Plagiarism Checker
# ─────────────────────────────────────────────────────────────

class PlagiarismChecker:
    """
    ╔══════════════════════════════════════════════════════╗
    ║          MAIN PLAGIARISM CHECKER CLASS               ║
    ╚══════════════════════════════════════════════════════╝

    Usage:
        checker = PlagiarismChecker(threshold=0.75)
        checker.load_directory("./documents")
        results = checker.check_all()
        checker.print_report(results)
    """

    VERDICTS = {
        (0.90, 1.01): ("🔴 HIGHLY PLAGIARISED",   "HIGHLY_PLAGIARISED"),
        (0.75, 0.90): ("🟠 LIKELY PLAGIARISED",    "LIKELY_PLAGIARISED"),
        (0.50, 0.75): ("🟡 SUSPICIOUS",            "SUSPICIOUS"),
        (0.30, 0.50): ("🟢 MINOR SIMILARITY",      "MINOR_SIMILARITY"),
        (0.00, 0.30): ("✅ ORIGINAL",              "ORIGINAL"),
    }

    def __init__(
        self,
        threshold: float = 0.75,
        weights: Optional[Dict[str, float]] = None,
        use_stemming: bool = False,
        ngram_n: int = 3,
    ):
        """
        Args:
            threshold: Minimum combined score to flag as plagiarised (0–1).
            weights: Algorithm weights for combined score.
                     Default: {"cosine": 0.5, "jaccard": 0.25, "ngram": 0.25}
            use_stemming: Apply word stemming before comparison.
            ngram_n: Word n-gram window size.
        """
        self.threshold = threshold
        self.weights = weights or {"cosine": 0.5, "jaccard": 0.25, "ngram": 0.25}
        self.use_stemming = use_stemming
        self.ngram_n = ngram_n
        self.preprocessor = TextPreprocessor()
        self.similarity = SimilarityEngine(self.preprocessor)
        self._docs: Dict[str, str] = {}   # filename → content

    # ── Document Loading ───────────────────────────────────────

    def load_text(self, name: str, content: str) -> None:
        """Add a document by name and raw text content."""
        self._docs[name] = content
        logger.info(f"Loaded document: {name} ({len(content)} chars)")

    def load_file(self, filepath: Union[str, Path]) -> None:
        """Load a single text file."""
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")
        content = fp.read_text(encoding="utf-8", errors="replace")
        self.load_text(fp.name, content)

    def load_directory(self, directory: Union[str, Path], extensions: Tuple[str, ...] = (".txt", ".md")) -> int:
        """
        Load all matching files from a directory.
        Returns count of loaded files.
        """
        d = Path(directory)
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {d}")
        loaded = 0
        for fp in sorted(d.iterdir()):
            if fp.suffix.lower() in extensions and fp.is_file():
                self.load_file(fp)
                loaded += 1
        logger.info(f"Loaded {loaded} documents from {d}")
        return loaded

    def clear(self) -> None:
        """Remove all loaded documents."""
        self._docs.clear()

    @property
    def document_names(self) -> List[str]:
        return list(self._docs.keys())

    # ── Verdict Helper ─────────────────────────────────────────

    def _get_verdict(self, score: float) -> str:
        for (lo, hi), (emoji_label, code) in self.VERDICTS.items():
            if lo <= score < hi:
                return code
        return "ORIGINAL"

    def _get_verdict_label(self, score: float) -> str:
        for (lo, hi), (emoji_label, code) in self.VERDICTS.items():
            if lo <= score < hi:
                return emoji_label
        return "✅ ORIGINAL"

    # ── Core Comparison ────────────────────────────────────────

    def compare_pair(self, name1: str, name2: str) -> DocumentPair:
        """
        Compare two loaded documents using all algorithms.
        Returns a DocumentPair with combined score and verdict.
        """
        t0 = time.perf_counter()
        text1 = self._docs[name1]
        text2 = self._docs[name2]

        cosine = self.similarity.cosine_sim(text1, text2)
        jaccard = self.similarity.jaccard_sim(text1, text2)
        ngram = self.similarity.ngram_sim(text1, text2, self.ngram_n)

        combined = (
            self.weights["cosine"] * cosine
            + self.weights["jaccard"] * jaccard
            + self.weights["ngram"] * ngram
        )

        common_phrases = self.similarity.find_common_phrases(text1, text2)
        overlap_words = self.similarity.word_overlap(text1, text2)

        pair = DocumentPair(
            doc1=name1,
            doc2=name2,
            cosine_score=cosine,
            jaccard_score=jaccard,
            ngram_score=ngram,
            combined_score=combined,
            verdict=self._get_verdict(combined),
            highlighted_sections=common_phrases,
            word_overlap=overlap_words,
            execution_time=time.perf_counter() - t0,
        )
        return pair

    # ── Full Check ─────────────────────────────────────────────

    def check_all(self, progress: bool = True) -> CheckResult:
        """
        Compare every document pair in the loaded corpus.
        Returns a CheckResult with all pairs and statistics.
        """
        import datetime
        names = list(self._docs.keys())
        pairs_list = list(itertools.combinations(names, 2))
        total = len(pairs_list)

        if total == 0:
            return CheckResult(
                total_documents=len(names),
                total_pairs=0,
                plagiarised_pairs=0,
                timestamp=datetime.datetime.now().isoformat(),
            )

        t_start = time.perf_counter()
        results: List[DocumentPair] = []

        for idx, (n1, n2) in enumerate(pairs_list, 1):
            if progress:
                pct = idx / total * 100
                bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
                print(f"\r  [{bar}] {pct:5.1f}%  Comparing: {n1} ↔ {n2}", end="", flush=True)
            pair = self.compare_pair(n1, n2)
            results.append(pair)

        if progress:
            print()  # newline after progress

        plagiarised = sum(
            1 for r in results if r.combined_score >= self.threshold
        )

        return CheckResult(
            total_documents=len(names),
            total_pairs=total,
            plagiarised_pairs=plagiarised,
            pairs=results,
            timestamp=datetime.datetime.now().isoformat(),
            duration=time.perf_counter() - t_start,
        )

    def check_against_corpus(self, query_text: str, query_name: str = "query") -> List[DocumentPair]:
        """
        Check one document against the entire loaded corpus.
        Useful for checking a new submission against a database.
        """
        self._docs[query_name] = query_text
        results = []
        for name in self.document_names:
            if name == query_name:
                continue
            results.append(self.compare_pair(query_name, name))
        del self._docs[query_name]
        return sorted(results, key=lambda r: r.combined_score, reverse=True)

    # ── Reporting ──────────────────────────────────────────────

    def print_report(self, result: CheckResult) -> None:
        """Print a formatted terminal report."""
        WIDTH = 70

        print("\n" + "═" * WIDTH)
        print("  📊  PLAGIARISM CHECK REPORT".center(WIDTH))
        print("═" * WIDTH)
        print(f"  📁 Documents Scanned : {result.total_documents}")
        print(f"  🔄 Pairs Compared    : {result.total_pairs}")
        print(f"  ⚠️  Flagged Pairs     : {result.plagiarised_pairs}")
        print(f"  📈 Plagiarism Rate   : {result.plagiarism_rate:.1f}%")
        print(f"  ⏱  Time Taken        : {result.duration:.3f}s")
        print(f"  🕐 Timestamp         : {result.timestamp}")
        print("─" * WIDTH)

        if not result.pairs:
            print("  No pairs to display.")
        else:
            # Sort by combined score descending
            sorted_pairs = sorted(result.pairs, key=lambda p: p.combined_score, reverse=True)
            for pair in sorted_pairs:
                label = self._get_verdict_label(pair.combined_score)
                print(f"\n  {label}")
                print(f"  📄 {pair.doc1}  ↔  {pair.doc2}")
                print(f"     Cosine    : {pair.cosine_score * 100:6.2f}%")
                print(f"     Jaccard   : {pair.jaccard_score * 100:6.2f}%")
                print(f"     N-gram    : {pair.ngram_score * 100:6.2f}%")
                print(f"     Combined  : {pair.combined_score * 100:6.2f}%")
                if pair.highlighted_sections:
                    print(f"     Phrases   : \"{pair.highlighted_sections[0]}\" ...")
                print(f"     Time      : {pair.execution_time * 1000:.1f}ms")

        print("\n" + "═" * WIDTH + "\n")
