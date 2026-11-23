"""High-level API.

    from plagiarism_checker import PlagiarismChecker

    pc = PlagiarismChecker()
    pc.add_directory("essays/")
    result = pc.check_all()
    for m in result.flagged():
        print(m.doc1, m.doc2, m.label, round(m.score, 2))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .features import Document, pair_features
from .fingerprint import ChunkedCandidates, matched_spans
from .io import load_directory, read_document
from .model import DEFAULT_MODEL, ReuseModel
from .text import Preprocessor

PathLike = Union[str, Path]
Span = Tuple[int, int]


@dataclass
class Match:
    doc1: str
    doc2: str
    score: float                    # P(reuse), 0..1
    label: str                      # cut / light / heavy / non
    probs: Dict[str, float]
    features: Dict[str, float]
    spans1: List[Span] = field(default_factory=list)   # char offsets in doc1
    spans2: List[Span] = field(default_factory=list)
    flagged: bool = False

    @property
    def pretty_label(self) -> str:
        from .model import LABELS
        return LABELS.get(self.label, self.label)

    def to_dict(self) -> dict:
        return {
            "doc1": self.doc1,
            "doc2": self.doc2,
            "score": round(self.score, 4),
            "label": self.label,
            "flagged": self.flagged,
            "probs": {k: round(v, 4) for k, v in self.probs.items()},
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "spans1": self.spans1,
            "spans2": self.spans2,
        }


@dataclass
class CheckResult:
    matches: List[Match]
    n_documents: int
    n_pairs_total: int           # all possible pairs
    n_pairs_scored: int          # pairs that survived the LSH filter
    threshold: float
    seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def flagged(self) -> List[Match]:
        return [m for m in self.matches if m.flagged]

    def to_dict(self) -> dict:
        return {
            "n_documents": self.n_documents,
            "n_pairs_total": self.n_pairs_total,
            "n_pairs_scored": self.n_pairs_scored,
            "threshold": self.threshold,
            "seconds": round(self.seconds, 3),
            "timestamp": self.timestamp,
            "matches": [m.to_dict() for m in self.matches],
        }


class PlagiarismChecker:
    """Compare documents pairwise or against a reference collection.

    threshold  -- a pair is flagged when P(reuse) >= threshold
    use_lsh    -- "auto" turns the MinHash pre-filter on above `lsh_min_docs`
                  documents; True/False forces it
    stem       -- stem words before comparison (helps a little on paraphrase,
                  but the shipped model was trained without it)
    """

    def __init__(self, threshold: float = 0.5, model: Optional[ReuseModel] = None,
                 use_lsh: Union[bool, str] = "auto", lsh_min_docs: int = 200,
                 stem: bool = False):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.model = model or ReuseModel.load(DEFAULT_MODEL)
        self.use_lsh = use_lsh
        self.lsh_min_docs = lsh_min_docs
        self.pre = Preprocessor(stem=stem)
        self.docs: Dict[str, Document] = {}

    # -- loading ------------------------------------------------------------

    def add_text(self, name: str, text: str) -> Document:
        if name in self.docs:
            raise ValueError(f"a document called {name!r} is already loaded")
        doc = Document(name, text, self.pre)
        self.docs[name] = doc
        return doc

    def add_file(self, path: PathLike, name: Optional[str] = None) -> Document:
        p = Path(path)
        return self.add_text(name or p.name, read_document(p))

    def add_directory(self, folder: PathLike, recursive: bool = False) -> int:
        loaded = load_directory(folder, recursive=recursive)
        for name, text in loaded.items():
            self.add_text(name, text)
        return len(loaded)

    def clear(self) -> None:
        self.docs.clear()

    # -- scoring ------------------------------------------------------------

    def compare(self, d1: Document, d2: Document, spans: bool = True) -> Match:
        feats = pair_features(d1, d2)
        pred = self.model.predict(feats)
        m = Match(d1.name, d2.name, pred.reuse_score, pred.label, pred.probs, feats)
        m.flagged = m.score >= self.threshold
        if spans:
            m.spans1, m.spans2 = matched_spans(d1.fingerprints, d2.fingerprints,
                                               d1.tokens, d2.tokens)
        return m

    def compare_texts(self, text1: str, text2: str) -> Match:
        return self.compare(Document("text1", text1, self.pre), Document("text2", text2, self.pre))

    def _lsh_on(self, n_docs: int) -> bool:
        if self.use_lsh == "auto":
            return n_docs >= self.lsh_min_docs
        return bool(self.use_lsh)

    def _candidates(self, names: Sequence[str]) -> List[Tuple[str, str]]:
        if not self._lsh_on(len(names)):
            return list(combinations(names, 2))
        cc = ChunkedCandidates()
        for n in names:
            cc.add(n, self.docs[n].content_words)
        order = {n: i for i, n in enumerate(names)}
        return sorted((tuple(sorted(p, key=order.get)) for p in cc.pairs()),
                      key=lambda p: (order[p[0]], order[p[1]]))

    def check_all(self, progress=None) -> CheckResult:
        """Score every pair in the loaded collection (or LSH candidates only).

        `progress`, if given, is called as progress(done, total).
        """
        t0 = time.perf_counter()
        names = list(self.docs)
        pairs = self._candidates(names)
        matches = []
        for i, (a, b) in enumerate(pairs, 1):
            matches.append(self.compare(self.docs[a], self.docs[b]))
            if progress:
                progress(i, len(pairs))
        matches.sort(key=lambda m: m.score, reverse=True)
        n = len(names)
        return CheckResult(matches, n, n * (n - 1) // 2, len(pairs),
                           self.threshold, time.perf_counter() - t0)

    def check_against(self, query: Union[str, Document], query_name: str = "query",
                      progress=None) -> CheckResult:
        """Compare one submission with every loaded document.

        The query is never added to the collection, so it can't overwrite or
        delete a loaded document with the same name (the old version did both).
        """
        t0 = time.perf_counter()
        q = query if isinstance(query, Document) else Document(query_name, query, self.pre)
        names = list(self.docs)
        if self._lsh_on(len(names)):
            cc = ChunkedCandidates()
            for n in names:
                cc.add(n, self.docs[n].content_words)
            hits = cc.query(q.content_words)
            names = [n for n in names if n in hits]
        matches = []
        for i, n in enumerate(names, 1):
            matches.append(self.compare(q, self.docs[n]))
            if progress:
                progress(i, len(names))
        matches.sort(key=lambda m: m.score, reverse=True)
        return CheckResult(matches, len(self.docs) + 1, len(self.docs), len(names),
                           self.threshold, time.perf_counter() - t0)

    def check_file_against(self, path: PathLike, progress=None) -> CheckResult:
        p = Path(path)
        return self.check_against(read_document(p), p.name, progress=progress)
