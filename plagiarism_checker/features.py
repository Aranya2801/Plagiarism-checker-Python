"""Pairwise similarity features.

Everything in FEATURES is a number in [0, 1] that depends only on the two
documents being compared, not on the rest of the collection. That matters:
the old code fitted TF-IDF on just the two documents in each pair, and with
two documents the IDF weights carry almost no information. Corpus-level
TF-IDF / LSA is still available through `CorpusIndex`, but it is kept out of
the scoring model so a two-file check behaves the same as a 1000-file one.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .fingerprint import Fingerprints, fingerprint_overlap, winnow
from .text import Preprocessor, Token, default_preprocessor, normalise, STOPWORDS


@dataclass
class Document:
    name: str
    text: str
    pre: Preprocessor = field(default_factory=default_preprocessor, repr=False)

    def __post_init__(self):
        # offsets in Token refer to the normalised text, so keep that version
        self.text = normalise(self.text)

    @cached_property
    def tokens(self) -> List[Token]:
        return self.pre.tokens(self.text)

    @cached_property
    def words(self) -> List[str]:
        return [t.text for t in self.tokens]

    @cached_property
    def content_words(self) -> List[str]:
        return [w for w in self.words if w not in STOPWORDS]

    def ngrams(self, n: int) -> Set[Tuple[str, ...]]:
        cache = self.__dict__.setdefault("_ngram_cache", {})
        if n not in cache:
            w = self.words
            cache[n] = {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
        return cache[n]

    @cached_property
    def fingerprints(self) -> Fingerprints:
        return winnow(self.words, k=4, window=4)

    @cached_property
    def word_tf(self) -> Counter:
        c = Counter(self.content_words)
        w = self.content_words
        c.update(" ".join(w[i:i + 2]) for i in range(len(w) - 1))
        return c

    @cached_property
    def char_tf(self) -> Counter:
        s = " " + " ".join(self.words) + " "
        c = Counter()
        for n in (3, 4, 5):
            c.update(s[i:i + n] for i in range(len(s) - n + 1))
        return c


# --------------------------------------------------------------------------
# individual measures
# --------------------------------------------------------------------------

def jaccard(a: Set, b: Set) -> float:
    if not a or not b:
        # Two empty documents are not "identical", they're just empty.
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: Set, b: Set) -> float:
    """Broder's containment, symmetrised with max().

    c(A, B) = |A ∩ B| / |A| answers "how much of A appears in B". Jaccard
    punishes length differences, so a paragraph pasted into a long essay looks
    unrelated under Jaccard but scores 1.0 here.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return max(inter / len(a), inter / len(b))


def _sublinear(c: Counter) -> Dict[str, float]:
    return {k: 1.0 + math.log(v) for k, v in c.items()}


def tf_cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    wa, wb = _sublinear(a), _sublinear(b)
    if len(wa) > len(wb):
        wa, wb = wb, wa
    dot = sum(v * wb.get(k, 0.0) for k, v in wa.items())
    na = math.sqrt(sum(v * v for v in wa.values()))
    nb = math.sqrt(sum(v * v for v in wb.values()))
    return dot / (na * nb) if na and nb else 0.0


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Word-level longest common subsequence, O(len(a) * len(b)) time,
    O(min) memory."""
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


# Above this many cells the exact DP gets slow in pure Python (~2 s), so we
# switch to difflib's matching blocks, which is a close lower bound.
_LCS_EXACT_LIMIT = 2_000_000


def lcs_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    """LCS length normalised by the shorter document."""
    if not a or not b:
        return 0.0
    if len(a) * len(b) <= _LCS_EXACT_LIMIT:
        n = lcs_length(a, b)
    else:
        sm = SequenceMatcher(None, a, b, autojunk=False)
        n = sum(m.size for m in sm.get_matching_blocks())
    return n / min(len(a), len(b))


# --------------------------------------------------------------------------
# feature vector
# --------------------------------------------------------------------------

FEATURES = (
    "containment_1",   # word unigram containment
    "containment_3",   # word trigram containment
    "containment_5",
    "jaccard_3",
    "lcs",
    "winnow",
    "word_cosine",     # sublinear TF on content words + bigrams
    "char_cosine",     # char 3-5 grams, catches morphology / small edits
)


def pair_features(d1: Document, d2: Document) -> Dict[str, float]:
    return {
        "containment_1": containment(set(d1.words), set(d2.words)),
        "containment_3": containment(d1.ngrams(3), d2.ngrams(3)),
        "containment_5": containment(d1.ngrams(5), d2.ngrams(5)),
        "jaccard_3": jaccard(d1.ngrams(3), d2.ngrams(3)),
        "lcs": lcs_ratio(d1.words, d2.words),
        "winnow": fingerprint_overlap(d1.fingerprints, d2.fingerprints),
        "word_cosine": tf_cosine(d1.word_tf, d2.word_tf),
        "char_cosine": tf_cosine(d1.char_tf, d2.char_tf),
    }


# --------------------------------------------------------------------------
# corpus-level vector space measures (used in experiments, optional in API)
# --------------------------------------------------------------------------

class CorpusIndex:
    """TF-IDF and LSA over a whole collection.

    Only meaningful when the collection is reasonably large (dozens of docs or
    more). Not part of the default scoring model for the reason given at the
    top of this file.
    """

    def __init__(self, docs: Sequence[Document], lsa_dims: int = 100):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self.names = [d.name for d in docs]
        self._pos = {n: i for i, n in enumerate(self.names)}
        texts = [" ".join(d.content_words) for d in docs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                          token_pattern=r"\S+")
        self.tfidf = self.vectorizer.fit_transform(texts)

        k = min(lsa_dims, self.tfidf.shape[0] - 1, self.tfidf.shape[1] - 1)
        self.lsa: Optional[np.ndarray] = None
        if k >= 2:
            svd = TruncatedSVD(n_components=k, random_state=0)
            self.lsa = normalize(svd.fit_transform(self.tfidf))

    def tfidf_cosine(self, a: str, b: str) -> float:
        i, j = self._pos[a], self._pos[b]
        return float(self.tfidf[i].multiply(self.tfidf[j]).sum())

    def lsa_cosine(self, a: str, b: str) -> float:
        if self.lsa is None:
            return float("nan")
        i, j = self._pos[a], self._pos[b]
        return float(np.clip(self.lsa[i] @ self.lsa[j], 0.0, 1.0))


class SemanticEncoder:
    """Optional sentence-embedding similarity (pip install sentence-transformers).

    Lexical features are weak on heavy paraphrase; embeddings help there but
    are slow and need a model download, so this is opt-in.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError("semantic similarity needs: pip install sentence-transformers") from e
        self.model = SentenceTransformer(model_name)
        self._cache: Dict[str, np.ndarray] = {}

    def embed(self, doc: Document) -> np.ndarray:
        if doc.name not in self._cache:
            self._cache[doc.name] = self.model.encode(doc.text, normalize_embeddings=True)
        return self._cache[doc.name]

    def similarity(self, d1: Document, d2: Document) -> float:
        return float(np.clip(self.embed(d1) @ self.embed(d2), 0.0, 1.0))
