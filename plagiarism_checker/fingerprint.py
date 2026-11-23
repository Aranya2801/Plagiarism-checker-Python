"""Document fingerprinting: winnowing for local matches, MinHash + LSH for
finding candidate pairs in big collections.

Winnowing follows Schleimer, Wilkerson & Aiken (2003), the algorithm behind
MOSS, but works on word k-grams instead of characters since we're dealing
with prose rather than code.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Hashable, Iterable, List, Sequence, Set, Tuple

import numpy as np

from .text import Token

PRIME_32 = 4294967291  # largest prime below 2^32


def stable_hash(s: str, bits: int = 64) -> int:
    # Python's built-in hash() is salted per process (PYTHONHASHSEED), so the
    # old MinHash gave different scores on every run. blake2b doesn't.
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=bits // 8).digest(), "little")


# --------------------------------------------------------------------------
# Winnowing
# --------------------------------------------------------------------------

@dataclass
class Fingerprints:
    # hash -> token positions where the k-gram starts
    positions: Dict[int, List[int]]
    k: int
    n_tokens: int

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def hashes(self) -> Set[int]:
        return set(self.positions)


def winnow(words: Sequence[str], k: int = 4, window: int = 4) -> Fingerprints:
    """Select fingerprints from word k-gram hashes.

    Any shared run of at least `window + k - 1` words is guaranteed to produce
    at least one common fingerprint (the "guarantee threshold" in the paper).
    With k=4, window=4 that's 7 words.
    """
    n = len(words)
    if n < k:
        # Too short for a full k-gram; fall back to a single hash of everything
        # so that two identical one-liners still match.
        if n == 0:
            return Fingerprints({}, k, 0)
        return Fingerprints({stable_hash(" ".join(words)): [0]}, k, n)

    hashes = [stable_hash(" ".join(words[i:i + k])) for i in range(n - k + 1)]
    picked: Dict[int, List[int]] = defaultdict(list)

    if len(hashes) <= window:
        i = min(range(len(hashes)), key=lambda j: (hashes[j], -j))
        picked[hashes[i]].append(i)
        return Fingerprints(dict(picked), k, n)

    last = -1
    for start in range(len(hashes) - window + 1):
        # rightmost minimum, as in the paper; avoids re-picking on ties
        best = start
        for j in range(start + 1, start + window):
            if hashes[j] <= hashes[best]:
                best = j
        if best != last:
            picked[hashes[best]].append(best)
            last = best
    return Fingerprints(dict(picked), k, n)


def fingerprint_overlap(a: Fingerprints, b: Fingerprints) -> float:
    """Shared fingerprints over the smaller set. Using min() rather than the
    union means a short text copied wholesale into a long one still scores 1."""
    if not len(a) or not len(b):
        return 0.0
    return len(a.hashes & b.hashes) / min(len(a), len(b))


def _merge(intervals: List[Tuple[int, int]], gap: int = 0) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def matched_spans(a: Fingerprints, b: Fingerprints,
                  tokens_a: Sequence[Token], tokens_b: Sequence[Token],
                  gap: int = 2) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Character spans in each document covered by shared fingerprints.

    Token ranges closer than `gap` tokens are merged, so a copied sentence with
    one word changed shows up as a single highlighted block instead of two.
    """
    shared = a.hashes & b.hashes
    if not shared:
        return [], []

    def spans(fp: Fingerprints, toks: Sequence[Token]):
        ranges = []
        for h in shared:
            for p in fp.positions[h]:
                ranges.append((p, min(p + fp.k, len(toks)) - 1))
        out = []
        for s, e in _merge(ranges, gap=gap):
            out.append((toks[s].start, toks[e].end))
        return out

    return spans(a, tokens_a), spans(b, tokens_b)


# --------------------------------------------------------------------------
# MinHash + LSH
# --------------------------------------------------------------------------

class MinHasher:
    """MinHash signatures using h(x) = (a*x + b) mod p with p = 2^32 - 5.

    x, a and b are all below p, so a*x + b < 2^64 and the arithmetic is exact
    in uint64. (The old version used int64 and randint(…, 2**61), which
    overflowed silently on Linux and raised on Windows, where numpy's default
    int is 32-bit. My first fix kept p = 2^61 - 1 but limited a to 32 bits,
    which makes h(x) almost monotone in x and MinHash overestimates Jaccard
    badly; tests/test_features.py checks for that now.)
    """

    def __init__(self, num_perm: int = 128, seed: int = 1):
        self.num_perm = num_perm
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, PRIME_32, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, PRIME_32, size=num_perm, dtype=np.uint64)

    def signature(self, shingles: Iterable[str]) -> np.ndarray:
        hv = np.fromiter((stable_hash(s) % PRIME_32 for s in set(shingles)), dtype=np.uint64)
        if hv.size == 0:
            return np.full(self.num_perm, PRIME_32, dtype=np.uint64)
        # (num_perm, n_shingles) matrix, then the min over shingles
        h = (np.outer(self.a, hv) + self.b[:, None]) % np.uint64(PRIME_32)
        return h.min(axis=1)

    @staticmethod
    def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))


def lsh_threshold(bands: int, rows: int) -> float:
    """Approximate Jaccard value where the banding S-curve crosses 0.5."""
    return (1.0 / bands) ** (1.0 / rows)


class LSHIndex:
    """Banded LSH over MinHash signatures.

    Two documents become a candidate pair if any band of their signatures is
    identical. P(candidate) = 1 - (1 - s^rows)^bands for Jaccard similarity s.
    """

    def __init__(self, bands: int = 32, rows: int = 4):
        self.bands = bands
        self.rows = rows
        self._buckets: Dict[Tuple[int, bytes], List[Hashable]] = defaultdict(list)

    def add(self, key: Hashable, sig: np.ndarray) -> None:
        if len(sig) < self.bands * self.rows:
            raise ValueError(f"signature too short for {self.bands}x{self.rows} bands")
        for b in range(self.bands):
            chunk = sig[b * self.rows:(b + 1) * self.rows]
            self._buckets[(b, chunk.tobytes())].append(key)

    def candidate_pairs(self) -> Set[Tuple[Hashable, Hashable]]:
        pairs = set()
        for keys in self._buckets.values():
            if len(keys) > 1:
                for x, y in combinations(sorted(set(keys), key=str), 2):
                    pairs.add((x, y))
        return pairs

    def query(self, sig: np.ndarray) -> Set[Hashable]:
        found = set()
        for b in range(self.bands):
            chunk = sig[b * self.rows:(b + 1) * self.rows]
            found.update(self._buckets.get((b, chunk.tobytes()), ()))
        return found


class ChunkedCandidates:
    """LSH over overlapping word windows instead of whole documents.

    Whole-document MinHash estimates Jaccard, which collapses when a short
    text is copied from a long one: a 200-word answer lifted from a 1000-word
    article has Jaccard ~0.2 even if every word was copied. Indexing short
    overlapping windows means the copied part only has to match one window of
    the source. Window/stride were picked with experiments/scaling.py and
    checked on the Clough & Stevenson corpus (experiments/results/results.md):
    a smaller stride helps when the copied passage doesn't line up with the
    window boundaries, at the cost of a slightly bigger index.
    """

    def __init__(self, window: int = 25, stride: int = 8, bands: int = 32,
                 rows: int = 4, seed: int = 1):
        self.window, self.stride = window, stride
        self.hasher = MinHasher(bands * rows, seed=seed)
        self.index = LSHIndex(bands, rows)
        self._seen: Set[Hashable] = set()

    def _chunks(self, words: Sequence[str]):
        if len(words) <= self.window:
            yield words
            return
        last = 0
        for s in range(0, len(words) - self.window + 1, self.stride):
            last = s
            yield words[s:s + self.window]
        if last + self.window < len(words):  # don't drop the tail
            yield words[-self.window:]

    def add(self, key: Hashable, words: Sequence[str]) -> None:
        self._seen.add(key)
        for ch in self._chunks(words):
            self.index.add(key, self.hasher.signature(ch))

    def pairs(self) -> Set[Tuple[Hashable, Hashable]]:
        return {(a, b) for a, b in self.index.candidate_pairs() if a != b}

    def query(self, words: Sequence[str]) -> Set[Hashable]:
        found: Set[Hashable] = set()
        for ch in self._chunks(words):
            found |= self.index.query(self.hasher.signature(ch))
        return found
