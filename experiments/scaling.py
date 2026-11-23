"""How the LSH pre-filter scales, on synthetic collections with planted copies.

    python experiments/scaling.py

Each document is 250 words drawn from a Zipf distribution over a 20k-word
vocabulary. The most frequent ranks are mapped to real English stopwords,
because that's what the top of a real word-frequency list looks like and the
LSH index ignores them. (Without this the fake "common words" collide all the
time and the filter keeps ~40% of pairs, which says nothing about real text.)
In 2% of documents a 60-word passage from another random document is pasted
in, and in another 2% the pasted passage has every fourth word replaced.
We check how many planted pairs survive the filter and how much work is saved.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plagiarism_checker import PlagiarismChecker  # noqa: E402
from plagiarism_checker.fingerprint import ChunkedCandidates  # noqa: E402
from plagiarism_checker.text import STOPWORDS  # noqa: E402

_SW = sorted(STOPWORDS)


def make_collection(n, seed=0, length=250, vocab=20000):
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, vocab + 1)
    p = 1 / ranks ** 1.05
    p /= p.sum()
    docs = [list(rng.choice(vocab, length, p=p)) for _ in range(n)]
    planted = {}
    k = max(1, n // 50)
    targets = rng.choice(n, 2 * k, replace=False)
    for i, t in enumerate(targets):
        src = int(rng.integers(n))
        while src == t:
            src = int(rng.integers(n))
        s = int(rng.integers(0, length - 60))
        passage = list(docs[src][s:s + 60])
        kind = "verbatim" if i < k else "edited"
        if kind == "edited":
            for j in range(0, 60, 4):
                passage[j] = int(rng.choice(vocab, p=p))
        at = int(rng.integers(0, length - 60))
        docs[t][at:at + 60] = passage
        planted[tuple(sorted((f"d{src}", f"d{t}")))] = kind
    word = lambda x: _SW[x] if x < len(_SW) else f"w{x}"  # noqa: E731
    texts = {f"d{i}": " ".join(word(x) for x in d) for i, d in enumerate(docs)}
    return texts, planted


def main():
    print("| documents | all pairs | pairs after LSH | kept | recall verbatim / edited | "
          "LSH time | est. brute-force time |")
    print("|---|---|---|---|---|---|---|")
    pc = PlagiarismChecker()
    # cost of one full comparison, measured on a sample
    texts, _ = make_collection(60, seed=99)
    for n, t in texts.items():
        pc.add_text(n, t)
    names = list(pc.docs)
    t0 = time.perf_counter()
    for i in range(0, 60, 2):
        pc.compare(pc.docs[names[i]], pc.docs[names[i + 1]])
    per_pair = (time.perf_counter() - t0) / 30

    for n in (500, 2000, 5000, 10000):
        texts, planted = make_collection(n, seed=n)
        pc = PlagiarismChecker()
        for name, t in texts.items():
            pc.add_text(name, t)
        t0 = time.perf_counter()
        cc = ChunkedCandidates()  # library defaults
        for name, d in pc.docs.items():
            cc.add(name, d.content_words)
        cand = {tuple(sorted(p)) for p in cc.pairs()}
        t_lsh = time.perf_counter() - t0
        total = n * (n - 1) // 2
        rec = {kind: np.mean([p in cand for p, k in planted.items() if k == kind])
               for kind in ("verbatim", "edited")}
        est = total * per_pair
        print(f"| {n} | {total:,} | {len(cand):,} | {len(cand) / total:.2%} | "
              f"{rec['verbatim']:.2f} / {rec['edited']:.2f} | {t_lsh:.1f} s | "
              f"{est / 60:.0f} min |")
    print(f"\nfull comparison cost: {per_pair * 1000:.1f} ms per pair")


if __name__ == "__main__":
    main()
