import math

import numpy as np
import pytest

from plagiarism_checker.features import (Document, containment, jaccard, lcs_length,
                                         lcs_ratio, pair_features, tf_cosine, FEATURES)
from plagiarism_checker.fingerprint import (PRIME_32, ChunkedCandidates, LSHIndex, MinHasher,
                                            matched_spans, stable_hash, winnow)
from plagiarism_checker.text import Preprocessor, light_stem

SOURCE = ("Inheritance is a basic concept of object oriented programming where the basic idea "
          "is to create new classes that add extra detail to existing classes. This is done by "
          "allowing the new classes to reuse the methods and variables of the existing classes.")


def doc(name, text):
    return Document(name, text)


# --- text --------------------------------------------------------------------

def test_tokens_keep_offsets():
    text = "Hello, World! It's   fine."
    toks = Preprocessor().tokens(text)
    assert [t.text for t in toks] == ["hello", "world", "it's", "fine"]
    for t in toks:
        assert text[t.start:t.end].lower() == t.text


def test_curly_apostrophe_matches_straight():
    a = Preprocessor().words("it’s a student’s book")
    b = Preprocessor().words("it's a student's book")
    assert a == b


def test_non_ascii_words_survive():
    # the old cleaner stripped everything outside a-z
    assert Preprocessor().words("Café naïve résumé") == ["café", "naïve", "résumé"]


@pytest.mark.parametrize("word,stem", [("classes", "class"), ("running", "runn"),
                                       ("studies", "study"), ("process", "process"),
                                       ("cat", "cat")])
def test_light_stem(word, stem):
    assert light_stem(word) == stem


# --- set measures ---------------------------------------------------------------

def test_jaccard_and_containment_basics():
    a, b = {1, 2, 3, 4}, {3, 4}
    assert jaccard(a, b) == 0.5
    assert containment(a, b) == 1.0          # all of b is inside a
    assert jaccard(set(), set()) == 0.0      # empty docs are not "identical"
    assert containment(set(), {1}) == 0.0


def test_containment_catches_paste_into_longer_doc():
    filler = " ".join(f"filler sentence number {i} about something else." for i in range(40))
    long = SOURCE + " " + filler
    d1, d2 = doc("short", SOURCE), doc("long", long)
    assert containment(d1.ngrams(3), d2.ngrams(3)) == 1.0
    assert jaccard(d1.ngrams(3), d2.ngrams(3)) < 0.5


def test_tf_cosine():
    from collections import Counter
    a = Counter("a b c".split())
    assert math.isclose(tf_cosine(a, a), 1.0)
    assert tf_cosine(a, Counter("x y".split())) == 0.0
    assert tf_cosine(a, Counter()) == 0.0


def test_lcs():
    assert lcs_length("a b c d e".split(), "a x c y e".split()) == 3
    assert lcs_ratio("a b c".split(), "a b c d e f".split()) == 1.0
    assert lcs_ratio([], ["a"]) == 0.0


def test_lcs_falls_back_on_huge_inputs(monkeypatch):
    import plagiarism_checker.features as F
    monkeypatch.setattr(F, "_LCS_EXACT_LIMIT", 10)
    a = "the cat sat on the mat today".split()
    assert 0 < F.lcs_ratio(a, a[::-1]) <= 1.0
    assert F.lcs_ratio(a, a) == 1.0


# --- fingerprints ------------------------------------------------------------------

def test_stable_hash_is_deterministic():
    assert stable_hash("abc") == stable_hash("abc")
    assert stable_hash("abc") != stable_hash("abd")


def test_winnow_guarantee():
    """A shared run of window + k - 1 words must share a fingerprint."""
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(500)]
    for _ in range(50):
        shared = list(rng.choice(vocab, 7))          # k=4, window=4 -> 7 words
        a = list(rng.choice(vocab, 40)) + shared + list(rng.choice(vocab, 40))
        b = list(rng.choice(vocab, 25)) + shared + list(rng.choice(vocab, 60))
        assert winnow(a).hashes & winnow(b).hashes


def test_winnow_short_inputs():
    assert len(winnow([])) == 0
    assert winnow(["one", "two"]).hashes == winnow(["one", "two"]).hashes


def test_matched_spans_point_at_copied_text():
    copied = "the new classes reuse the methods and variables of the existing classes"
    a = doc("a", SOURCE)
    b = doc("b", "My own intro sentence here. " + copied + ". And my own ending.")
    s1, s2 = matched_spans(a.fingerprints, b.fingerprints, a.tokens, b.tokens)
    assert s1 and s2
    highlighted = " ".join(b.text[s:e] for s, e in s2).lower()
    assert "methods and variables" in highlighted
    assert "own intro" not in highlighted


def test_minhash_estimates_jaccard():
    mh = MinHasher(num_perm=256)
    a = [f"s{i}" for i in range(100)]
    b = [f"s{i}" for i in range(50, 150)]      # true Jaccard = 50/150
    est = MinHasher.jaccard(mh.signature(a), mh.signature(b))
    assert abs(est - 1 / 3) < 0.08
    assert MinHasher.jaccard(mh.signature(a), mh.signature(a)) == 1.0


def test_minhash_no_overflow_and_reproducible():
    sig1 = MinHasher(128, seed=3).signature(["x", "y", "z"])
    sig2 = MinHasher(128, seed=3).signature(["z", "y", "x"])
    assert sig1.dtype == np.uint64
    assert (sig1 == sig2).all()
    assert (sig1 < PRIME_32).all()


def test_lsh_finds_near_duplicates_only():
    mh, idx = MinHasher(128), LSHIndex(32, 4)
    base = [f"t{i}" for i in range(200)]
    idx.add("a", mh.signature(base))
    idx.add("b", mh.signature(base[:190] + ["new"] * 10))
    idx.add("c", mh.signature([f"u{i}" for i in range(200)]))
    assert idx.candidate_pairs() == {("a", "b")}


def test_chunked_candidates_find_partial_copy():
    rng = np.random.default_rng(1)
    vocab = [f"w{i}" for i in range(3000)]
    source = list(rng.choice(vocab, 1000))
    answer = list(rng.choice(vocab, 30)) + source[400:460] + list(rng.choice(vocab, 30))
    other = list(rng.choice(vocab, 120))
    cc = ChunkedCandidates()
    cc.add("source", source)
    cc.add("other", other)
    assert "source" in cc.query(answer)
    assert "other" not in cc.query(answer)


# --- feature vector -----------------------------------------------------------------

def test_pair_features_ranges_and_symmetry():
    a, b = doc("a", SOURCE), doc("b", "Inheritance lets new classes reuse existing ones.")
    f, g = pair_features(a, b), pair_features(b, a)
    assert set(f) == set(FEATURES)
    for k in FEATURES:
        assert 0.0 <= f[k] <= 1.0 + 1e-9
        assert math.isclose(f[k], g[k], abs_tol=1e-9), k


def test_identical_docs_score_one():
    f = pair_features(doc("a", SOURCE), doc("b", SOURCE))
    assert all(math.isclose(v, 1.0) for v in f.values())
