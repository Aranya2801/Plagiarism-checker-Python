"""
╔══════════════════════════════════════════════════════════════╗
║               PLAGIARISM CHECKER — Test Suite               ║
╚══════════════════════════════════════════════════════════════╝
Run: pytest tests/ -v
"""

import pytest
from plagiarism_checker.engine import (
    PlagiarismChecker,
    TextPreprocessor,
    SimilarityEngine,
    DocumentPair,
)

# ─── Fixtures ──────────────────────────────────────────────────────

IDENTICAL   = "The quick brown fox jumps over the lazy dog."
PARAPHRASED = "A fast auburn fox leaps above the sleepy canine."
UNRELATED   = "The history of Rome spans centuries of political evolution."
COPY_PASTE  = IDENTICAL  # exact duplicate

LONG_A = """
Artificial intelligence is transforming the modern world in profound ways.
Machine learning algorithms can now recognize patterns in data that were
previously invisible to human analysts. From healthcare diagnostics to
financial forecasting, AI systems are being deployed across every major
industry. The development of large language models has opened new frontiers
in natural language processing.
"""

LONG_B = """
AI is changing the modern world significantly. Machine learning can identify
patterns in data that were hard for humans to see. AI systems work in
healthcare, finance, and other industries. Large language models opened new
possibilities in natural language processing and text generation.
"""

LONG_C = """
Ancient Rome began as a small city-state and grew into one of history's
greatest empires. Roman engineering produced aqueducts and roads that
still stand today. The Pax Romana was a long period of peace and prosperity.
Julius Caesar, Augustus, and Nero were among Rome's most famous rulers.
"""


# ─── TextPreprocessor Tests ────────────────────────────────────────

class TestTextPreprocessor:
    def setup_method(self):
        self.pp = TextPreprocessor()

    def test_clean_removes_urls(self):
        text = "Visit https://example.com for more info."
        cleaned = self.pp.clean(text)
        assert "https" not in cleaned
        assert "example" not in cleaned

    def test_clean_removes_emails(self):
        text = "Contact me at hello@example.com please."
        cleaned = self.pp.clean(text)
        assert "@" not in cleaned

    def test_clean_lowercases(self):
        assert self.pp.clean("Hello WORLD") == "hello world"

    def test_tokenize_removes_stopwords(self):
        tokens = self.pp.tokenize("the quick brown fox")
        assert "the" not in tokens

    def test_tokenize_removes_short_words(self):
        tokens = self.pp.tokenize("a to is the fox")
        # Only 'fox' should survive (len >= 3 and not stopword)
        for t in tokens:
            assert len(t) >= 3

    def test_ngrams_length(self):
        ngrams = self.pp.get_ngrams("hello world", n=3)
        for ng in ngrams:
            assert len(ng) == 3

    def test_word_ngrams(self):
        bigrams = self.pp.get_word_ngrams("one two three four", n=2)
        assert ("one", "two") in bigrams or len(bigrams) > 0

    def test_stem_suffix_removal(self):
        assert self.pp.stem("running") in ("runn", "run", "running")
        assert self.pp.stem("kindness") in ("kind", "kindness")


# ─── SimilarityEngine Tests ────────────────────────────────────────

class TestSimilarityEngine:
    def setup_method(self):
        self.engine = SimilarityEngine()

    def test_cosine_identical(self):
        score = self.engine.cosine_sim(IDENTICAL, IDENTICAL)
        assert score > 0.95, f"Identical docs should have cosine ≈ 1, got {score}"

    def test_cosine_unrelated(self):
        score = self.engine.cosine_sim(IDENTICAL, UNRELATED)
        assert score < 0.5, f"Unrelated docs should have low cosine, got {score}"

    def test_jaccard_identical(self):
        score = self.engine.jaccard_sim(IDENTICAL, IDENTICAL)
        assert score == pytest.approx(1.0)

    def test_jaccard_unrelated(self):
        score = self.engine.jaccard_sim(LONG_A, LONG_C)
        assert score < 0.3

    def test_jaccard_partial(self):
        score = self.engine.jaccard_sim(LONG_A, LONG_B)
        assert score > 0.05

    def test_ngram_identical(self):
        score = self.engine.ngram_sim(LONG_A, LONG_A, n=2)
        assert score == pytest.approx(1.0)

    def test_ngram_unrelated(self):
        score = self.engine.ngram_sim(LONG_A, LONG_C, n=2)
        assert score < 0.3

    def test_cosine_symmetric(self):
        s1 = self.engine.cosine_sim(LONG_A, LONG_B)
        s2 = self.engine.cosine_sim(LONG_B, LONG_A)
        assert abs(s1 - s2) < 1e-6

    def test_minhash_identical(self):
        score = self.engine.minhash_sim(LONG_A, LONG_A)
        assert score > 0.9

    def test_minhash_unrelated(self):
        score = self.engine.minhash_sim(LONG_A, LONG_C)
        assert score < 0.5

    def test_word_overlap(self):
        overlap = self.engine.word_overlap(LONG_A, LONG_B)
        assert isinstance(overlap, list)

    def test_find_common_phrases(self):
        phrases = self.engine.find_common_phrases(LONG_A, LONG_A)
        assert len(phrases) > 0

    def test_tfidf_matrix_shape(self):
        texts = [LONG_A, LONG_B, LONG_C]
        self.engine.build_tfidf_index(texts)
        matrix = self.engine.cosine_sim_matrix()
        assert matrix.shape == (3, 3)


# ─── PlagiarismChecker Tests ───────────────────────────────────────

class TestPlagiarismChecker:
    def setup_method(self):
        self.checker = PlagiarismChecker(threshold=0.75)

    def _load_pair(self, t1, t2):
        self.checker.clear()
        self.checker.load_text("doc1.txt", t1)
        self.checker.load_text("doc2.txt", t2)

    def test_load_text(self):
        self.checker.load_text("test.txt", "Hello world")
        assert "test.txt" in self.checker.document_names

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            self.checker.load_file("/nonexistent/path/file.txt")

    def test_identical_flagged(self):
        self._load_pair(COPY_PASTE, COPY_PASTE)
        result = self.checker.check_all(progress=False)
        assert result.plagiarised_pairs == 1

    def test_unrelated_not_flagged(self):
        self._load_pair(LONG_A, LONG_C)
        result = self.checker.check_all(progress=False)
        assert result.plagiarised_pairs == 0

    def test_result_counts(self):
        self.checker.clear()
        for i, text in enumerate([LONG_A, LONG_B, LONG_C]):
            self.checker.load_text(f"doc{i}.txt", text)
        result = self.checker.check_all(progress=False)
        assert result.total_documents == 3
        assert result.total_pairs == 3

    def test_combined_score_range(self):
        self._load_pair(LONG_A, LONG_B)
        result = self.checker.check_all(progress=False)
        for pair in result.pairs:
            assert 0.0 <= pair.combined_score <= 1.0

    def test_verdict_assignment(self):
        self._load_pair(COPY_PASTE, COPY_PASTE)
        result = self.checker.check_all(progress=False)
        assert result.pairs[0].verdict in {
            "HIGHLY_PLAGIARISED", "LIKELY_PLAGIARISED", "SUSPICIOUS",
            "MINOR_SIMILARITY", "ORIGINAL"
        }

    def test_check_against_corpus(self):
        self.checker.clear()
        self.checker.load_text("ref1.txt", LONG_B)
        self.checker.load_text("ref2.txt", LONG_C)
        pairs = self.checker.check_against_corpus(LONG_A, "query")
        assert len(pairs) == 2
        # Results sorted by score descending
        assert pairs[0].combined_score >= pairs[1].combined_score

    def test_plagiarism_rate(self):
        self.checker.clear()
        for i, text in enumerate([LONG_A, LONG_B, LONG_C]):
            self.checker.load_text(f"doc{i}.txt", text)
        result = self.checker.check_all(progress=False)
        assert 0.0 <= result.plagiarism_rate <= 100.0

    def test_empty_check(self):
        self.checker.clear()
        result = self.checker.check_all(progress=False)
        assert result.total_pairs == 0

    def test_single_document(self):
        self.checker.clear()
        self.checker.load_text("solo.txt", LONG_A)
        result = self.checker.check_all(progress=False)
        assert result.total_pairs == 0

    def test_document_pair_to_dict(self):
        pair = DocumentPair(
            doc1="a.txt", doc2="b.txt",
            cosine_score=0.8, jaccard_score=0.6, ngram_score=0.7,
            combined_score=0.73, verdict="SUSPICIOUS"
        )
        d = pair.to_dict()
        assert d["document_1"] == "a.txt"
        assert "combined_score" in d["scores"]
        assert d["verdict"] == "SUSPICIOUS"


# ─── Utils Tests ───────────────────────────────────────────────────

class TestUtils:
    def test_document_fingerprint(self):
        from plagiarism_checker.utils import document_fingerprint, are_duplicates
        fp = document_fingerprint("hello world")
        assert len(fp) == 64
        assert are_duplicates("same", "same")
        assert not are_duplicates("one", "two")

    def test_document_stats(self):
        from plagiarism_checker.utils import document_stats
        stats = document_stats("Hello world. How are you?")
        assert stats["words"] > 0
        assert stats["sentences"] > 0
        assert stats["characters"] > 0

    def test_word_count(self):
        from plagiarism_checker.utils import word_count
        assert word_count("one two three") == 3

    def test_create_sample_documents(self, tmp_path):
        from plagiarism_checker.utils import create_sample_documents, SAMPLE_DOCUMENTS
        create_sample_documents(str(tmp_path))
        created = list(tmp_path.iterdir())
        assert len(created) == len(SAMPLE_DOCUMENTS)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
