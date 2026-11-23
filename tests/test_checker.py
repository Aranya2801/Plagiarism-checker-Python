import json
from pathlib import Path

import pytest

from plagiarism_checker import PlagiarismChecker
from plagiarism_checker.cli import main
from plagiarism_checker.io import read_text
from plagiarism_checker.model import ReuseModel
from plagiarism_checker.report import render_html

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture
def checker():
    pc = PlagiarismChecker()
    pc.add_directory(SAMPLES)
    return pc


def test_model_loads_and_outputs_probabilities():
    m = ReuseModel.load()
    feats = {f: 0.5 for f in m.features}
    p = m.predict(feats)
    assert abs(sum(p.probs.values()) - 1.0) < 1e-9
    assert set(p.probs) == {"cut", "light", "heavy", "non"}


def test_copy_is_flagged_and_unrelated_is_not(checker):
    res = checker.check_all()
    assert res.n_pairs_scored == 6
    top = res.matches[0]
    assert {top.doc1, top.doc2} == {"original_essay.txt", "partial_copy.txt"}
    assert top.flagged and top.score > 0.9
    for m in res.matches:
        if "unrelated_history.txt" in (m.doc1, m.doc2):
            assert not m.flagged and m.score < 0.1


def test_matches_are_sorted(checker):
    scores = [m.score for m in checker.check_all().matches]
    assert scores == sorted(scores, reverse=True)


def test_check_against_does_not_touch_collection(checker):
    before = dict(checker.docs)
    # same name as a loaded document: must not overwrite or delete it
    res = checker.check_against("completely different text about cooking pasta", "partial_copy.txt")
    assert checker.docs == before
    assert len(res.matches) == 4


def test_duplicate_names_rejected():
    pc = PlagiarismChecker()
    pc.add_text("a", "x")
    with pytest.raises(ValueError):
        pc.add_text("a", "y")


def test_threshold_validation():
    with pytest.raises(ValueError):
        PlagiarismChecker(threshold=1.5)


def test_lsh_mode_keeps_the_copied_pair(checker):
    checker.use_lsh = True
    res = checker.check_all()
    assert res.n_pairs_scored <= res.n_pairs_total
    pairs = {frozenset((m.doc1, m.doc2)) for m in res.matches}
    assert frozenset(("original_essay.txt", "partial_copy.txt")) in pairs


def test_empty_documents_are_not_plagiarism():
    m = PlagiarismChecker().compare_texts("", "")
    assert not m.flagged


def test_html_is_escaped(checker):
    checker.add_text("<script>alert(1)</script>.txt", "<b>hello</b> " * 20)
    res = checker.check_all()
    html = render_html(res, {n: d.text for n, d in checker.docs.items()})
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_read_text_encodings(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes("caf\xe9".encode("cp1252"))
    assert read_text(p) == "café"
    p.write_bytes("﻿hello".encode("utf-8"))
    assert read_text(p) == "hello"
    p.write_bytes("hi there".encode("utf-16"))
    assert read_text(p) == "hi there"


# --- CLI ----------------------------------------------------------------------

def test_cli_directory(tmp_path, capsys):
    out = tmp_path / "r.json"
    html = tmp_path / "r.html"
    assert main([str(SAMPLES), "--json", str(out), "--html", str(html), "-q", "--no-color"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_documents"] == 4
    assert html.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "flagged" in capsys.readouterr().out


def test_cli_against_skips_itself(capsys):
    q = SAMPLES / "partial_copy.txt"
    assert main([str(q), "--against", str(SAMPLES), "-q", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "3 pairs scored" in out


def test_cli_fail_on_match():
    files = [str(SAMPLES / "original_essay.txt"), str(SAMPLES / "partial_copy.txt")]
    assert main(files + ["--fail-on-match", "-q"]) == 2
    files = [str(SAMPLES / "original_essay.txt"), str(SAMPLES / "unrelated_history.txt")]
    assert main(files + ["--fail-on-match", "-q"]) == 0


def test_cli_bad_input(tmp_path):
    assert main([str(tmp_path / "missing.txt")]) == 1
    assert main([str(SAMPLES / "original_essay.txt")]) == 1
