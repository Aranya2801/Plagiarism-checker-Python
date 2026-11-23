"""Tokenisation and normalisation.

Every token keeps its character offsets in the original string. The old
version threw the original text away after cleaning, which is why the
"matching phrases" it printed never looked like anything in the documents.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

# Letters and digits, allowing an apostrophe inside a word (don't, student's).
# [^\W_] is \w without the underscore, so this works for non-ASCII text too.
_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")

# A short list on purpose. Long stopword lists hurt n-gram containment because
# copied passages lose most of their n-grams when the function words go.
STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has
have he her his i if in into is it its me my no not of on or our she so such than
that the their them then there these they this those to us was we were what when
where which while who whom why will with would you your
""".split())


@dataclass(frozen=True)
class Token:
    text: str   # normalised form (lowercased, maybe stemmed)
    start: int  # offset into the original document
    end: int


def normalise(text: str) -> str:
    # NFKC folds things like ligatures and full-width characters. Curly quotes
    # are flattened so "it’s" and "it's" tokenise the same way.
    text = unicodedata.normalize("NFKC", text)
    return text.replace("’", "'").replace("‘", "'")


def light_stem(word: str) -> str:
    """Cheap suffix stripping. Good enough to merge plurals and -ing/-ed forms.

    If NLTK is installed, `Preprocessor(stem=True)` uses its Porter stemmer.
    """
    if len(word) <= 4:
        return word
    for suffix in ("ational", "ization", "fulness", "ousness", "iveness",
                   "ations", "ation", "ments", "ment", "ness", "ings", "ing",
                   "ies", "ied", "edly", "ed", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: -len(suffix)]
            if suffix in ("ies", "ied"):
                stem += "y"
            elif suffix == "s" and stem.endswith("s"):
                return word  # "class", "process"
            return stem
    return word


def _porter_or_light():
    try:
        from nltk.stem import PorterStemmer  # type: ignore
        return PorterStemmer().stem
    except ImportError:
        return light_stem


class Preprocessor:
    def __init__(self, stem: bool = False, drop_stopwords: bool = False,
                 min_len: int = 1):
        self.stem = stem
        self.drop_stopwords = drop_stopwords
        self.min_len = min_len
        self._stemmer = _porter_or_light() if stem else None

    def tokens(self, text: str) -> List[Token]:
        text = normalise(text)
        out = []
        for m in _WORD_RE.finditer(text):
            w = m.group().lower()
            if len(w) < self.min_len:
                continue
            if self.drop_stopwords and w in STOPWORDS:
                continue
            if self._stemmer is not None:
                w = self._stemmer(w)
            out.append(Token(w, m.start(), m.end()))
        return out

    def words(self, text: str) -> List[str]:
        return [t.text for t in self.tokens(text)]

    def content_words(self, text: str) -> List[str]:
        return [w for w in self.words(text) if w not in STOPWORDS]

    def flat(self, text: str) -> str:
        """Lowercased words joined by single spaces (used for character n-grams)."""
        return " ".join(self.words(text))


_default: Optional[Preprocessor] = None


def default_preprocessor() -> Preprocessor:
    global _default
    if _default is None:
        _default = Preprocessor()
    return _default
