"""Text reuse / plagiarism detection with lexical similarity and fingerprinting."""

__version__ = "3.0.0"

from .checker import CheckResult, Match, PlagiarismChecker  # noqa: E402
from .features import Document, pair_features  # noqa: E402

__all__ = ["PlagiarismChecker", "CheckResult", "Match", "Document", "pair_features"]
