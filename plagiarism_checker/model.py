"""Turning feature vectors into a verdict.

The shipped weights are a multinomial logistic regression trained on the
Clough & Stevenson corpus by experiments/benchmark.py. Inference is a few
lines of numpy, so scikit-learn isn't needed just to score documents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

DEFAULT_MODEL = Path(__file__).with_name("data") / "model.json"

LABELS = {
    "cut": "near copy",
    "light": "light revision",
    "heavy": "heavy revision",
    "non": "original",
}


@dataclass
class Prediction:
    label: str                    # one of LABELS keys
    probs: Dict[str, float]

    @property
    def reuse_score(self) -> float:
        """P(text was reused in any form) = 1 - P(original)."""
        return 1.0 - self.probs.get("non", 0.0)

    @property
    def pretty_label(self) -> str:
        return LABELS.get(self.label, self.label)


class ReuseModel:
    def __init__(self, features: Sequence[str], classes: Sequence[str],
                 coef: np.ndarray, intercept: np.ndarray,
                 mean: np.ndarray, scale: np.ndarray, meta: Optional[dict] = None):
        self.features = list(features)
        self.classes = list(classes)
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = np.asarray(intercept, dtype=float)
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.meta = meta or {}

    def predict(self, feats: Mapping[str, float]) -> Prediction:
        x = np.array([feats[f] for f in self.features], dtype=float)
        z = self.coef @ ((x - self.mean) / self.scale) + self.intercept
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        probs = {c: float(v) for c, v in zip(self.classes, p)}
        return Prediction(self.classes[int(np.argmax(p))], probs)

    # -- (de)serialisation ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "features": self.features,
            "classes": self.classes,
            "coef": np.round(self.coef, 6).tolist(),
            "intercept": np.round(self.intercept, 6).tolist(),
            "mean": np.round(self.mean, 6).tolist(),
            "scale": np.round(self.scale, 6).tolist(),
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL) -> "ReuseModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["features"], d["classes"], d["coef"], d["intercept"],
                   d["mean"], d["scale"], d.get("meta"))

    @classmethod
    def from_sklearn(cls, clf, scaler, features: List[str], meta=None) -> "ReuseModel":
        coef, intercept = clf.coef_, clf.intercept_
        if coef.shape[0] == 1:  # binary LR stores one row; expand to two classes
            coef = np.vstack([-coef[0] / 2, coef[0] / 2])
            intercept = np.array([-intercept[0] / 2, intercept[0] / 2])
        return cls(features, [str(c) for c in clf.classes_], coef, intercept,
                   scaler.mean_, scaler.scale_, meta)
