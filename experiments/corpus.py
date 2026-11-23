"""Loader for the Clough & Stevenson (2011) corpus of plagiarised short answers.

The corpus has 5 short Wikipedia source texts (tasks a-e, computer science
topics) and 95 answers written by 19 participants. Each answer was written in
one of four ways:

    cut    copy-paste from the source
    light  light revision of the source
    heavy  heavy revision / paraphrase of the source
    non    written from the participant's own knowledge

Get it from the original page (or any mirror that keeps file_information.csv):
    https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html
and unzip so the folder holds file_information.csv plus the .txt files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from plagiarism_checker.features import Document

CATEGORIES = ("cut", "light", "heavy", "non")


@dataclass
class Answer:
    doc: Document
    task: str
    category: str
    participant: str


@dataclass
class Corpus:
    sources: Dict[str, Document]   # task -> source text
    answers: List[Answer]

    @property
    def all_docs(self) -> List[Document]:
        return list(self.sources.values()) + [a.doc for a in self.answers]


def load(folder: str | Path) -> Corpus:
    folder = Path(folder)
    info = folder / "file_information.csv"
    if not info.exists():
        raise FileNotFoundError(
            f"{info} not found. Download the corpus first (see experiments/README.md)."
        )

    sources, answers = {}, []
    with open(info, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name, task, cat = row["File"].strip(), row["Task"].strip(), row["Category"].strip()
            # a few files in the distribution aren't valid utf-8
            text = (folder / name).read_bytes().decode("utf-8", errors="replace")
            doc = Document(name, text)
            if cat == "orig":
                sources[task] = doc
            else:
                # file names look like g0pA_taska.txt -> participant g0pA
                answers.append(Answer(doc, task, cat, name.split("_")[0]))

    missing = {a.task for a in answers} - set(sources)
    if missing:
        raise ValueError(f"no source text for tasks: {sorted(missing)}")
    return Corpus(sources, answers)
