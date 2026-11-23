"""Command line interface.

    plagcheck essays/                         every pair in a folder
    plagcheck a.txt b.txt c.txt               just these files
    plagcheck submission.txt --against refs/  one file vs. a collection
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .checker import CheckResult, Match, PlagiarismChecker
from .io import read_document


class Style:
    def __init__(self, enabled: bool):
        if enabled and os.name == "nt":
            os.system("")  # turns on ANSI escape handling in the Windows console
        self.on = enabled

    def _c(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def bold(self, s): return self._c("1", s)
    def dim(self, s): return self._c("2", s)
    def red(self, s): return self._c("31;1", s)
    def yellow(self, s): return self._c("33", s)
    def green(self, s): return self._c("32", s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plagcheck",
        description="Find reused text between documents (.txt .md .rst .tex, .pdf and .docx with extras).",
        epilog="examples:\n"
               "  plagcheck samples/\n"
               "  plagcheck essay1.txt essay2.txt\n"
               "  plagcheck submission.txt --against reference/ --html report.html",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("paths", nargs="+", help="a folder, two or more files, or one file with --against")
    p.add_argument("--against", metavar="DIR", help="compare the single input file with every document in DIR")
    p.add_argument("-t", "--threshold", type=float, default=0.5,
                   help="flag pairs with P(reuse) at or above this (default 0.5)")
    p.add_argument("--top", type=int, default=10, help="how many pairs to print (default 10, 0 = all)")
    p.add_argument("-r", "--recursive", action="store_true", help="include subfolders")
    p.add_argument("--lsh", choices=("auto", "on", "off"), default="auto",
                   help="MinHash pre-filter for big collections (auto = on from 200 docs)")
    p.add_argument("--stem", action="store_true", help="stem words before comparing")
    p.add_argument("--html", metavar="FILE", help="write an HTML report with highlighted matches")
    p.add_argument("--json", metavar="FILE", help="write all scores as JSON")
    p.add_argument("--fail-on-match", action="store_true",
                   help="exit with status 2 if anything is flagged (handy in scripts/CI)")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _progress(style: Style, quiet: bool):
    if quiet or not sys.stderr.isatty():
        return None

    def cb(done, total):
        width = 30
        n = int(width * done / max(total, 1))
        sys.stderr.write(f"\r  scoring [{'#' * n}{'.' * (width - n)}] {done}/{total}")
        if done == total:
            sys.stderr.write("\r" + " " * (width + 30) + "\r")
        sys.stderr.flush()
    return cb


def _print_match(m: Match, st: Style) -> None:
    score = f"{m.score:.2f}"
    score = st.red(score) if m.flagged else (st.yellow(score) if m.score >= 0.3 else st.green(score))
    print(f"  {score}  {m.doc1}  <->  {m.doc2}")
    f = m.features
    print(st.dim(f"        {m.pretty_label:<15} 3-gram containment {f['containment_3']:.2f}"
                 f"  winnowing {f['winnow']:.2f}  LCS {f['lcs']:.2f}"))


def print_result(res: CheckResult, top: int, st: Style) -> None:
    flagged = res.flagged()
    scored = f"{res.n_pairs_scored} pair{'s' if res.n_pairs_scored != 1 else ''} scored"
    if res.n_pairs_scored < res.n_pairs_total:
        scored += f" (of {res.n_pairs_total}, rest skipped by LSH)"
    print(f"\n{st.bold('plagcheck')} {res.n_documents} documents, {scored}, {res.seconds:.2f}s")
    if flagged:
        print(st.red(f"{len(flagged)} pair(s) flagged at P(reuse) >= {res.threshold:.2f}"))
    else:
        print(st.green(f"nothing flagged at P(reuse) >= {res.threshold:.2f}"))
    print()
    shown = res.matches if top == 0 else res.matches[:top]
    for m in shown:
        _print_match(m, st)
    if len(shown) < len(res.matches):
        print(st.dim(f"\n  ... {len(res.matches) - len(shown)} more (use --top 0 to see all)"))
    print()


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    st = Style(not args.no_color and sys.stdout.isatty() and "NO_COLOR" not in os.environ)

    paths = [Path(p) for p in args.paths]
    for p in paths:
        if not p.exists():
            print(f"error: {p} does not exist", file=sys.stderr)
            return 1

    pc = PlagiarismChecker(threshold=args.threshold, stem=args.stem,
                           use_lsh={"auto": "auto", "on": True, "off": False}[args.lsh])
    progress = _progress(st, args.quiet)

    try:
        if args.against:
            if len(paths) != 1 or not paths[0].is_file():
                print("error: --against takes exactly one input file", file=sys.stderr)
                return 1
            n = pc.add_directory(args.against, recursive=args.recursive)
            if n == 0:
                print(f"error: no readable documents in {args.against}", file=sys.stderr)
                return 1
            query = paths[0]
            # if the submission sits inside the reference folder, don't compare it with itself
            for name in list(pc.docs):
                if (Path(args.against) / name).resolve() == query.resolve():
                    del pc.docs[name]
            from .features import Document
            qdoc = Document(query.name, read_document(query), pc.pre)
            res = pc.check_against(qdoc, progress=progress)
            texts = {qdoc.name: qdoc.text}
        else:
            if len(paths) == 1 and paths[0].is_dir():
                pc.add_directory(paths[0], recursive=args.recursive)
            elif all(p.is_file() for p in paths) and len(paths) >= 2:
                for p in paths:
                    name = p.name if p.name not in pc.docs else str(p)
                    pc.add_file(p, name=name)
            else:
                print("error: give a folder, two or more files, or one file with --against DIR",
                      file=sys.stderr)
                return 1
            if len(pc.docs) < 2:
                print("error: need at least two documents with text", file=sys.stderr)
                return 1
            res = pc.check_all(progress=progress)
            texts = {}
    except ImportError as e:  # missing optional PDF / docx dependency
        print(f"error: {e}", file=sys.stderr)
        return 1

    texts.update({n: d.text for n, d in pc.docs.items()})
    print_result(res, args.top, st)

    if args.json:
        Path(args.json).write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
        print(f"JSON written to {args.json}")
    if args.html:
        from .report import write_html
        write_html(res, texts, args.html)
        print(f"HTML report written to {args.html}")

    return 2 if (args.fail_on_match and res.flagged()) else 0


if __name__ == "__main__":
    sys.exit(main())
