"""
╔══════════════════════════════════════════════════════════════╗
║              PLAGIARISM CHECKER — CLI Interface              ║
║         Beautiful terminal UI with coloured output           ║
╚══════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import sys
import json
import argparse
import textwrap
from pathlib import Path
from typing import Optional

from .engine import PlagiarismChecker, CheckResult, DocumentPair


# ─── ANSI colour helpers ───────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    ORANGE  = "\033[33m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BG_RED  = "\033[41m"
    BG_GREEN= "\033[42m"

def colour_score(score: float) -> str:
    pct = score * 100
    if pct >= 90:
        return f"{C.BOLD}{C.RED}{pct:.1f}%{C.RESET}"
    elif pct >= 75:
        return f"{C.BOLD}{C.ORANGE}{pct:.1f}%{C.RESET}"
    elif pct >= 50:
        return f"{C.BOLD}{C.YELLOW}{pct:.1f}%{C.RESET}"
    elif pct >= 30:
        return f"{C.BOLD}{C.CYAN}{pct:.1f}%{C.RESET}"
    else:
        return f"{C.BOLD}{C.GREEN}{pct:.1f}%{C.RESET}"

VERDICT_STYLE = {
    "HIGHLY_PLAGIARISED": (C.BG_RED + C.WHITE + C.BOLD, "🔴 HIGHLY PLAGIARISED"),
    "LIKELY_PLAGIARISED":  (C.RED + C.BOLD,              "🟠 LIKELY PLAGIARISED"),
    "SUSPICIOUS":          (C.YELLOW + C.BOLD,           "🟡 SUSPICIOUS"),
    "MINOR_SIMILARITY":    (C.CYAN + C.BOLD,             "🟢 MINOR SIMILARITY"),
    "ORIGINAL":            (C.GREEN + C.BOLD,            "✅ ORIGINAL"),
}


# ─── Banner ────────────────────────────────────────────────────────
BANNER = f"""
{C.CYAN}{C.BOLD}
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██████╗  ██╗      █████╗  ██████╗     ██████╗██╗  ██╗███████╗ ║
║   ██╔══██╗ ██║     ██╔══██╗██╔════╝    ██╔════╝██║  ██║██╔════╝ ║
║   ██████╔╝ ██║     ███████║██║  ███╗   ██║     ███████║█████╗   ║
║   ██╔═══╝  ██║     ██╔══██║██║   ██║   ██║     ██╔══██║██╔══╝   ║
║   ██║      ███████╗██║  ██║╚██████╔╝   ╚██████╗██║  ██║███████╗ ║
║   ╚═╝      ╚══════╝╚═╝  ╚═╝ ╚═════╝     ╚═════╝╚═╝  ╚═╝╚══════╝ ║
║                                                                  ║
║        {C.MAGENTA}AI-Powered Plagiarism Detection Engine v2.0{C.CYAN}            ║
║          {C.DIM}TF-IDF · Cosine Similarity · N-gram · LSH{C.CYAN}            ║
╚══════════════════════════════════════════════════════════════════╝
{C.RESET}"""


# ─── Argument Parser ───────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plagcheck",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            ╔═══════════════════════════════════════╗
            ║  Plagiarism Checker Pro — CLI v2.0    ║
            ╚═══════════════════════════════════════╝
            Detect plagiarism using TF-IDF, Cosine Similarity,
            Jaccard index, and n-gram shingle matching.
        """),
        epilog=textwrap.dedent("""\
            Examples:
              plagcheck -d ./essays
              plagcheck -d ./essays --threshold 0.6 --output report.json
              plagcheck -f paper1.txt paper2.txt
              plagcheck -q submission.txt -d ./corpus
        """),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "-d", "--directory",
        metavar="DIR",
        help="Directory of .txt / .md files to compare pairwise",
    )
    mode.add_argument(
        "-f", "--files",
        metavar="FILE",
        nargs=2,
        help="Compare exactly two files",
    )
    mode.add_argument(
        "-q", "--query",
        metavar="FILE",
        help="Check one file against a corpus directory (use with -d)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        metavar="T",
        help="Similarity threshold to flag plagiarism (0–1, default: 0.75)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Save JSON report to file",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--show-phrases",
        action="store_true",
        help="Show matching phrases in report",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        metavar="S",
        help="Only show pairs above this combined score (0–1)",
    )
    parser.add_argument(
        "--stemming",
        action="store_true",
        help="Enable word stemming (slower, more accurate)",
    )
    parser.add_argument(
        "--weights",
        metavar="C:J:N",
        default="0.5:0.25:0.25",
        help="Algorithm weights as cosine:jaccard:ngram (default: 0.5:0.25:0.25)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Plagiarism Checker Pro v2.0.0",
    )
    return parser


# ─── Pretty Report Printer ─────────────────────────────────────────
def print_summary_header(result: CheckResult) -> None:
    W = 68
    print("\n" + f"{C.CYAN}{'═' * W}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}{'  📊  PLAGIARISM ANALYSIS REPORT'.center(W)}{C.RESET}")
    print(f"{C.CYAN}{'═' * W}{C.RESET}")
    print(f"  {C.BOLD}Documents Scanned :{C.RESET} {C.WHITE}{result.total_documents}{C.RESET}")
    print(f"  {C.BOLD}Pairs Compared    :{C.RESET} {C.WHITE}{result.total_pairs}{C.RESET}")

    flag_colour = C.RED if result.plagiarised_pairs > 0 else C.GREEN
    print(f"  {C.BOLD}Flagged Pairs     :{C.RESET} {flag_colour}{C.BOLD}{result.plagiarised_pairs}{C.RESET}")

    rate_colour = C.RED if result.plagiarism_rate > 50 else (C.YELLOW if result.plagiarism_rate > 20 else C.GREEN)
    print(f"  {C.BOLD}Plagiarism Rate   :{C.RESET} {rate_colour}{C.BOLD}{result.plagiarism_rate:.1f}%{C.RESET}")
    print(f"  {C.BOLD}Time Taken        :{C.RESET} {C.WHITE}{result.duration:.3f}s{C.RESET}")
    print(f"  {C.BOLD}Timestamp         :{C.RESET} {C.DIM}{result.timestamp}{C.RESET}")
    print(f"{C.CYAN}{'─' * W}{C.RESET}")


def print_pair(pair: DocumentPair, show_phrases: bool = False) -> None:
    style, label = VERDICT_STYLE.get(
        pair.verdict, (C.WHITE, pair.verdict)
    )
    print(f"\n  {style}{label}{C.RESET}")
    print(f"  {C.BOLD}📄 {pair.doc1}{C.RESET}  {C.DIM}↔{C.RESET}  {C.BOLD}{pair.doc2}{C.RESET}")

    # Score bars
    def bar(score: float, width: int = 20) -> str:
        filled = int(score * width)
        empty  = width - filled
        c = C.RED if score >= 0.75 else (C.YELLOW if score >= 0.50 else C.GREEN)
        return f"{c}{'█' * filled}{'░' * empty}{C.RESET}"

    print(f"     Cosine    {bar(pair.cosine_score)}  {colour_score(pair.cosine_score)}")
    print(f"     Jaccard   {bar(pair.jaccard_score)}  {colour_score(pair.jaccard_score)}")
    print(f"     N-gram    {bar(pair.ngram_score)}  {colour_score(pair.ngram_score)}")
    print(f"     {C.BOLD}Combined  {bar(pair.combined_score)}  {colour_score(pair.combined_score)}{C.RESET}")
    print(f"     Words shared: {C.CYAN}{len(pair.word_overlap)}{C.RESET}  |  Time: {C.DIM}{pair.execution_time*1000:.1f}ms{C.RESET}")

    if show_phrases and pair.highlighted_sections:
        print(f"     {C.MAGENTA}Matching phrases:{C.RESET}")
        for phrase in pair.highlighted_sections[:3]:
            print(f"       {C.DIM}» \"{phrase}\"{C.RESET}")


def print_full_report(
    result: CheckResult,
    show_phrases: bool = False,
    min_score: float = 0.0,
) -> None:
    print_summary_header(result)

    sorted_pairs = sorted(result.pairs, key=lambda p: p.combined_score, reverse=True)
    filtered = [p for p in sorted_pairs if p.combined_score >= min_score]

    if not filtered:
        print(f"\n  {C.GREEN}{C.BOLD}✅  All documents appear to be original!{C.RESET}\n")
    else:
        for pair in filtered:
            print_pair(pair, show_phrases=show_phrases)

    W = 68
    print(f"\n{C.CYAN}{'═' * W}{C.RESET}\n")


# ─── JSON Export ───────────────────────────────────────────────────
def export_json(result: CheckResult, path: str) -> None:
    data = {
        "summary": {
            "total_documents": result.total_documents,
            "total_pairs":     result.total_pairs,
            "flagged_pairs":   result.plagiarised_pairs,
            "plagiarism_rate": round(result.plagiarism_rate, 2),
            "duration_seconds":round(result.duration, 4),
            "timestamp":       result.timestamp,
        },
        "pairs": [p.to_dict() for p in result.pairs],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  {C.GREEN}✅  JSON report saved → {path}{C.RESET}")


# ─── Parse Weights ─────────────────────────────────────────────────
def parse_weights(w_str: str) -> dict:
    try:
        parts = [float(x) for x in w_str.split(":")]
        assert len(parts) == 3
        total = sum(parts)
        return {
            "cosine":  parts[0] / total,
            "jaccard": parts[1] / total,
            "ngram":   parts[2] / total,
        }
    except Exception:
        print(f"{C.YELLOW}⚠  Invalid weights '{w_str}', using defaults.{C.RESET}")
        return {"cosine": 0.5, "jaccard": 0.25, "ngram": 0.25}


# ─── Main Entry ───────────────────────────────────────────────────
def main(argv=None) -> int:
    print(BANNER)
    parser = build_parser()
    args = parser.parse_args(argv)

    weights = parse_weights(args.weights)
    checker = PlagiarismChecker(
        threshold=args.threshold,
        weights=weights,
        use_stemming=args.stemming,
    )

    # ── Mode: two files ─────────────────────────
    if args.files:
        f1, f2 = Path(args.files[0]), Path(args.files[1])
        for fp in (f1, f2):
            if not fp.exists():
                print(f"{C.RED}Error: file not found: {fp}{C.RESET}")
                return 1
        checker.load_file(f1)
        checker.load_file(f2)
        result = checker.check_all(progress=not args.no_progress)

    # ── Mode: query vs corpus ────────────────────
    elif args.query:
        q = Path(args.query)
        if not q.exists():
            print(f"{C.RED}Error: query file not found: {q}{C.RESET}")
            return 1
        if not args.directory:
            print(f"{C.RED}Error: --query requires --directory{C.RESET}")
            return 1
        checker.load_directory(args.directory)
        query_text = q.read_text(encoding="utf-8", errors="replace")
        pairs = checker.check_against_corpus(query_text, q.name)

        # Wrap in CheckResult for unified display
        import datetime
        flagged = sum(1 for p in pairs if p.combined_score >= args.threshold)
        result = CheckResult(
            total_documents=len(checker.document_names) + 1,
            total_pairs=len(pairs),
            plagiarised_pairs=flagged,
            pairs=pairs,
            timestamp=datetime.datetime.now().isoformat(),
            duration=sum(p.execution_time for p in pairs),
        )

    # ── Mode: directory ──────────────────────────
    else:
        loaded = checker.load_directory(args.directory)
        if loaded < 2:
            print(f"{C.YELLOW}⚠  Need at least 2 documents. Found: {loaded}{C.RESET}")
            return 1
        result = checker.check_all(progress=not args.no_progress)

    # ── Print report ─────────────────────────────
    print_full_report(
        result,
        show_phrases=args.show_phrases,
        min_score=args.min_score,
    )

    # ── Save JSON ────────────────────────────────
    if args.output:
        export_json(result, args.output)

    return 0 if result.plagiarised_pairs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
