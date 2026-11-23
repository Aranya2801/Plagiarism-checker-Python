"""Self-contained HTML report with matched passages highlighted side by side."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Dict, Sequence, Tuple

from . import __version__
from .checker import CheckResult

CSS = """
body { font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: #1f2328;
       max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
h1 { font-size: 1.4rem; margin-bottom: .2rem; }
.sub { color: #59636e; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }
th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #d1d9e0; }
th { font-weight: 600; background: #f6f8fa; }
td.num { font-variant-numeric: tabular-nums; text-align: right; }
.flag { color: #b42318; font-weight: 600; }
.pair { border: 1px solid #d1d9e0; border-radius: 6px; margin: 1.5rem 0; }
.pair > header { padding: .6rem .9rem; background: #f6f8fa; border-bottom: 1px solid #d1d9e0; }
.cols { display: grid; grid-template-columns: 1fr 1fr; }
.cols > div { padding: .8rem .9rem; white-space: pre-wrap; max-height: 420px; overflow: auto;
              font-size: 14px; }
.cols > div + div { border-left: 1px solid #d1d9e0; }
.cols h3 { font-size: .9rem; margin: 0 0 .5rem; color: #59636e; white-space: normal; }
mark { background: #ffe58f; padding: 0 1px; }
.feat { font-size: 13px; color: #59636e; }
"""


def _highlight(text: str, spans: Sequence[Tuple[int, int]]) -> str:
    out, pos = [], 0
    for s, e in sorted(spans):
        if s < pos:
            s = pos
        if s >= e:
            continue
        out.append(escape(text[pos:s]))
        out.append(f"<mark>{escape(text[s:e])}</mark>")
        pos = e
    out.append(escape(text[pos:]))
    return "".join(out)


def _coverage(text: str, spans) -> float:
    return sum(e - s for s, e in spans) / max(len(text), 1)


def render_html(result: CheckResult, texts: Dict[str, str], max_details: int = 20,
                title: str = "Text reuse report") -> str:
    rows = []
    for m in result.matches:
        cls = ' class="flag"' if m.flagged else ""
        rows.append(
            f"<tr><td>{escape(m.doc1)}</td><td>{escape(m.doc2)}</td>"
            f"<td class='num'{cls}>{m.score:.2f}</td><td>{escape(m.pretty_label)}</td>"
            f"<td class='num'>{m.features['containment_3']:.2f}</td>"
            f"<td class='num'>{m.features['winnow']:.2f}</td></tr>"
        )

    details = []
    shown = [m for m in result.matches if m.flagged][:max_details]
    for m in shown:
        t1, t2 = texts.get(m.doc1, ""), texts.get(m.doc2, "")
        feats = ", ".join(f"{k} {v:.2f}" for k, v in m.features.items())
        details.append(f"""
<section class="pair">
  <header><strong>{escape(m.doc1)}</strong> &harr; <strong>{escape(m.doc2)}</strong>
    &nbsp;&middot;&nbsp; P(reuse) {m.score:.2f} &nbsp;&middot;&nbsp; {escape(m.pretty_label)}
    <div class="feat">{escape(feats)}</div></header>
  <div class="cols">
    <div><h3>{escape(m.doc1)} &middot; {_coverage(t1, m.spans1):.0%} highlighted</h3>{_highlight(t1, m.spans1)}</div>
    <div><h3>{escape(m.doc2)} &middot; {_coverage(t2, m.spans2):.0%} highlighted</h3>{_highlight(t2, m.spans2)}</div>
  </div>
</section>""")

    if not shown:
        details.append("<p>No pair reached the threshold.</p>")

    lsh_note = ""
    if result.n_pairs_scored < result.n_pairs_total:
        lsh_note = (f" ({result.n_pairs_scored} of {result.n_pairs_total} pairs scored after "
                    f"MinHash/LSH filtering)")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title><style>{CSS}</style></head>
<body>
<h1>{escape(title)}</h1>
<p class="sub">{result.n_documents} documents, {len(result.matches)} pairs{lsh_note},
{len(result.flagged())} flagged at P(reuse) &ge; {result.threshold:.2f}.
Generated {escape(result.timestamp)} by plagiarism_checker {__version__}.</p>

<h2>Flagged pairs</h2>
<p class="feat">Highlighted text is covered by shared winnowing fingerprints (runs of 7+ matching words,
small gaps merged). Paraphrased reuse can score high without much highlighting.</p>
{''.join(details)}

<h2>All pairs</h2>
<table><thead><tr><th>document</th><th>document</th><th>P(reuse)</th><th>type</th>
<th>3-gram containment</th><th>winnowing</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>
"""


def write_html(result: CheckResult, texts: Dict[str, str], path: str | Path, **kw) -> Path:
    path = Path(path)
    path.write_text(render_html(result, texts, **kw), encoding="utf-8")
    return path
